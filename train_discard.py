from pathlib import Path
from dataloader import (
    MahjongDiscardDataset,
    load_processed_discard_dataset,
)
from model import MahjongResNet
import torch
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader

# ============================================================
# Config
# ============================================================

BATCH_SIZE = 256
EPOCHS = 25
LR = 3e-4
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

BEST_DISCARD_MODEL_PATH = CHECKPOINT_DIR / "best_discard.pt"
DISCARD_RESUME_PATH = CHECKPOINT_DIR / "latest_discard.pt"

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# Optional for reproducibility; may reduce speed slightly
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def masked_prediction(logits, mask):
    return logits.masked_fill(~mask, -1e9)


def split_packed_dict(packed, train_ratio=0.85, seed=42):
    n = packed["y"].shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    split = int(train_ratio * n)

    train_idx = perm[:split]
    val_idx = perm[split:]

    train_packed = {k: v[train_idx] for k, v in packed.items()}
    val_packed = {k: v[val_idx] for k, v in packed.items()}
    return train_packed, val_packed


def get_current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def evaluate_discard_branch(discard_model, loader):
    discard_model.eval()
    total_loss, total, correct_top1, correct_top3 = 0.0, 0, 0, 0

    with torch.no_grad():
        for x, mask, hist, hist_mask, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            hist = hist.to(DEVICE, non_blocking=True)
            hist_mask = hist_mask.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            discard_logits, _ = discard_model(x, hist, hist_mask)
            logits = masked_prediction(discard_logits, mask)

            loss = F.cross_entropy(logits, y)
            total_loss += loss.item() * x.size(0)
            total += x.size(0)

            pred = logits.argmax(dim=1)
            correct_top1 += (pred == y).sum().item()

            _, top3 = logits.topk(3, dim=1)
            correct_top3 += (top3 == y.unsqueeze(1)).any(dim=1).sum().item()

    n = max(total, 1)
    return total_loss / n, correct_top1 / n, correct_top3 / n


def train_discard():
    discard_samples, meta = load_processed_discard_dataset("./processed_dataset")
    print(meta)

    d_train_packed, d_val_packed = split_packed_dict(
        discard_samples, train_ratio=0.85, seed=SEED
    )
    d_train = MahjongDiscardDataset(d_train_packed)
    d_val = MahjongDiscardDataset(d_val_packed)

    d_train_loader = DataLoader(
        d_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )
    d_val_loader = DataLoader(
        d_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    wrapper = MahjongResNet().to(DEVICE)
    discard_model = wrapper.discard_model

    total_params = sum(p.numel() for p in discard_model.parameters())
    print(f"\nDiscard params:     {total_params:,}")
    print(f"Discard train/val:  {len(d_train):,} / {len(d_val):,}")

    optimizer = torch.optim.AdamW(
        discard_model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # Reduce LR when val_top1 stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        threshold=1e-4,
        threshold_mode="abs",
        min_lr=1e-6,
    )

    start_epoch = 1
    best_val_top1 = 0.0

    if DISCARD_RESUME_PATH.exists():
        print(f"Resuming discard from {DISCARD_RESUME_PATH}")
        ckpt = torch.load(DISCARD_RESUME_PATH, map_location=DEVICE)

        discard_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Resume scheduler too if present
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        start_epoch = ckpt["epoch"] + 1
        best_val_top1 = ckpt.get("best_val_top1", 0.0)

        print(f"  Resumed at epoch {start_epoch}, best_val_top1={best_val_top1:.4f}")
        print(f"  Current LR after resume: {get_current_lr(optimizer):.6g}")

    print(f"\nDevice: {DEVICE}")
    print(f"Discard LR: {LR}, Weight Decay: {WEIGHT_DECAY}, Epochs: {EPOCHS}\n")

    for epoch in range(start_epoch, EPOCHS + 1):
        discard_model.train()

        train_loss_sum, train_total, train_correct = 0.0, 0, 0

        for x, mask, hist, hist_mask, y in d_train_loader:
            x = x.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            hist = hist.to(DEVICE, non_blocking=True)
            hist_mask = hist_mask.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            discard_logits, _ = discard_model(x, hist, hist_mask)
            logits = masked_prediction(discard_logits, mask)

            loss = F.cross_entropy(logits, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(discard_model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * x.size(0)
            train_total += x.size(0)

            with torch.no_grad():
                train_correct += (logits.argmax(dim=1) == y).sum().item()

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        val_loss, val_top1, val_top3 = evaluate_discard_branch(discard_model, d_val_loader)

        prev_lr = get_current_lr(optimizer)
        scheduler.step(val_top1)
        new_lr = get_current_lr(optimizer)
        lr_changed = abs(new_lr - prev_lr) > 1e-12

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"discard: loss={train_loss:.4f} "
            f"acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_top1={val_top1:.4f} val_top3={val_top3:.4f} "
            f"lr={new_lr:.6g}"
        )

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            torch.save(discard_model.state_dict(), BEST_DISCARD_MODEL_PATH)
            print(f"  -> New best discard! val_top1={val_top1:.4f}")

        if lr_changed:
            print(f"  -> LR reduced from {prev_lr:.6g} to {new_lr:.6g}")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": discard_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_top1": best_val_top1,
            },
            DISCARD_RESUME_PATH,
        )

    print(f"\nDone. Best discard val_top1={best_val_top1:.4f}")
    print(f"Saved best discard branch to: {BEST_DISCARD_MODEL_PATH}")


if __name__ == "__main__":
    train_discard()
