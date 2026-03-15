from pathlib import Path
from dataloader import (
    MahjongDiscardDataset, MahjongCallDataset,
    build_dataset,
)
import kagglehub
from model import SmallMahjongResNet
import torch
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader
from collections import Counter

# ============================================================
# Config
# ============================================================

DATASET_HANDLE = "shokanekolouis/tenhou-to-mjai"
DATASET_PATH = Path(kagglehub.dataset_download(DATASET_HANDLE))
print("DATASET_PATH:", DATASET_PATH)

YEARS = [str(y) for y in range(2023, 2025)]
MAX_FILES = 5000
MAX_DISCARD_SAMPLES = 1_000_000
MAX_CALL_SAMPLES = 500_000
BATCH_SIZE = 256
EPOCHS = 50
LR = 3e-4                        # lower base LR (was 1e-3)
WARMUP_EPOCHS = 5                 # warmup before cosine decay
LABEL_SMOOTHING = 0.05            # reduced (was 0.1)
CALL_LOSS_WEIGHT = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = CHECKPOINT_DIR / "best.pt"
RESUME_PATH = CHECKPOINT_DIR / "latest.pt"

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


# ============================================================
# Masked prediction (mask only at inference, NOT in loss)
# ============================================================

def masked_prediction(logits, mask):
    """Apply mask only for argmax prediction, not for loss computation."""
    masked = logits.clone()
    masked[~mask] = -1e9
    return masked


# ============================================================
# LR scheduler with linear warmup + cosine decay
# ============================================================

def get_lr(epoch, warmup_epochs, max_epochs, base_lr):
    if epoch <= warmup_epochs:
        return base_lr * epoch / max(warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
    return base_lr * 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())


# ============================================================
# Class weights for call (sqrt-dampened to avoid instability)
# ============================================================

def compute_call_weights(samples):
    labels = [s[1] for s in samples]
    counts = Counter(labels)
    total = len(labels)
    num_classes = 3

    weights = torch.ones(num_classes)
    for cls in range(num_classes):
        if counts[cls] > 0:
            raw = total / (num_classes * counts[cls])
            weights[cls] = raw ** 0.5  # sqrt dampening

    print(f"Call label distribution: {dict(counts)}")
    print(f"Call class weights (sqrt-dampened): {[f'{w:.3f}' for w in weights.tolist()]}")
    return weights


# ============================================================
# Evaluate
# ============================================================

def evaluate_discard(model, loader):
    model.eval()
    total_loss = 0.0
    total = 0
    correct_top1 = 0
    correct_top3 = 0

    with torch.no_grad():
        for x, mask, y in loader:
            x, mask, y = x.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
            logits = model.forward_discard(x)

            loss = F.cross_entropy(logits, y, label_smoothing=LABEL_SMOOTHING)
            total_loss += loss.item() * x.size(0)
            total += x.size(0)

            pred_logits = masked_prediction(logits, mask)
            pred = pred_logits.argmax(dim=1)
            correct_top1 += (pred == y).sum().item()

            _, top3 = pred_logits.topk(3, dim=1)
            correct_top3 += (top3 == y.unsqueeze(1)).any(dim=1).sum().item()

    n = max(total, 1)
    return total_loss / n, correct_top1 / n, correct_top3 / n


def evaluate_call(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    class_correct = [0, 0, 0]
    class_total = [0, 0, 0]

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model.forward_call(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            total += x.size(0)

            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()

            for cls in range(3):
                cls_mask = (y == cls)
                class_total[cls] += cls_mask.sum().item()
                class_correct[cls] += (pred[cls_mask] == cls).sum().item()

    n = max(total, 1)
    label_names = {0: "pass", 1: "pon", 2: "chi"}
    class_acc = {}
    for cls in range(3):
        class_acc[label_names[cls]] = class_correct[cls] / max(class_total[cls], 1)

    return total_loss / n, correct / n, class_acc


# ============================================================
# Train
# ============================================================

def train():
    # --- data ---
    discard_samples, call_samples = build_dataset(
        DATASET_PATH, YEARS, MAX_FILES,
        MAX_DISCARD_SAMPLES, MAX_CALL_SAMPLES,
    )

    if len(discard_samples) < 100:
        print("Too few discard samples.")
        return

    # split discard
    random.shuffle(discard_samples)
    d_split = int(0.85 * len(discard_samples))
    d_train = MahjongDiscardDataset(discard_samples[:d_split])
    d_val = MahjongDiscardDataset(discard_samples[d_split:])

    d_train_loader = DataLoader(d_train, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=2, pin_memory=True)
    d_val_loader = DataLoader(d_val, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    # split call
    has_call = len(call_samples) >= 100
    if has_call:
        random.shuffle(call_samples)
        c_split = int(0.85 * len(call_samples))
        c_train = MahjongCallDataset(call_samples[:c_split])
        c_val = MahjongCallDataset(call_samples[c_split:])

        c_train_loader = DataLoader(c_train, batch_size=BATCH_SIZE, shuffle=True,
                                    num_workers=2, pin_memory=True)
        c_val_loader = DataLoader(c_val, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=2, pin_memory=True)

        call_weights = compute_call_weights(call_samples[:c_split]).to(DEVICE)
    else:
        print("Not enough call samples; training discard only.")

    # --- model ---
    model = SmallMahjongResNet(in_channels=16).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params:       {total_params:,}")
    print(f"Discard train/val:  {len(d_train):,} / {len(d_val):,}")
    if has_call:
        print(f"Call train/val:     {len(c_train):,} / {len(c_val):,}")

    call_criterion = (
        torch.nn.CrossEntropyLoss(weight=call_weights, label_smoothing=LABEL_SMOOTHING)
        if has_call else None
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # --- resume ---
    start_epoch = 1
    best_val_acc = 0.0

    if RESUME_PATH and RESUME_PATH.exists():
        print(f"Resuming from {RESUME_PATH}")
        ckpt = torch.load(RESUME_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"  Resumed at epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")

    print(f"Device: {DEVICE}")
    print(f"LR: {LR}, Warmup: {WARMUP_EPOCHS} epochs, Total: {EPOCHS} epochs")
    print()

    # --- training loop ---
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()

        # update learning rate
        current_lr = get_lr(epoch, WARMUP_EPOCHS, EPOCHS, LR)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # ---- discard training ----
        d_loss_sum, d_total, d_correct = 0.0, 0, 0
        for x, mask, y in d_train_loader:
            x, mask, y = x.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            logits = model.forward_discard(x)

            # KEY FIX: compute loss on raw logits, NOT masked logits
            # the target tile is always in hand, so raw CE works fine
            # masking with -1e9 was causing the loss to explode to 68 million
            loss = F.cross_entropy(logits, y, label_smoothing=LABEL_SMOOTHING)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            d_loss_sum += loss.item() * x.size(0)
            d_total += x.size(0)

            # accuracy still uses mask (only count valid predictions)
            with torch.no_grad():
                pred = masked_prediction(logits, mask).argmax(dim=1)
                d_correct += (pred == y).sum().item()

        # ---- call training ----
        c_loss_sum, c_total, c_correct = 0.0, 0, 0
        if has_call:
            for x, y in c_train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)

                optimizer.zero_grad()
                logits = model.forward_call(x)
                loss = call_criterion(logits, y) * CALL_LOSS_WEIGHT
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                c_loss_sum += loss.item() * x.size(0)
                c_correct += (logits.argmax(dim=1) == y).sum().item()
                c_total += x.size(0)

        # ---- evaluate ----
        d_val_loss, d_val_top1, d_val_top3 = evaluate_discard(model, d_val_loader)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | lr={current_lr:.6f} | "
            f"discard: loss={d_loss_sum/d_total:.4f} acc={d_correct/d_total:.4f} "
            f"val_loss={d_val_loss:.4f} val_top1={d_val_top1:.4f} val_top3={d_val_top3:.4f}"
        )

        if has_call and c_total > 0:
            c_val_loss, c_val_acc, c_class_acc = evaluate_call(model, c_val_loader, call_criterion)
            print(
                f"         call:    loss={c_loss_sum/c_total:.4f} acc={c_correct/c_total:.4f} "
                f"val_acc={c_val_acc:.4f} "
                f"[pass={c_class_acc['pass']:.3f} pon={c_class_acc['pon']:.3f} chi={c_class_acc['chi']:.3f}]"
            )

        # save best model
        if d_val_top1 > best_val_acc:
            best_val_acc = d_val_top1
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> New best model! val_top1={d_val_top1:.4f}")

        # save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
        }, CHECKPOINT_DIR / "latest.pt")

    print(f"\nDone. Best discard val_top1={best_val_acc:.4f}")
    print(f"Model saved at: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    train()