
from pathlib import Path
from dataloader import (
    MahjongTsumoDataset,
    load_processed_tsumo_dataset,
    TSUMO_ACTION_NAMES,
)
from gamestate import TSUMO_ACTION_TO_IDX
from model import MahjongDecisionNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader

IGNORE_INDEX = -100

# ============================================================
# Config
# ============================================================

BATCH_SIZE = 256
EPOCHS = 30
LR = 3e-4
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

NUM_TSUMO_ACTIONS = len(TSUMO_ACTION_NAMES)

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

BEST_TSUMO_MODEL_PATH = CHECKPOINT_DIR / "best_tsumo.pt"
TSUMO_RESUME_PATH = CHECKPOINT_DIR / "latest_tsumo.pt"

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

tsumo_action_weights = torch.tensor([
    0.2,  # none
    1.0,  # dahai
    1.2,  # reach
    1.4,  # kan
    1.5,  # hora
], dtype=torch.float32, device=DEVICE)


def masked_prediction(logits, mask):
    return logits.masked_fill(~mask, -1e9)


def split_packed_dict(packed, train_ratio=0.85, seed=42):
    n = packed["action_y"].shape[0]
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


def compute_joint_top1(action_logits, tile_logits, action_mask, tile_mask, action_y, tile_y):
    action_pred = masked_prediction(action_logits, action_mask).argmax(dim=1)
    tile_pred = masked_prediction(tile_logits, tile_mask).argmax(dim=1)

    need_tile = tile_y != IGNORE_INDEX
    correct_action = action_pred == action_y
    correct_tile = (~need_tile) | (tile_pred == tile_y)
    return (correct_action & correct_tile).sum().item()


def evaluate_tsumo_branch(model, loader, action_criterion):
    model.eval()
    total_loss, total = 0.0, 0
    correct_joint_top1 = 0
    correct_action_top1 = 0

    # per-action action-only accuracy
    action_correct = torch.zeros(NUM_TSUMO_ACTIONS, device=DEVICE)
    action_total = torch.zeros(NUM_TSUMO_ACTIONS, device=DEVICE)

    # NEW: dahai + tile joint accuracy
    dahai_idx = TSUMO_ACTION_TO_IDX["dahai"]
    dahai_tile_correct = 0
    dahai_tile_total = 0

    with torch.no_grad():
        for x, action_mask, tile_mask, hist, hist_mask, action_y, tile_y in loader:
            x = x.to(DEVICE, non_blocking=True)
            action_mask = action_mask.to(DEVICE, non_blocking=True)
            tile_mask = tile_mask.to(DEVICE, non_blocking=True)
            hist = hist.to(DEVICE, non_blocking=True)
            hist_mask = hist_mask.to(DEVICE, non_blocking=True)
            action_y = action_y.to(DEVICE, non_blocking=True)
            tile_y = tile_y.to(DEVICE, non_blocking=True)

            action_logits, tile_logits, _ = model(x, hist, hist_mask)
            masked_action_logits = masked_prediction(action_logits, action_mask)

            action_loss = action_criterion(masked_action_logits, action_y)

            need_tile = tile_y != IGNORE_INDEX
            if need_tile.any():
                masked_tile_logits = masked_prediction(tile_logits[need_tile], tile_mask[need_tile])
                tile_loss = F.cross_entropy(masked_tile_logits, tile_y[need_tile])
            else:
                tile_loss = torch.tensor(0.0, device=DEVICE)

            loss = action_loss + tile_loss
            total_loss += loss.item() * x.size(0)
            total += x.size(0)

            action_pred = masked_action_logits.argmax(dim=1)
            tile_pred = masked_prediction(tile_logits, tile_mask).argmax(dim=1)

            correct_action_top1 += (action_pred == action_y).sum().item()
            correct_joint_top1 += compute_joint_top1(
                action_logits, tile_logits, action_mask, tile_mask, action_y, tile_y
            )

            # per-action action-only accuracy
            for a in range(NUM_TSUMO_ACTIONS):
                cls_mask = (action_y == a)
                action_total[a] += cls_mask.sum()
                action_correct[a] += (action_pred[cls_mask] == a).sum()

            # NEW: dahai + tile joint accuracy among gold dahai samples
            dahai_mask = (action_y == dahai_idx)
            dahai_tile_total += dahai_mask.sum().item()
            if dahai_mask.any():
                dahai_tile_correct += (
                    (action_pred[dahai_mask] == dahai_idx) &
                    (tile_pred[dahai_mask] == tile_y[dahai_mask])
                ).sum().item()

    n = max(total, 1)

    action_acc = {}
    for a in range(NUM_TSUMO_ACTIONS):
        total_a = action_total[a].item()
        correct_a = action_correct[a].item()
        action_acc[TSUMO_ACTION_NAMES[a]] = correct_a / total_a if total_a > 0 else 0.0

    dahai_tile_acc = dahai_tile_correct / max(dahai_tile_total, 1)

    return (
        total_loss / n,
        correct_joint_top1 / n,
        correct_action_top1 / n,
        action_acc,
        dahai_tile_acc,   # NEW
    )


def train_tsumo_decision():
    tsumo_samples, meta = load_processed_tsumo_dataset("./processed_dataset")
    print(meta)

    train_packed, val_packed = split_packed_dict(tsumo_samples, train_ratio=0.85, seed=SEED)
    train_ds = MahjongTsumoDataset(train_packed)
    val_ds = MahjongTsumoDataset(val_packed)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)

    wrapper = MahjongDecisionNet().to(DEVICE)
    tsumo_model = wrapper.tsumo_model

    total_params = sum(p.numel() for p in tsumo_model.parameters())
    print(f"\nTsumo params:      {total_params:,}")
    print(f"Tsumo train/val:   {len(train_ds):,} / {len(val_ds):,}")

    optimizer = torch.optim.AdamW(tsumo_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        threshold=1e-4,
        threshold_mode="abs",
        min_lr=1e-6,
    )
    action_criterion = nn.CrossEntropyLoss(weight=tsumo_action_weights)

    start_epoch = 1
    best_joint_top1 = 0.0

    if TSUMO_RESUME_PATH.exists():
        print(f"Resuming tsumo branch from {TSUMO_RESUME_PATH}")
        ckpt = torch.load(TSUMO_RESUME_PATH, map_location=DEVICE)

        tsumo_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        start_epoch = ckpt["epoch"] + 1
        best_joint_top1 = ckpt.get("best_joint_top1", 0.0)

        print(f"  Resumed at epoch {start_epoch}, best_joint_top1={best_joint_top1:.4f}")
        print(f"  Current LR after resume: {get_current_lr(optimizer):.6g}")

    print(f"\nDevice: {DEVICE}")
    print(f"Tsumo LR: {LR}, Weight Decay: {WEIGHT_DECAY}, Epochs: {EPOCHS}\n")

    for epoch in range(start_epoch, EPOCHS + 1):
        tsumo_model.train()

        train_loss_sum, train_total = 0.0, 0
        train_correct_joint, train_correct_action = 0, 0

        for x, action_mask, tile_mask, hist, hist_mask, action_y, tile_y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            action_mask = action_mask.to(DEVICE, non_blocking=True)
            tile_mask = tile_mask.to(DEVICE, non_blocking=True)
            hist = hist.to(DEVICE, non_blocking=True)
            hist_mask = hist_mask.to(DEVICE, non_blocking=True)
            action_y = action_y.to(DEVICE, non_blocking=True)
            tile_y = tile_y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            action_logits, tile_logits, _ = tsumo_model(x, hist, hist_mask)
            masked_action_logits = masked_prediction(action_logits, action_mask)

            action_loss = action_criterion(masked_action_logits, action_y)

            need_tile = tile_y != IGNORE_INDEX
            if need_tile.any():
                gold_tile = tile_y[need_tile]  # [N]

                masked_tile_logits = masked_prediction(tile_logits[need_tile], tile_mask[need_tile])
                tile_loss = F.cross_entropy(masked_tile_logits, gold_tile)
            else:
                tile_loss = torch.tensor(0.0, device=DEVICE)

            loss = action_loss + tile_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(tsumo_model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * x.size(0)
            train_total += x.size(0)

            with torch.no_grad():
                train_correct_action += (masked_action_logits.argmax(dim=1) == action_y).sum().item()
                train_correct_joint += compute_joint_top1(
                    action_logits, tile_logits, action_mask, tile_mask, action_y, tile_y
                )

        train_loss = train_loss_sum / max(train_total, 1)
        train_joint_top1 = train_correct_joint / max(train_total, 1)
        train_action_top1 = train_correct_action / max(train_total, 1)

        val_loss, val_joint_top1, val_action_top1, val_action_acc, val_dahai_tile_acc = evaluate_tsumo_branch(tsumo_model, val_loader, action_criterion)

        prev_lr = get_current_lr(optimizer)
        scheduler.step(val_joint_top1)
        new_lr = get_current_lr(optimizer)
        lr_changed = abs(new_lr - prev_lr) > 1e-12

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"tsumo: loss={train_loss:.4f} "
            f"train_joint_top1={train_joint_top1:.4f} "
            f"train_action_top1={train_action_top1:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_joint_top1={val_joint_top1:.4f} "
            f"val_action_top1={val_action_top1:.4f} "
            f"val_dahai_tile_acc={val_dahai_tile_acc:.4f} "
            f"lr={new_lr:.6g}"
            "  Val action acc: " + ", ".join([f"{k}={v:.3f}" for k, v in val_action_acc.items()])
        )

        if val_joint_top1 > best_joint_top1:
            best_joint_top1 = val_joint_top1
            torch.save(tsumo_model.state_dict(), BEST_TSUMO_MODEL_PATH)
            print(f"  -> New best tsumo branch! val_joint_top1={val_joint_top1:.4f}")

        if lr_changed:
            print(f"  -> LR reduced from {prev_lr:.6g} to {new_lr:.6g}")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": tsumo_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_joint_top1": best_joint_top1,
            },
            TSUMO_RESUME_PATH,
        )

    print(f"\nDone. Best tsumo val_joint_top1={best_joint_top1:.4f}")
    print(f"Saved best tsumo branch to: {BEST_TSUMO_MODEL_PATH}")


if __name__ == "__main__":
    train_tsumo_decision()
