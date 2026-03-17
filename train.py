"""
Train script optimized for M1 Mac (8GB RAM).

Usage:
    python train.py              # full training
    python train.py --dry-run    # quick test: scan 10 files, 1 epoch, verify everything works
"""
import sys
import os
import gc
from pathlib import Path
from dataloader import (
    MahjongDiscardDataset, MahjongCallDataset, MahjongRiichiDataset,
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
# Detect dry-run mode
# ============================================================
DRY_RUN = "--dry-run" in sys.argv

# ============================================================
# Config — conservative for 8GB M1 Mac
# ============================================================

DATASET_HANDLE = "shokanekolouis/tenhou-to-mjai"
DATASET_PATH = Path(kagglehub.dataset_download(DATASET_HANDLE))
print("DATASET_PATH:", DATASET_PATH)

if DRY_RUN:
    print("\n*** DRY RUN MODE — quick test only ***\n")
    MAX_FILES = 10
    MAX_DISCARD_SAMPLES = 1000
    MAX_CALL_SAMPLES = 500
    MAX_RIICHI_SAMPLES = 200
    BATCH_SIZE = 32
    EPOCHS = 2
    NUM_SCAN_WORKERS = 1
else:
    MAX_FILES = 1500
    MAX_DISCARD_SAMPLES = 300_000
    MAX_CALL_SAMPLES = 100_000
    MAX_RIICHI_SAMPLES = 30_000
    BATCH_SIZE = 64
    EPOCHS = 30
    NUM_SCAN_WORKERS = 1         # single process to save memory

YEARS = [str(y) for y in range(2023, 2025)]
LR = 3e-4
WARMUP_EPOCHS = 5
LABEL_SMOOTHING = 0.05
CALL_LOSS_WEIGHT = 0.5
RIICHI_LOSS_WEIGHT = 0.3
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 42

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = CHECKPOINT_DIR / "best.pt"
RESUME_PATH = CHECKPOINT_DIR / "latest.pt"

random.seed(SEED)
torch.manual_seed(SEED)


def masked_prediction(logits, mask):
    masked = logits.clone()
    masked[~mask] = -1e9
    return masked


def get_lr(epoch, warmup_epochs, max_epochs, base_lr):
    if epoch <= warmup_epochs:
        return base_lr * epoch / max(warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
    return base_lr * 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())


def compute_class_weights(samples, num_classes, label_idx=1):
    labels = [s[label_idx] for s in samples]
    counts = Counter(labels)
    total = len(labels)
    weights = torch.ones(num_classes)
    for cls in range(num_classes):
        if counts[cls] > 0:
            weights[cls] = (total / (num_classes * counts[cls])) ** 0.5
    print(f"  Distribution: {dict(counts)}")
    print(f"  Weights (sqrt): {[f'{w:.3f}' for w in weights.tolist()]}")
    return weights


def print_memory():
    """Print approximate memory usage."""
    import resource
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    # macOS reports in bytes, Linux in KB
    if sys.platform == "darwin":
        mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    print(f"  Peak memory: ~{mem_mb:.0f} MB")


# ============================================================
# Evaluate
# ============================================================

def evaluate_discard(model, loader):
    model.eval()
    total_loss, total, correct_top1, correct_top3 = 0.0, 0, 0, 0
    with torch.no_grad():
        for x, mask, y in loader:
            x, mask, y = x.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
            logits = model.forward_discard(x)
            loss = F.cross_entropy(logits, y, label_smoothing=LABEL_SMOOTHING)
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            pred_logits = masked_prediction(logits, mask)
            correct_top1 += (pred_logits.argmax(dim=1) == y).sum().item()
            _, top3 = pred_logits.topk(3, dim=1)
            correct_top3 += (top3 == y.unsqueeze(1)).any(dim=1).sum().item()
    n = max(total, 1)
    return total_loss / n, correct_top1 / n, correct_top3 / n


def evaluate_binary(model, loader, criterion, forward_fn):
    model.eval()
    total_loss, total, correct = 0.0, 0, 0
    num_classes = None
    class_correct, class_total = {}, {}
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = forward_fn(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            if num_classes is None:
                num_classes = logits.shape[1]
                for c in range(num_classes):
                    class_correct[c] = 0
                    class_total[c] = 0
            for c in range(num_classes):
                m = (y == c)
                class_total[c] += m.sum().item()
                class_correct[c] += (pred[m] == c).sum().item()
    n = max(total, 1)
    class_acc = {c: class_correct[c] / max(class_total[c], 1) for c in range(num_classes or 0)}
    return total_loss / n, correct / n, class_acc


# ============================================================
# Train
# ============================================================

def train():
    # --- data ---
    print("=" * 50)
    print("Phase 1: Data Loading")
    print("=" * 50)

    discard_samples, call_samples, riichi_samples = build_dataset(
        DATASET_PATH, YEARS, MAX_FILES,
        MAX_DISCARD_SAMPLES, MAX_CALL_SAMPLES, MAX_RIICHI_SAMPLES,
        num_workers=NUM_SCAN_WORKERS,
    )

    if len(discard_samples) < 50:
        print("Too few discard samples.")
        return

    print_memory()

    # --- build datasets and free raw samples ---
    print("\n" + "=" * 50)
    print("Phase 2: Building Datasets")
    print("=" * 50)

    # discard
    random.shuffle(discard_samples)
    d_split = int(0.85 * len(discard_samples))
    d_train = MahjongDiscardDataset(discard_samples[:d_split])
    d_val = MahjongDiscardDataset(discard_samples[d_split:])
    del discard_samples  # free memory
    gc.collect()

    d_train_loader = DataLoader(d_train, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=0, pin_memory=False)
    d_val_loader = DataLoader(d_val, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)

    # call
    has_call = len(call_samples) >= 50
    if has_call:
        random.shuffle(call_samples)
        c_split = int(0.85 * len(call_samples))
        print("Call weights:")
        call_weights = compute_class_weights(call_samples[:c_split], 3).to(DEVICE)
        c_train = MahjongCallDataset(call_samples[:c_split])
        c_val = MahjongCallDataset(call_samples[c_split:])
        del call_samples
        gc.collect()
        c_train_loader = DataLoader(c_train, batch_size=BATCH_SIZE, shuffle=True,
                                    num_workers=0, pin_memory=False)
        c_val_loader = DataLoader(c_val, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=0, pin_memory=False)
        call_criterion = torch.nn.CrossEntropyLoss(weight=call_weights, label_smoothing=LABEL_SMOOTHING)
    else:
        del call_samples
        gc.collect()
        print("Not enough call samples; skipping.")

    # riichi
    has_riichi = len(riichi_samples) >= 50
    if has_riichi:
        random.shuffle(riichi_samples)
        r_split = int(0.85 * len(riichi_samples))
        print("Riichi weights:")
        riichi_weights = compute_class_weights(riichi_samples[:r_split], 2).to(DEVICE)
        r_train = MahjongRiichiDataset(riichi_samples[:r_split])
        r_val = MahjongRiichiDataset(riichi_samples[r_split:])
        del riichi_samples
        gc.collect()
        r_train_loader = DataLoader(r_train, batch_size=BATCH_SIZE, shuffle=True,
                                    num_workers=0, pin_memory=False)
        r_val_loader = DataLoader(r_val, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=0, pin_memory=False)
        riichi_criterion = torch.nn.CrossEntropyLoss(weight=riichi_weights, label_smoothing=LABEL_SMOOTHING)
    else:
        del riichi_samples
        gc.collect()
        print("Not enough riichi samples; skipping.")

    print_memory()

    # --- model ---
    print("\n" + "=" * 50)
    print("Phase 3: Training")
    print("=" * 50)

    model = SmallMahjongResNet(in_channels=16).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params:       {total_params:,}")
    print(f"Discard train/val:  {len(d_train):,} / {len(d_val):,}")
    if has_call:
        print(f"Call train/val:     {len(c_train):,} / {len(c_val):,}")
    if has_riichi:
        print(f"Riichi train/val:   {len(r_train):,} / {len(r_val):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # resume
    start_epoch = 1
    best_val_acc = 0.0
    if not DRY_RUN and RESUME_PATH and RESUME_PATH.exists():
        print(f"Resuming from {RESUME_PATH}")
        ckpt = torch.load(RESUME_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"  Resumed at epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")

    print(f"Device: {DEVICE}")
    print(f"LR: {LR}, Warmup: {WARMUP_EPOCHS}, Epochs: {EPOCHS}")
    print()

    # --- training loop ---
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        current_lr = get_lr(epoch, WARMUP_EPOCHS, EPOCHS, LR)
        for pg in optimizer.param_groups:
            pg['lr'] = current_lr

        # discard
        d_loss_sum, d_total, d_correct = 0.0, 0, 0
        for x, mask, y in d_train_loader:
            x, mask, y = x.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model.forward_discard(x)
            loss = F.cross_entropy(logits, y, label_smoothing=LABEL_SMOOTHING)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            d_loss_sum += loss.item() * x.size(0)
            d_total += x.size(0)
            with torch.no_grad():
                d_correct += (masked_prediction(logits, mask).argmax(1) == y).sum().item()

        # call
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
                c_correct += (logits.argmax(1) == y).sum().item()
                c_total += x.size(0)

        # riichi
        r_loss_sum, r_total, r_correct = 0.0, 0, 0
        if has_riichi:
            for x, y in r_train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                logits = model.forward_riichi(x)
                loss = riichi_criterion(logits, y) * RIICHI_LOSS_WEIGHT
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                r_loss_sum += loss.item() * x.size(0)
                r_correct += (logits.argmax(1) == y).sum().item()
                r_total += x.size(0)

        # evaluate
        d_val_loss, d_val_top1, d_val_top3 = evaluate_discard(model, d_val_loader)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | lr={current_lr:.6f} | "
            f"discard: loss={d_loss_sum/d_total:.4f} acc={d_correct/d_total:.4f} "
            f"val_loss={d_val_loss:.4f} val_top1={d_val_top1:.4f} val_top3={d_val_top3:.4f}"
        )

        if has_call and c_total > 0:
            _, c_val_acc, c_cls = evaluate_binary(model, c_val_loader, call_criterion, model.forward_call)
            print(
                f"         call:    loss={c_loss_sum/c_total:.4f} acc={c_correct/c_total:.4f} "
                f"val_acc={c_val_acc:.4f} "
                f"[pass={c_cls[0]:.3f} pon={c_cls[1]:.3f} chi={c_cls[2]:.3f}]"
            )

        if has_riichi and r_total > 0:
            _, r_val_acc, r_cls = evaluate_binary(model, r_val_loader, riichi_criterion, model.forward_riichi)
            print(
                f"         riichi:  loss={r_loss_sum/r_total:.4f} acc={r_correct/r_total:.4f} "
                f"val_acc={r_val_acc:.4f} "
                f"[no_riichi={r_cls[0]:.3f} riichi={r_cls[1]:.3f}]"
            )

        if d_val_top1 > best_val_acc:
            best_val_acc = d_val_top1
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> New best model! val_top1={d_val_top1:.4f}")

        if not DRY_RUN:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
            }, CHECKPOINT_DIR / "latest.pt")

    print(f"\nDone. Best discard val_top1={best_val_acc:.4f}")
    print(f"Model saved at: {BEST_MODEL_PATH}")

    if DRY_RUN:
        print("\n*** DRY RUN PASSED — safe to run full training ***")


if __name__ == "__main__":
    train()