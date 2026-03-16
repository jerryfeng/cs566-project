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
# Config
# ============================================================

DATASET_HANDLE = "shokanekolouis/tenhou-to-mjai"
DATASET_PATH = Path(kagglehub.dataset_download(DATASET_HANDLE))
print("DATASET_PATH:", DATASET_PATH)

YEARS = [str(y) for y in range(2023, 2025)]
MAX_FILES = 5000
MAX_DISCARD_SAMPLES = 1_000_000
MAX_CALL_SAMPLES = 500_000
MAX_RIICHI_SAMPLES = 200_000
BATCH_SIZE = 256
EPOCHS = 50
LR = 3e-4
WARMUP_EPOCHS = 5
LABEL_SMOOTHING = 0.05
CALL_LOSS_WEIGHT = 0.5
RIICHI_LOSS_WEIGHT = 0.3
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
    """Generic eval for call (3-class) or riichi (2-class)."""
    model.eval()
    total_loss, total, correct = 0.0, 0, 0
    num_classes = None
    class_correct = {}
    class_total = {}

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
    class_acc = {}
    for c in range(num_classes or 0):
        class_acc[c] = class_correct[c] / max(class_total[c], 1)

    return total_loss / n, correct / n, class_acc


# ============================================================
# Train
# ============================================================

def train():
    discard_samples, call_samples, riichi_samples = build_dataset(
        DATASET_PATH, YEARS, MAX_FILES,
        MAX_DISCARD_SAMPLES, MAX_CALL_SAMPLES, MAX_RIICHI_SAMPLES,
    )

    if len(discard_samples) < 100:
        print("Too few discard samples.")
        return

    # --- discard ---
    random.shuffle(discard_samples)
    d_split = int(0.85 * len(discard_samples))
    d_train = MahjongDiscardDataset(discard_samples[:d_split])
    d_val = MahjongDiscardDataset(discard_samples[d_split:])
    d_train_loader = DataLoader(d_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    d_val_loader = DataLoader(d_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # --- call ---
    has_call = len(call_samples) >= 100
    if has_call:
        random.shuffle(call_samples)
        c_split = int(0.85 * len(call_samples))
        c_train = MahjongCallDataset(call_samples[:c_split])
        c_val = MahjongCallDataset(call_samples[c_split:])
        c_train_loader = DataLoader(c_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        c_val_loader = DataLoader(c_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        print("Call weights:")
        call_weights = compute_class_weights(call_samples[:c_split], 3).to(DEVICE)
        call_criterion = torch.nn.CrossEntropyLoss(weight=call_weights, label_smoothing=LABEL_SMOOTHING)

    # --- riichi ---
    has_riichi = len(riichi_samples) >= 100
    if has_riichi:
        random.shuffle(riichi_samples)
        r_split = int(0.85 * len(riichi_samples))
        r_train = MahjongRiichiDataset(riichi_samples[:r_split])
        r_val = MahjongRiichiDataset(riichi_samples[r_split:])
        r_train_loader = DataLoader(r_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        r_val_loader = DataLoader(r_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        print("Riichi weights:")
        riichi_weights = compute_class_weights(riichi_samples[:r_split], 2).to(DEVICE)
        riichi_criterion = torch.nn.CrossEntropyLoss(weight=riichi_weights, label_smoothing=LABEL_SMOOTHING)
    else:
        print("Not enough riichi samples; skipping riichi training.")

    # --- model ---
    model = SmallMahjongResNet(in_channels=16).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal params:       {total_params:,}")
    print(f"Discard train/val:  {len(d_train):,} / {len(d_val):,}")
    if has_call:
        print(f"Call train/val:     {len(c_train):,} / {len(c_val):,}")
    if has_riichi:
        print(f"Riichi train/val:   {len(r_train):,} / {len(r_val):,}")

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

    print(f"\nDevice: {DEVICE}")
    print(f"LR: {LR}, Warmup: {WARMUP_EPOCHS}, Epochs: {EPOCHS}\n")

    # --- training loop ---
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        current_lr = get_lr(epoch, WARMUP_EPOCHS, EPOCHS, LR)
        for pg in optimizer.param_groups:
            pg['lr'] = current_lr

        # ---- discard ----
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

        # ---- call ----
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

        # ---- riichi ----
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

        # ---- evaluate ----
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