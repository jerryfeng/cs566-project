from pathlib import Path
from dataloader import (
    MahjongDiscardDataset, MahjongCallDataset, MahjongRiichiDataset, PackedMahjongCallDataset, PackedMahjongDiscardDataset, PackedMahjongRiichiDataset,
    build_dataset, load_processed_dataset,
)
import kagglehub
from model import SmallMahjongResNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader
from collections import Counter

# ============================================================
# Config
# ============================================================

DATASET_HANDLE = "shokanekolouis/tenhou-to-mjai"
DATASET_PATH = Path(kagglehub.dataset_download(DATASET_HANDLE))

# YEARS = [str(y) for y in range(2023, 2025)]
# MAX_FILES = 5000
# MAX_DISCARD_SAMPLES = 1_000_000
# MAX_CALL_SAMPLES = 300_000
# MAX_RIICHI_SAMPLES = 50_000
# NUM_DATA_LOADER_WORKERS = 16
BATCH_SIZE = 256
EPOCHS = 20
LR = 3e-4
WARMUP_EPOCHS = 2
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
    return logits.masked_fill(~mask, -1e9)


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
            logits = masked_prediction(model.forward_discard(x), mask)
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item() * x.size(0)
            total += x.size(0)

            correct_top1 += (logits.argmax(dim=1) == y).sum().item()
            _, top3 = logits.topk(3, dim=1)
            correct_top3 += (top3 == y.unsqueeze(1)).any(dim=1).sum().item()

    n = max(total, 1)
    return total_loss / n, correct_top1 / n, correct_top3 / n

class AsymmetricCallLoss(nn.Module):
    def __init__(self, class_weights=None, label_smoothing=0.0, cost_matrix=None, cost_scale=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
            reduction="none",
        )
        self.cost_scale = cost_scale

        if cost_matrix is None:
            cost_matrix = torch.zeros(3, 3)

        self.register_buffer("cost_matrix", cost_matrix.float())

    def forward(self, logits, targets):
        # Standard CE per sample
        ce_loss = self.ce(logits, targets)  # [B]

        # Predicted probabilities
        probs = torch.softmax(logits, dim=1)  # [B, C]

        # For each sample, pick row = true class
        # row shape: [B, C]
        row_costs = self.cost_matrix[targets]

        # Expected penalty under predicted distribution
        # If true=pass and model puts high prob on chi, penalty is large
        expected_cost = (row_costs * probs).sum(dim=1)  # [B]

        return (ce_loss + self.cost_scale * expected_cost).mean()

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

def compute_class_weights_from_labels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels.long(), minlength=num_classes).float()
    weights = counts.sum() / counts.clamp(min=1.0)
    weights = weights / weights.mean()
    print("class counts:", counts.tolist())
    print("class weights:", weights.tolist())
    return weights

# ============================================================
# Train
# ============================================================

def train():
    discard_samples, call_samples, riichi_samples, meta = load_processed_dataset("./processed_dataset")
    print(meta)

    # --- discard ---
    d_train_packed, d_val_packed = split_packed_dict(discard_samples, train_ratio=0.85, seed=42)
    d_train = PackedMahjongDiscardDataset(d_train_packed)
    d_val = PackedMahjongDiscardDataset(d_val_packed)

    d_train_loader = DataLoader(
        d_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
    )
    d_val_loader = DataLoader(
        d_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
    )

    # --- call ---
    has_call = call_samples["y"].shape[0] >= 100
    if has_call:
        c_train_packed, c_val_packed = split_packed_dict(call_samples, train_ratio=0.85, seed=42)
        c_train = PackedMahjongCallDataset(c_train_packed)
        c_val = PackedMahjongCallDataset(c_val_packed)

        c_train_loader = DataLoader(
            c_train,
            batch_size=BATCH_SIZE,
            shuffle=True,
            pin_memory=True,
        )
        c_val_loader = DataLoader(
            c_val,
            batch_size=BATCH_SIZE,
            shuffle=False,
            pin_memory=True,
        )

        print("Call weights:")
        call_weights = compute_class_weights_from_labels(c_train_packed["y"], 3).to(DEVICE)
        call_cost_matrix = torch.tensor([
            [0.0, 3.0, 5.0],   # true pass
            [0.5, 0.0, 1.0],   # true pon
            [0.5, 1.0, 0.0],   # true chi
        ], dtype=torch.float32)

        call_criterion = AsymmetricCallLoss(
            class_weights=call_weights,
            label_smoothing=0.0,
            cost_matrix=call_cost_matrix,
            cost_scale=0.5,
        ).to(DEVICE)

    # --- riichi ---
    has_riichi = riichi_samples["y"].shape[0] >= 100
    if has_riichi:
        r_train_packed, r_val_packed = split_packed_dict(riichi_samples, train_ratio=0.85, seed=42)
        r_train = PackedMahjongRiichiDataset(r_train_packed)
        r_val = PackedMahjongRiichiDataset(r_val_packed)

        r_train_loader = DataLoader(
            r_train,
            batch_size=BATCH_SIZE,
            shuffle=True,
            pin_memory=True,
        )
        r_val_loader = DataLoader(
            r_val,
            batch_size=BATCH_SIZE,
            shuffle=False,
            pin_memory=True,
        )

        print("Riichi weights:")
        riichi_weights = compute_class_weights_from_labels(r_train_packed["y"], 2).to(DEVICE)
        riichi_criterion = torch.nn.CrossEntropyLoss(
            weight=riichi_weights,
            label_smoothing=LABEL_SMOOTHING,
        )
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
            logits = masked_prediction(model.forward_discard(x), mask)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            d_loss_sum += loss.item() * x.size(0)
            d_total += x.size(0)
            with torch.no_grad():
                d_correct += (logits.argmax(1) == y).sum().item()

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