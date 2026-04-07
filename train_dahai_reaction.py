from pathlib import Path
from dataloader import (
    MahjongDahaiDataset,
    load_processed_dahai_dataset,
    DAHAI_ACTION_NAMES,
)
from model import MahjongDecisionNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader

# ============================================================
# Config
# ============================================================

BATCH_SIZE = 256
EPOCHS = 30
LR = 3e-4
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

NUM_DAHAI_CLASSES = len(DAHAI_ACTION_NAMES)
PASS_CLASS = 0
DAHAI_CLASS_NAMES = [DAHAI_ACTION_NAMES[i] for i in range(NUM_DAHAI_CLASSES)]

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

BEST_DAHAI_MODEL_PATH = CHECKPOINT_DIR / "best_dahai.pt"
DAHAI_RESUME_PATH = CHECKPOINT_DIR / "latest_dahai.pt"

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

dahai_weights = torch.tensor([
    1.0,  # none
    1.9,  # chi_low
    1.9,  # chi_mid
    1.9,  # chi_high
    1.6,  # pon
    2.0,  # kan
    1.4,  # hora
], dtype=torch.float32, device=DEVICE)

# ============================================================
# Confidence thresholds for prediction-time filtering only
# ============================================================

CHI_CLASSES = [1, 2, 3]
PON_CLASS = 4
KAN_CLASS = 5

DAHAI_CONF_THRESHOLDS = {
    1: 0.75,  # chi_low
    2: 0.75,  # chi_mid
    3: 0.75,  # chi_high
    4: 0.68,  # pon
    5: 0.90,  # kan / daiminkan
}


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


def masked_prediction(logits, mask):
    return logits.masked_fill(~mask, -1e9)


def masked_prediction_with_thresholds(logits, mask, thresholds):
    """
    Prediction-time masking:
    1) mask illegal actions
    2) compute probabilities on legal actions
    3) suppress low-confidence chi / pon / kan classes
    4) re-mask suppressed actions to -1e9 so remaining legal logits are
       effectively renormalized by downstream softmax/argmax
    """
    masked_logits = logits.masked_fill(~mask, -1e9)
    probs = F.softmax(masked_logits, dim=1)

    filtered_mask = mask.clone()

    for cls_idx, thresh in thresholds.items():
        low_conf = probs[:, cls_idx] < thresh
        filtered_mask[:, cls_idx] &= ~low_conf

    # Safety: always keep at least one legal action.
    # If thresholding removed everything, fall back to original legal mask.
    has_any_legal = filtered_mask.any(dim=1)
    filtered_mask = torch.where(
        has_any_legal.unsqueeze(1),
        filtered_mask,
        mask,
    )

    return logits.masked_fill(~filtered_mask, -1e9), filtered_mask


def compute_multiclass_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int):
    y_true = y_true.view(-1).long().cpu()
    y_pred = y_pred.view(-1).long().cpu()

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1

    total = confusion.sum().item()
    correct = confusion.diag().sum().item()
    accuracy = correct / max(total, 1)

    per_class = []
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0

    precisions, recalls, f1s = [], [], []

    for c in range(num_classes):
        tp = confusion[c, c].item()
        fp = confusion[:, c].sum().item() - tp
        fn = confusion[c, :].sum().item() - tp
        support = confusion[c, :].sum().item()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

        per_class.append({
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support

    macro_precision = sum(precisions) / num_classes
    macro_recall = sum(recalls) / num_classes
    macro_f1 = sum(f1s) / num_classes

    weighted_precision /= max(total, 1)
    weighted_recall /= max(total, 1)
    weighted_f1 /= max(total, 1)

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion": confusion,
    }


def format_dahai_metrics(report):
    metrics = report["metrics"]
    lines = []
    lines.append(
        f"val_acc={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} weighted_f1={metrics['weighted_f1']:.4f}"
    )
    cls_parts = []
    for i, name in enumerate(DAHAI_CLASS_NAMES):
        c = metrics["per_class"][i]
        cls_parts.append(f"{name}:P={c['precision']:.3f}/R={c['recall']:.3f}/F1={c['f1']:.3f}")
    lines.append(" | ".join(cls_parts[:4]))
    lines.append(" | ".join(cls_parts[4:]))
    return "\n                   ".join(lines)


def evaluate_dahai_branch(model, loader, criterion):
    model.eval()
    total_loss, total = 0.0, 0
    all_true, all_pred = [], []

    with torch.no_grad():
        for x, mask, hist, hist_mask, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            hist = hist.to(DEVICE, non_blocking=True)
            hist_mask = hist_mask.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            logits, _ = model(x, hist, hist_mask)

            # loss still uses plain legality masking
            train_logits = masked_prediction(logits, mask)
            loss = criterion(train_logits, y)

            # prediction uses thresholded masking
            pred_logits, _ = masked_prediction_with_thresholds(
                logits, mask, DAHAI_CONF_THRESHOLDS
            )
            pred = pred_logits.argmax(dim=1)

            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            all_true.append(y.detach().cpu())
            all_pred.append(pred.detach().cpu())

    if total == 0:
        return {"loss": 0.0, "metrics": compute_multiclass_metrics(torch.tensor([]), torch.tensor([]), NUM_DAHAI_CLASSES)}

    y_true = torch.cat(all_true, dim=0)
    y_pred = torch.cat(all_pred, dim=0)
    metrics = compute_multiclass_metrics(y_true, y_pred, NUM_DAHAI_CLASSES)
    return {
        "loss": total_loss / total,
        "metrics": metrics,
    }


def train_dahai_reaction():
    dahai_samples, meta = load_processed_dahai_dataset("./processed_dataset")
    print(meta)

    if dahai_samples["y"].shape[0] < 100:
        print("Not enough dahai samples.")
        return

    train_packed, val_packed = split_packed_dict(dahai_samples, train_ratio=0.85, seed=SEED)
    train_ds = MahjongDahaiDataset(train_packed)
    val_ds = MahjongDahaiDataset(val_packed)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)

    wrapper = MahjongDecisionNet().to(DEVICE)
    dahai_model = wrapper.dahai_model

    total_params = sum(p.numel() for p in dahai_model.parameters())
    print(f"\nDahai params:      {total_params:,}")
    print(f"Dahai train/val:   {len(train_ds):,} / {len(val_ds):,}")

    optimizer = torch.optim.AdamW(dahai_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        threshold=1e-4,
        threshold_mode="abs",
        min_lr=1e-6,
    )
    criterion = nn.CrossEntropyLoss(weight=dahai_weights)

    start_epoch = 1
    best_weighted_f1 = -0.1

    if DAHAI_RESUME_PATH.exists():
        print(f"Resuming dahai branch from {DAHAI_RESUME_PATH}")
        ckpt = torch.load(DAHAI_RESUME_PATH, map_location=DEVICE)
        dahai_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_weighted_f1 = ckpt.get("best_weighted_f1", 0.0)
        print(f"  Resumed at epoch {start_epoch}, best_weighted_f1={best_weighted_f1:.4f}")
        print(f"  Current LR after resume: {get_current_lr(optimizer):.6g}")

    print(f"\nDevice: {DEVICE}")
    print(f"Dahai LR: {LR}, Weight Decay: {WEIGHT_DECAY}, Epochs: {EPOCHS}\n")
    print(f"Prediction thresholds: {DAHAI_CONF_THRESHOLDS}\n")

    for epoch in range(start_epoch, EPOCHS + 1):
        dahai_model.train()
        train_loss_sum, train_total = 0.0, 0
        train_true, train_pred = [], []

        for x, mask, hist, hist_mask, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            hist = hist.to(DEVICE, non_blocking=True)
            hist_mask = hist_mask.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = dahai_model(x, hist, hist_mask)

            # training loss: only legal-mask, no threshold filtering
            train_logits = masked_prediction(logits, mask)
            loss = criterion(train_logits, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(dahai_model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * x.size(0)
            train_total += x.size(0)

            with torch.no_grad():
                # metrics: thresholded prediction
                pred_logits, _ = masked_prediction_with_thresholds(
                    logits, mask, DAHAI_CONF_THRESHOLDS
                )
                pred = pred_logits.argmax(dim=1)
                train_true.append(y.detach().cpu())
                train_pred.append(pred.detach().cpu())

        train_true = torch.cat(train_true, dim=0)
        train_pred = torch.cat(train_pred, dim=0)
        train_metrics = compute_multiclass_metrics(train_true, train_pred, NUM_DAHAI_CLASSES)

        val_report = evaluate_dahai_branch(dahai_model, val_loader, criterion)
        weighted_f1 = val_report["metrics"]["weighted_f1"]

        prev_lr = get_current_lr(optimizer)
        scheduler.step(weighted_f1)
        new_lr = get_current_lr(optimizer)
        lr_changed = abs(new_lr - prev_lr) > 1e-12

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"dahai: loss={train_loss_sum/max(train_total,1):.4f} "
            f"acc={train_metrics['accuracy']:.4f} "
            f"macro_f1={train_metrics['macro_f1']:.4f}"
        )
        print(f"                   {format_dahai_metrics(val_report)}")
        print(f"                   lr={new_lr:.6g}")

        if weighted_f1 > best_weighted_f1:
            best_weighted_f1 = weighted_f1
            torch.save(dahai_model.state_dict(), BEST_DAHAI_MODEL_PATH)
            print(f"  -> New best dahai branch! best_weighted_f1={best_weighted_f1:.4f}")

        if lr_changed:
            print(f"  -> LR reduced from {prev_lr:.6g} to {new_lr:.6g}")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": dahai_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_weighted_f1": best_weighted_f1,
            },
            DAHAI_RESUME_PATH,
        )

    print(f"\nDone. Best dahai weighted_f1={best_weighted_f1:.4f}")
    print(f"Saved best dahai branch to: {BEST_DAHAI_MODEL_PATH}")


if __name__ == "__main__":
    train_dahai_reaction()