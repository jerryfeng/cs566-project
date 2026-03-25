from pathlib import Path
from dataloader import (
    MahjongCallDataset,
    load_processed_call_dataset,
)
from model import MahjongResNet
import torch
import torch.nn as nn
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

NUM_CALL_CLASSES = 8
PASS_CLASS = 0
CALL_CLASS_NAMES = ["pass", "chi", "pon", "hora", "dmk", "ank", "kak", "rii"]

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

BEST_CALL_MODEL_PATH = CHECKPOINT_DIR / "best_call.pt"
CALL_RESUME_PATH = CHECKPOINT_DIR / "latest_call.pt"

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

call_thresholds = {
    1: 0.83,  # chi
    2: 0.74,  # pon
    4: 0.99,  # daiminkan
    5: 0.96,  # ankan
    6: 0.99,  # kakan
}

call_weights = torch.tensor([
    1.0,  # pass
    1.7,  # chi
    1.5,  # pon
    1.3,  # hora
    3.0,  # daiminkan
    2.0,  # ankan
    2.2,  # kakan
    1.4,  # riichi
], dtype=torch.float32, device=DEVICE)

# CALL_QUALITY_CONFIG = {
#     0: {"name": "none",   "p_target": 0.90, "r_target": 0.90},
#     1: {"name": "chi",    "p_target": 0.85, "r_target": 0.60},
#     2: {"name": "pon",    "p_target": 0.85, "r_target": 0.60},
#     3: {"name": "hora",   "p_target": 0.90, "r_target": 0.95},
#     4: {"name": "dmk",    "p_target": 0.90, "r_target": None},
#     5: {"name": "ank",    "p_target": 0.85, "r_target": 0.30},
#     6: {"name": "kak",    "p_target": 0.85, "r_target": 0.30},
#     7: {"name": "riichi", "p_target": 0.70, "r_target": 0.80},
# }

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


def masked_call_prediction(logits, legal_mask):
    """
    Apply legality mask + thresholds, return masked probabilities.
    """
    masked_logits = logits.clone()
    masked_logits[~legal_mask] = -1e9

    probs = F.softmax(masked_logits, dim=-1)

    for action_idx, threshold in call_thresholds.items():
        p = probs[:, action_idx]
        reject = p < threshold
        if reject.any():
            probs[reject, action_idx] = 0.0

    return probs


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

    precisions = []
    recalls = []
    f1s = []

    for c in range(num_classes):
        tp = confusion[c, c].item()
        fp = confusion[:, c].sum().item() - tp
        fn = confusion[c, :].sum().item() - tp
        support = confusion[c, :].sum().item()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

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


def compute_binary_call_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, pass_class: int = 0):
    y_true = y_true.view(-1).long().cpu()
    y_pred = y_pred.view(-1).long().cpu()

    true_call = (y_true != pass_class)
    pred_call = (y_pred != pass_class)

    tp = (true_call & pred_call).sum().item()
    fp = (~true_call & pred_call).sum().item()
    fn = (true_call & ~pred_call).sum().item()
    tn = (~true_call & ~pred_call).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    pass_total = (~true_call).sum().item()
    pass_to_call_fpr = fp / max(pass_total, 1)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pass_to_call_fpr": pass_to_call_fpr,
    }


def format_call_metrics(report):
    metrics = report["metrics"]
    binary = report["binary"]

    lines = []
    lines.append(
        f"val_acc={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} weighted_f1={metrics['weighted_f1']:.4f}"
    )
    lines.append(
        f"call(any): P={binary['precision']:.3f} R={binary['recall']:.3f} F1={binary['f1']:.3f} "
        f"| pass->call_FPR={binary['pass_to_call_fpr']:.4f}"
    )

    cls_parts = []
    for i, name in enumerate(CALL_CLASS_NAMES):
        c = metrics["per_class"][i]
        cls_parts.append(
            f"{name}:P={c['precision']:.3f}/R={c['recall']:.3f}/F1={c['f1']:.3f}"
        )

    lines.append(" | ".join(cls_parts[:4]))
    lines.append(" | ".join(cls_parts[4:]))

    return "\n                   ".join(lines)


def evaluate_call_branch(call_model, loader, criterion):
    call_model.eval()
    total_loss, total = 0.0, 0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for x, mask, hist, hist_mask, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            hist = hist.to(DEVICE, non_blocking=True)
            hist_mask = hist_mask.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            call_logits, _ = call_model(x, hist, hist_mask)

            train_logits = masked_prediction(call_logits, mask)
            loss = criterion(train_logits, y)

            pred_probs = masked_call_prediction(call_logits, mask)
            pred = pred_probs.argmax(dim=1)

            total_loss += loss.item() * x.size(0)
            total += x.size(0)

            all_true.append(y.detach().cpu())
            all_pred.append(pred.detach().cpu())

    n = max(total, 1)

    if len(all_true) == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "metrics": None,
            "binary": None,
        }

    y_true = torch.cat(all_true, dim=0)
    y_pred = torch.cat(all_pred, dim=0)

    metrics = compute_multiclass_metrics(y_true, y_pred, NUM_CALL_CLASSES)
    binary = compute_binary_call_metrics(y_true, y_pred, pass_class=PASS_CLASS)

    return {
        "loss": total_loss / n,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "metrics": metrics,
        "binary": binary,
    }

def train_call():
    call_samples, meta = load_processed_call_dataset("./processed_dataset")
    print(meta)

    if call_samples["y"].shape[0] < 100:
        print("Not enough call samples.")
        return

    c_train_packed, c_val_packed = split_packed_dict(call_samples, train_ratio=0.85, seed=SEED)
    c_train = MahjongCallDataset(c_train_packed)
    c_val = MahjongCallDataset(c_val_packed)

    c_train_loader = DataLoader(
        c_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )
    c_val_loader = DataLoader(
        c_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    wrapper = MahjongResNet().to(DEVICE)
    call_model = wrapper.call_model

    total_params = sum(p.numel() for p in call_model.parameters())
    print(f"\nCall params:        {total_params:,}")
    print(f"Call train/val:     {len(c_train):,} / {len(c_val):,}")

    optimizer = torch.optim.AdamW(
        call_model.parameters(),
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

    criterion = nn.CrossEntropyLoss(weight=call_weights)

    start_epoch = 1
    best_weighted_f1 = -0.1

    if CALL_RESUME_PATH.exists():
        print(f"Resuming call from {CALL_RESUME_PATH}")
        ckpt = torch.load(CALL_RESUME_PATH, map_location=DEVICE)
        call_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Resume scheduler too if present
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_weighted_f1 = ckpt.get("best_weighted_f1", 0.0)
        print(f"  Resumed at epoch {start_epoch}, best_weighted_f1={best_weighted_f1:.4f}")
        print(f"  Current LR after resume: {get_current_lr(optimizer):.6g}")

    print(f"\nDevice: {DEVICE}")
    print(f"Call LR: {LR}, Weight Decay: {WEIGHT_DECAY}, Epochs: {EPOCHS}\n")

    for epoch in range(start_epoch, EPOCHS + 1):
        call_model.train()

        train_loss_sum, train_total = 0.0, 0
        train_true = []
        train_pred = []

        for x, mask, hist, hist_mask, y in c_train_loader:
            x = x.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            hist = hist.to(DEVICE, non_blocking=True)
            hist_mask = hist_mask.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            call_logits, _ = call_model(x, hist, hist_mask)

            # train with legality mask only
            train_logits = masked_prediction(call_logits, mask)
            loss = criterion(train_logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(call_model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * x.size(0)
            train_total += x.size(0)

            with torch.no_grad():
                pred_probs = masked_call_prediction(call_logits, mask)
                pred = pred_probs.argmax(dim=1)
                train_true.append(y.detach().cpu())
                train_pred.append(pred.detach().cpu())

        train_true = torch.cat(train_true, dim=0)
        train_pred = torch.cat(train_pred, dim=0)
        train_metrics = compute_multiclass_metrics(train_true, train_pred, NUM_CALL_CLASSES)
        train_binary = compute_binary_call_metrics(train_true, train_pred, pass_class=PASS_CLASS)

        val_report = evaluate_call_branch(call_model, c_val_loader, criterion)
        weighted_f1 = val_report["metrics"]["weighted_f1"]

        prev_lr = get_current_lr(optimizer)
        scheduler.step(weighted_f1)
        new_lr = get_current_lr(optimizer)
        lr_changed = abs(new_lr - prev_lr) > 1e-12

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"call: loss={train_loss_sum/max(train_total,1):.4f} "
            f"acc={train_metrics['accuracy']:.4f} macro_f1={train_metrics['macro_f1']:.4f} "
            f"callF1={train_binary['f1']:.4f} pass->call_FPR={train_binary['pass_to_call_fpr']:.4f}"
        )
        print(f"                   {format_call_metrics(val_report)}")
        f"lr={new_lr:.6g}"


        if weighted_f1 > best_weighted_f1:
            best_weighted_f1 = weighted_f1
            torch.save(call_model.state_dict(), BEST_CALL_MODEL_PATH)
            print(f"  -> New best call! best_weighted_f1={best_weighted_f1:.4f}")
        
        if lr_changed:
            print(f"  -> LR reduced from {prev_lr:.6g} to {new_lr:.6g}")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": call_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_weighted_f1": best_weighted_f1,
            },
            CALL_RESUME_PATH,
        )

    print(f"\nDone. Best call weighted_f1={best_weighted_f1:.4f}")
    print(f"Saved best call branch to: {BEST_CALL_MODEL_PATH}")


if __name__ == "__main__":
    train_call()
