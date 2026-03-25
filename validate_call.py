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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

NUM_CALL_CLASSES = 8
PASS_CLASS = 0
CALL_CLASS_NAMES = ["pass", "chi", "pon", "hora", "dmk", "ank", "kak", "rii"]

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

BEST_CALL_MODEL_PATH = CHECKPOINT_DIR / "best_call.pt"

# Size of fresh evaluation subset
EVAL_VAL_SAMPLES = 150_000
TRAIN_RATIO = 0.85  # kept only so the split structure matches train_call

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

call_thresholds = {
    1: 0.83,  # chi
    2: 0.74,  # pon
    4: 0.99,  # daiminkan
    5: 0.96,  # ankan
    6: 0.94,  # kakan
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


def take_subset_packed(packed, max_samples, seed=42):
    n = packed["y"].shape[0]
    if n <= max_samples:
        return packed

    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:max_samples]
    return {k: v[idx] for k, v in packed.items()}


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
        f"val_loss={report['loss']:.4f} "
        f"val_acc={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"weighted_f1={metrics['weighted_f1']:.4f}"
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

            eval_logits = masked_prediction(call_logits, mask)
            loss = criterion(eval_logits, y)

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


def validate_best_call():
    call_samples, meta = load_processed_call_dataset("./processed_dataset")
    print(meta)

    if call_samples["y"].shape[0] < 100:
        print("Not enough call samples.")
        return

    # Keep same split structure as train_call, but only use val split.
    _, c_val_packed = split_packed_dict(call_samples, train_ratio=TRAIN_RATIO, seed=SEED)

    # Fresh random subset of 150k from validation split.
    c_val_packed = take_subset_packed(c_val_packed, EVAL_VAL_SAMPLES, seed=SEED + 999)

    c_val = MahjongCallDataset(c_val_packed)
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
    print(f"Eval val samples:   {len(c_val):,}")

    if not BEST_CALL_MODEL_PATH.exists():
        print(f"Missing best model: {BEST_CALL_MODEL_PATH}")
        return

    print(f"Loading best call model from: {BEST_CALL_MODEL_PATH}")
    state_dict = torch.load(BEST_CALL_MODEL_PATH, map_location=DEVICE)
    call_model.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss(weight=call_weights)

    print(f"\nDevice: {DEVICE}")
    print("Running validation only...\n")

    val_report = evaluate_call_branch(call_model, c_val_loader, criterion)
    print(f"                   {format_call_metrics(val_report)}")
    print("\nConfusion matrix:")
    print(val_report["metrics"]["confusion"])


if __name__ == "__main__":
    validate_best_call()
