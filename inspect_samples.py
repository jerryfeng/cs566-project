import torch
from torch.utils.data import DataLoader
from collections import Counter

from dataloader import (
    load_processed_call_dataset,
    MahjongCallDataset,
    CALL_CHI,
    CALL_ACTION_NAMES,
)

# ============================================================
# Config
# ============================================================

DATA_DIR = "./processed_dataset"
NUM_SAMPLES_TO_SHOW = 8
BATCH_SIZE = 64
SHUFFLE = True

# ============================================================
# Helpers
# ============================================================

def print_tensor_stats(name, t):
    print(f"{name}: shape={tuple(t.shape)}, dtype={t.dtype}")
    if t.numel() == 0:
        return
    if t.dtype == torch.bool:
        print(f"  true count={t.sum().item()}, false count={(~t).sum().item()}")
    elif torch.is_floating_point(t) or t.dtype in (
        torch.int8, torch.int16, torch.int32, torch.int64,
        torch.uint8
    ):
        flat = t.view(-1)
        print(
            f"  min={flat.min().item()}, max={flat.max().item()}, "
            f"sum={flat.sum().item()}"
        )

def inspect_chi_sample(x, mask, hist, hist_mask, y, idx_in_batch=None):
    print("\n" + "=" * 80)
    if idx_in_batch is not None:
        print(f"Sample index in batch: {idx_in_batch}")
    print(f"Label: {y.item()} ({CALL_ACTION_NAMES.get(int(y.item()), 'unknown')})")

    print_tensor_stats("x", x)
    print_tensor_stats("mask", mask)
    print_tensor_stats("hist", hist)
    print_tensor_stats("hist_mask", hist_mask)

    print("\n[mask]")
    print(mask)

    if mask.ndim == 1:
        legal_ids = torch.nonzero(mask > 0, as_tuple=False).view(-1).tolist()
        legal_names = [CALL_ACTION_NAMES.get(i, str(i)) for i in legal_ids]
        print(f"Legal action ids: {legal_ids}")
        print(f"Legal action names: {legal_names}")
        print(f"Is chi legal? {bool(mask[CALL_CHI].item()) if CALL_CHI < mask.shape[0] else 'CALL_CHI out of range'}")

    print("\n[x]")
    print(x)

    if x.ndim == 2:
        print("\n[x channel sums]")
        print(x.sum(dim=1))

    valid_hist = hist[~hist_mask] if hist_mask.ndim == 1 else hist
    print(f"\nValid history length: {valid_hist.shape[0]}")

    if valid_hist.shape[0] > 0:
        n = min(10, valid_hist.shape[0])
        print(f"\n[last {n} valid history events]")
        print(valid_hist[-n:])

    print("=" * 80)

# ============================================================
# Load dataset
# ============================================================

packed, meta = load_processed_call_dataset(DATA_DIR)
dataset = MahjongCallDataset(packed)

print(f"Dataset size: {len(dataset)}")

label_counts = Counter(dataset.y.tolist())
print("\nLabel counts:")
for k, v in sorted(label_counts.items()):
    print(f"  {k:2d} ({CALL_ACTION_NAMES.get(k, 'unknown')}): {v}")

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

# ============================================================
# Scan for chi samples
# ============================================================

shown = 0
for batch_idx, batch in enumerate(loader):
    x, mask, hist, hist_mask, y = batch

    chi_indices = torch.nonzero(y == CALL_CHI, as_tuple=False).view(-1)
    if chi_indices.numel() == 0:
        continue

    print(f"\nBatch {batch_idx}: found {chi_indices.numel()} chi samples")

    for i in chi_indices.tolist():
        inspect_chi_sample(
            x=x[i],
            mask=mask[i],
            hist=hist[i],
            hist_mask=hist_mask[i],
            y=y[i],
            idx_in_batch=i,
        )
        shown += 1
        if shown >= NUM_SAMPLES_TO_SHOW:
            print(f"\nDone. Displayed {shown} chi samples.")
            raise SystemExit

print(f"\nFinished scanning. Displayed {shown} chi samples.")