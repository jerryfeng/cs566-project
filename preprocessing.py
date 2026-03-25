from pathlib import Path
from collections import Counter
import kagglehub
import torch

from dataloader import (
    build_and_save_dataset,
    CALL_ACTION_NAMES,
)

YEARS = [2025, 2026]
MAX_FILES = 10_000
MAX_DISCARD_SAMPLES = 1_000_000
MAX_CALL_SAMPLES = 1_000_000
NUM_DATA_LOADER_WORKERS = 16

DATASET_HANDLE = "shokanekolouis/tenhou-to-mjai"
DATASET_PATH = Path(kagglehub.dataset_download(DATASET_HANDLE))

OUT_DIR = Path("./processed_dataset")
CLEANUP_SHARDS_AFTER_MERGE = True


if __name__ == "__main__":
    meta = {
        "root_dir": str(DATASET_PATH),
        "years": YEARS,
        "max_files": MAX_FILES,
        "max_discard_samples": MAX_DISCARD_SAMPLES,
        "max_call_samples": MAX_CALL_SAMPLES,
        "num_workers": NUM_DATA_LOADER_WORKERS,
        "dataset_handle": DATASET_HANDLE,
    }

    discard_data, call_data, shard_paths = build_and_save_dataset(
        root_dir=DATASET_PATH,
        years=YEARS,
        max_files=MAX_FILES,
        max_discard_samples=MAX_DISCARD_SAMPLES,
        max_call_samples=MAX_CALL_SAMPLES,
        num_workers=NUM_DATA_LOADER_WORKERS,
        out_dir=OUT_DIR,
        shard_subdir="shards",
        meta=meta,
        cleanup_shards=CLEANUP_SHARDS_AFTER_MERGE,
    )

    call_counts = Counter(call_data["y"].tolist())
    call_breakdown = {
        int(action_id): {
            "name": CALL_ACTION_NAMES[action_id],
            "count": int(call_counts.get(action_id, 0)),
        }
        for action_id in sorted(CALL_ACTION_NAMES)
    }

    # update meta in memory
    meta["call_action_breakdown"] = call_breakdown

    # overwrite meta.pt with enriched metadata
    torch.save(meta, OUT_DIR / "meta.pt")

    print(f"Final dataset saved to {OUT_DIR}")
    print(f"Discard samples: {discard_data['y'].shape[0]}")
    print(f"Call samples:    {call_data['y'].shape[0]}")
    print(f"Shard count:     {len(shard_paths)}")

    print("\nCall action breakdown:")
    for action_id in sorted(CALL_ACTION_NAMES):
        name = CALL_ACTION_NAMES[action_id]
        count = call_breakdown[action_id]["count"]
        print(f"  {action_id:>2} ({name:<10}): {count}")

    print(f"\nUpdated metadata saved to {OUT_DIR / 'meta.pt'}")
