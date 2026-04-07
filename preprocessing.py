
from pathlib import Path
from collections import Counter
import kagglehub
import torch

from dataloader import (
    build_and_save_dataset,
    DAHAI_ACTION_NAMES,
    TSUMO_ACTION_NAMES,
)

YEARS = [2025, 2026]
MAX_FILES = 10_000
MAX_DAHAI_SAMPLES = 1_000_000
MAX_TSUMO_SAMPLES = 2_000_000
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
        "max_dahai_samples": MAX_DAHAI_SAMPLES,
        "max_tsumo_samples": MAX_TSUMO_SAMPLES,
        "num_workers": NUM_DATA_LOADER_WORKERS,
        "dataset_handle": DATASET_HANDLE,
    }

    dahai_data, tsumo_data, shard_paths = build_and_save_dataset(
        root_dir=DATASET_PATH,
        years=YEARS,
        max_files=MAX_FILES,
        max_dahai_samples=MAX_DAHAI_SAMPLES,
        max_tsumo_samples=MAX_TSUMO_SAMPLES,
        num_workers=NUM_DATA_LOADER_WORKERS,
        out_dir=OUT_DIR,
        shard_subdir="shards",
        meta=meta,
        cleanup_shards=CLEANUP_SHARDS_AFTER_MERGE,
    )

    dahai_counts = Counter(dahai_data["y"].tolist())
    tsumo_action_counts = Counter(tsumo_data["action_y"].tolist())

    meta["dahai_action_breakdown"] = {
        int(action_id): {
            "name": DAHAI_ACTION_NAMES[action_id],
            "count": int(dahai_counts.get(action_id, 0)),
        }
        for action_id in sorted(DAHAI_ACTION_NAMES)
    }

    meta["tsumo_action_breakdown"] = {
        int(action_id): {
            "name": TSUMO_ACTION_NAMES[action_id],
            "count": int(tsumo_action_counts.get(action_id, 0)),
        }
        for action_id in sorted(TSUMO_ACTION_NAMES)
    }

    torch.save(meta, OUT_DIR / "meta.pt")

    print(f"Final dataset saved to {OUT_DIR}")
    print(f"Dahai samples: {dahai_data['y'].shape[0]}")
    print(f"Tsumo samples: {tsumo_data['action_y'].shape[0]}")
    print(f"Shard count:   {len(shard_paths)}")

    print("\nDahai action breakdown:")
    for action_id in sorted(DAHAI_ACTION_NAMES):
        name = DAHAI_ACTION_NAMES[action_id]
        count = meta["dahai_action_breakdown"][action_id]["count"]
        print(f"  {action_id:>2} ({name:<10}): {count}")

    print("\nTsumo action breakdown:")
    for action_id in sorted(TSUMO_ACTION_NAMES):
        name = TSUMO_ACTION_NAMES[action_id]
        count = meta["tsumo_action_breakdown"][action_id]["count"]
        print(f"  {action_id:>2} ({name:<10}): {count}")

    print(f"\nUpdated metadata saved to {OUT_DIR / 'meta.pt'}")
