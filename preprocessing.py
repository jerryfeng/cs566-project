from pathlib import Path
import kagglehub

from dataloader import build_and_save_dataset

YEARS = [2023, 2024]
MAX_FILES = 6000
MAX_DISCARD_SAMPLES = 1_000_000
MAX_CALL_SAMPLES = 400_000
MAX_RIICHI_SAMPLES = 100_000
NUM_DATA_LOADER_WORKERS = 16

DATASET_HANDLE = "shokanekolouis/tenhou-to-mjai"
DATASET_PATH = Path(kagglehub.dataset_download(DATASET_HANDLE))

OUT_DIR = Path("./processed_dataset")
CLEANUP_SHARDS_AFTER_MERGE = False


if __name__ == "__main__":
    meta = {
        "root_dir": str(DATASET_PATH),
        "years": YEARS,
        "max_files": MAX_FILES,
        "max_discard_samples": MAX_DISCARD_SAMPLES,
        "max_call_samples": MAX_CALL_SAMPLES,
        "max_riichi_samples": MAX_RIICHI_SAMPLES,
        "num_workers": NUM_DATA_LOADER_WORKERS,
        "dataset_handle": DATASET_HANDLE,
    }

    discard_data, call_data, riichi_data, shard_paths = build_and_save_dataset(
        root_dir=DATASET_PATH,
        years=YEARS,
        max_files=MAX_FILES,
        max_discard_samples=MAX_DISCARD_SAMPLES,
        max_call_samples=MAX_CALL_SAMPLES,
        max_riichi_samples=MAX_RIICHI_SAMPLES,
        num_workers=NUM_DATA_LOADER_WORKERS,
        out_dir=OUT_DIR,
        shard_subdir="shards",
        meta=meta,
        cleanup_shards=CLEANUP_SHARDS_AFTER_MERGE,
    )

    print(f"Final dataset saved to {OUT_DIR}")
    print(f"Discard samples: {discard_data['y'].shape[0]}")
    print(f"Call samples:    {call_data['y'].shape[0]}")
    print(f"Riichi samples:  {riichi_data['y'].shape[0]}")
    print(f"Shard count:     {len(shard_paths)}")
