from pathlib import Path
import torch
import gzip
import random
import json
import copy
import math
import tempfile
import numpy as np
from torch.utils.data import Dataset
from multiprocessing import Pool

from gamestate import (
    RoundState,
    pai_to_idx,
    idx_to_pai,
    NUM_TILES,
    NUM_CALL_KINDS,
    NUM_TSUMO_ACTIONS,
    CALL_KIND_TO_IDX,
    TSUMO_ACTION_TO_IDX,
)

DEFAULT_HIST_LEN = 64
IGNORE_INDEX = -100

DAHAI_ACTION_NAMES = {
    CALL_KIND_TO_IDX["none"]: "none",
    CALL_KIND_TO_IDX["chi_low"]: "chi_low",
    CALL_KIND_TO_IDX["chi_mid"]: "chi_mid",
    CALL_KIND_TO_IDX["chi_high"]: "chi_high",
    CALL_KIND_TO_IDX["pon"]: "pon",
    CALL_KIND_TO_IDX["kan"]: "kan",
    CALL_KIND_TO_IDX["hora"]: "hora",
}

TSUMO_ACTION_NAMES = {
    TSUMO_ACTION_TO_IDX["none"]: "none",
    TSUMO_ACTION_TO_IDX["dahai"]: "dahai",
    TSUMO_ACTION_TO_IDX["reach"]: "reach",
    TSUMO_ACTION_TO_IDX["kan"]: "kan",
    TSUMO_ACTION_TO_IDX["hora"]: "hora",
}


def open_mjson(path):
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _make_feature(state: RoundState, actor: int):
    feat = state.to_feature(actor)
    hist, hist_mask = state.get_history(observer=actor, max_len=DEFAULT_HIST_LEN)
    return feat, hist, hist_mask


def _make_dahai_sample(state: RoundState, actor: int, action_mask: torch.Tensor, label: int):
    feat, hist, hist_mask = _make_feature(state, actor)
    return feat, action_mask.clone(), hist, hist_mask, int(label)


def _make_tsumo_sample(
    state: RoundState,
    actor: int,
    action_mask: torch.Tensor,
    tile_mask: torch.Tensor,
    action_label: int,
    tile_label: int,
):
    feat, hist, hist_mask = _make_feature(state, actor)
    return (
        feat,
        action_mask.clone(),
        tile_mask.clone(),
        hist,
        hist_mask,
        int(action_label),
        int(tile_label),
    )


def _tile_mask_for_action(masks: dict, action_label: int) -> torch.Tensor:
    if action_label == TSUMO_ACTION_TO_IDX["dahai"]:
        return masks["discard_mask"].clone()
    if action_label == TSUMO_ACTION_TO_IDX["reach"]:
        return masks["reach_mask"].clone()
    if action_label == TSUMO_ACTION_TO_IDX["kan"]:
        return masks["kan_mask"].clone()
    return torch.zeros(NUM_TILES, dtype=torch.bool)


def _resolve_tsumo_action_from_event(state: RoundState, event: dict, pending: dict):
    actor = pending["actor"]
    etype = event["type"]
    masks = pending["masks"]

    if etype == "hora" and event.get("actor") == actor:
        return TSUMO_ACTION_TO_IDX["hora"], IGNORE_INDEX, _tile_mask_for_action(masks, TSUMO_ACTION_TO_IDX["hora"])

    if etype == "ankan" and event.get("actor") == actor:
        consumed = event.get("consumed", [])
        tile = pai_to_idx(consumed[0]) if consumed else IGNORE_INDEX
        return TSUMO_ACTION_TO_IDX["kan"], tile, _tile_mask_for_action(masks, TSUMO_ACTION_TO_IDX["kan"])

    if etype == "kakan" and event.get("actor") == actor:
        pai = event.get("pai")
        tile = pai_to_idx(pai) if pai is not None else IGNORE_INDEX
        return TSUMO_ACTION_TO_IDX["kan"], tile, _tile_mask_for_action(masks, TSUMO_ACTION_TO_IDX["kan"])

    if etype == "reach" and event.get("actor") == actor:
        pending["saw_reach"] = True
        return None

    if etype == "dahai" and event.get("actor") == actor:
        pai = event.get("pai")
        tile = pai_to_idx(pai) if pai is not None else IGNORE_INDEX
        if pending.get("saw_reach", False):
            return TSUMO_ACTION_TO_IDX["reach"], tile, _tile_mask_for_action(masks, TSUMO_ACTION_TO_IDX["reach"])
        return TSUMO_ACTION_TO_IDX["dahai"], tile, _tile_mask_for_action(masks, TSUMO_ACTION_TO_IDX["dahai"])

    return None


def extract_all_from_file(
    path,
    max_dahai_samples=500,
    max_tsumo_samples=500,
    hist_len=DEFAULT_HIST_LEN,
):
    dahai_samples = []
    tsumo_samples = []

    state = RoundState()
    pending_dahai = None
    pending_tsumo = None

    # Per-file worker tasks pass 0 when that branch should collect nothing.
    # Negative values are treated as uncapped; positive values are capped.
    def dahai_enabled():
        return max_dahai_samples != 0

    def tsumo_enabled():
        return max_tsumo_samples != 0

    def dahai_full():
        return max_dahai_samples > 0 and len(dahai_samples) >= max_dahai_samples

    def tsumo_full():
        return max_tsumo_samples > 0 and len(tsumo_samples) >= max_tsumo_samples

    def can_collect_dahai():
        return dahai_enabled() and not dahai_full()

    def can_collect_tsumo():
        return tsumo_enabled() and not tsumo_full()

    def flush_pending_dahai_as_pass():
        nonlocal pending_dahai
        if pending_dahai is None or not can_collect_dahai():
            pending_dahai = None
            return
        for actor, payload in pending_dahai["by_actor"].items():
            if not can_collect_dahai():
                break
            if payload["resolved"]:
                continue
            dahai_samples.append(
                _make_dahai_sample(
                    state=payload["snapshot_state"],
                    actor=actor,
                    action_mask=payload["mask"],
                    label=CALL_KIND_TO_IDX["none"],
                )
            )
            payload["resolved"] = True
        pending_dahai = None

    def flush_pending_tsumo_skip():
        nonlocal pending_tsumo
        pending_tsumo = None

    with open_mjson(path) as f:
        for line in f:
            if dahai_full() and tsumo_full():
                break

            line = line.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type")

            if etype == "start_kyoku":
                flush_pending_dahai_as_pass()
                flush_pending_tsumo_skip()
                state.apply_event(event)
                continue

            # Resolve pending reaction labels before state mutation if needed
            if pending_dahai is not None and etype in {"chi", "pon", "daiminkan", "hora", "tsumo", "end_kyoku", "ryukyoku"}:
                if etype in {"chi", "pon", "daiminkan"}:
                    actor = event.get("actor")
                    if actor in pending_dahai["by_actor"]:
                        payload = pending_dahai["by_actor"][actor]
                        if not payload["resolved"] and can_collect_dahai():
                            action_mask = payload["mask"]
                            if etype == "chi":
                                label_name = RoundState.classify_chi_from_event(event)
                            elif etype == "pon":
                                label_name = "pon"
                            else:
                                label_name = "kan"
                            if action_mask[CALL_KIND_TO_IDX[label_name]]:
                                dahai_samples.append(
                                    _make_dahai_sample(
                                        state=payload["snapshot_state"],
                                        actor=actor,
                                        action_mask=action_mask,
                                        label=CALL_KIND_TO_IDX[label_name],
                                    )
                                )
                            payload["resolved"] = True
                    flush_pending_dahai_as_pass()
                elif etype == "hora":
                    actor = event.get("actor")
                    if actor in pending_dahai["by_actor"]:
                        payload = pending_dahai["by_actor"][actor]
                        action_mask = payload["mask"]
                        if not payload["resolved"] and can_collect_dahai() and action_mask[CALL_KIND_TO_IDX["hora"]]:
                            dahai_samples.append(
                                _make_dahai_sample(
                                    state=payload["snapshot_state"],
                                    actor=actor,
                                    action_mask=action_mask,
                                    label=CALL_KIND_TO_IDX["hora"],
                                )
                            )
                            payload["resolved"] = True
                    # multiple ron support: do not flush immediately here; if another hora follows it can still resolve
                    # flush on next non-hora or end of kyoku
                else:
                    flush_pending_dahai_as_pass()

            if pending_tsumo is not None:
                resolved = _resolve_tsumo_action_from_event(state, event, pending_tsumo)
                if resolved is not None:
                    if can_collect_tsumo():
                        action_label, tile_label, tile_mask = resolved
                        action_mask = pending_tsumo["masks"]["action_mask"]
                        if action_mask[action_label] and (tile_label == IGNORE_INDEX or tile_mask[tile_label]):
                            tsumo_samples.append(
                                _make_tsumo_sample(
                                    state=pending_tsumo["snapshot_state"],
                                    actor=pending_tsumo["actor"],
                                    action_mask=action_mask,
                                    tile_mask=tile_mask,
                                    action_label=action_label,
                                    tile_label=tile_label,
                                )
                            )
                    pending_tsumo = None
                elif etype in {"tsumo", "start_kyoku", "end_kyoku", "ryukyoku"} and pending_tsumo is not None:
                    flush_pending_tsumo_skip()

            state.apply_event(event)

            if etype == "dahai" and can_collect_dahai():
                discarder = event.get("actor")
                by_actor = {}
                for player in range(4):
                    if player == discarder:
                        continue
                    mask = state.legal_dahai_reaction_mask(player)
                    if mask.sum().item() > 1:
                        by_actor[player] = {
                            "mask": mask,
                            "resolved": False,
                            "snapshot_state": copy.deepcopy(state),
                        }
                pending_dahai = {"by_actor": by_actor} if by_actor else None
                continue

            if etype == "tsumo" and can_collect_tsumo():
                actor = event.get("actor")
                masks = state.legal_tsumo_action_masks(actor)
                pending_tsumo = {
                    "actor": actor,
                    "masks": masks,
                    "snapshot_state": copy.deepcopy(state),
                    "saw_reach": False,
                }
                continue

            if etype in {"end_kyoku", "ryukyoku"}:
                flush_pending_dahai_as_pass()
                flush_pending_tsumo_skip()

    flush_pending_dahai_as_pass()
    flush_pending_tsumo_skip()
    return dahai_samples, tsumo_samples


def find_gz_files(root_dir: Path, max_files):
    files = list(root_dir.rglob("*.mjson"))
    random.shuffle(files)
    return files[:max_files]


def pack_dahai_samples(samples):
    if not samples:
        return {
            "x": torch.empty((0, 31, NUM_TILES), dtype=torch.float32),
            "mask": torch.empty((0, NUM_CALL_KINDS), dtype=torch.bool),
            "hist": torch.empty((0, DEFAULT_HIST_LEN, 8), dtype=torch.long),
            "hist_mask": torch.empty((0, DEFAULT_HIST_LEN), dtype=torch.bool),
            "y": torch.empty((0,), dtype=torch.long),
        }

    return {
        "x": torch.stack([s[0] for s in samples]).float(),
        "mask": torch.stack([s[1] for s in samples]).bool(),
        "hist": torch.stack([s[2] for s in samples]).long(),
        "hist_mask": torch.stack([s[3] for s in samples]).bool(),
        "y": torch.tensor([s[4] for s in samples], dtype=torch.long),
    }


def pack_tsumo_samples(samples):
    if not samples:
        return {
            "x": torch.empty((0, 31, NUM_TILES), dtype=torch.float32),
            "action_mask": torch.empty((0, NUM_TSUMO_ACTIONS), dtype=torch.bool),
            "tile_mask": torch.empty((0, NUM_TILES), dtype=torch.bool),
            "hist": torch.empty((0, DEFAULT_HIST_LEN, 8), dtype=torch.long),
            "hist_mask": torch.empty((0, DEFAULT_HIST_LEN), dtype=torch.bool),
            "action_y": torch.empty((0,), dtype=torch.long),
            "tile_y": torch.empty((0,), dtype=torch.long),
        }

    return {
        "x": torch.stack([s[0] for s in samples]).float(),
        "action_mask": torch.stack([s[1] for s in samples]).bool(),
        "tile_mask": torch.stack([s[2] for s in samples]).bool(),
        "hist": torch.stack([s[3] for s in samples]).long(),
        "hist_mask": torch.stack([s[4] for s in samples]).bool(),
        "action_y": torch.tensor([s[5] for s in samples], dtype=torch.long),
        "tile_y": torch.tensor([s[6] for s in samples], dtype=torch.long),
    }


def _save_shard(shard_path, dahai_samples, tsumo_samples, source_path=None):
    shard = {
        "dahai": pack_dahai_samples(dahai_samples),
        "tsumo": pack_tsumo_samples(tsumo_samples),
        "counts": {
            "dahai": len(dahai_samples),
            "tsumo": len(tsumo_samples),
        },
        "source_path": str(source_path) if source_path is not None else None,
    }
    torch.save(shard, shard_path)


def _take_from_part(part, key, take):
    return part[key][:take] if take > 0 else part[key][:0]


def _get_packed_count(part, label_key):
    return int(part[label_key].shape[0])


def _prepare_memmap_tensor(base_dir: Path, name: str, shape, dtype: torch.dtype):
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"{name}.mmap"
    np_dtype = {
        torch.float32: np.float32,
        torch.bool: np.bool_,
        torch.long: np.int64,
    }[dtype]
    mm = np.memmap(path, mode="w+", dtype=np_dtype, shape=shape)
    return path, mm


def _copy_tensor_into_memmap(mm, start: int, src: torch.Tensor):
    if src.numel() == 0:
        return
    arr = src.detach().cpu().numpy()
    mm[start:start + arr.shape[0]] = arr


def _close_memmap(mm):
    base = getattr(mm, "base", None)
    mmap_obj = getattr(base, "close", None)
    if callable(mmap_obj):
        base.close()
    elif hasattr(mm, "_mmap") and mm._mmap is not None:
        mm._mmap.close()


def _finalize_memmap_tensor(mm_path: Path, shape, dtype: torch.dtype):
    np_dtype = {
        torch.float32: np.float32,
        torch.bool: np.bool_,
        torch.long: np.int64,
    }[dtype]
    mm = np.memmap(mm_path, mode="r", dtype=np_dtype, shape=shape)
    # np.array(..., copy=True) avoids the non-writable NumPy warning from torch.from_numpy
    arr = np.array(mm, copy=True)
    _close_memmap(mm)
    del mm
    tensor = torch.from_numpy(arr)
    try:
        mm_path.unlink()
    except FileNotFoundError:
        pass
    return tensor


def _merge_dataset_incremental(shard_paths, spec, max_samples, temp_dir: Path):
    if max_samples == 0:
        return spec["empty_fn"]()

    total = 0
    for shard_path in shard_paths:
        shard = torch.load(shard_path, weights_only=False)
        part = shard[spec["shard_key"]]
        part_n = _get_packed_count(part, spec["label_key"])
        if max_samples > 0:
            take = min(part_n, max_samples - total)
        else:
            take = part_n
        total += max(take, 0)
        if max_samples > 0 and total >= max_samples:
            break

    if total == 0:
        return spec["empty_fn"]()

    mm_specs = {}
    for key, shape_fn, dtype in spec["fields"]:
        shape = shape_fn(total)
        mm_path, mm = _prepare_memmap_tensor(temp_dir / spec["shard_key"], key, shape, dtype)
        mm_specs[key] = {
            "path": mm_path,
            "mm": mm,
            "shape": shape,
            "dtype": dtype,
        }

    written = 0
    for shard_path in shard_paths:
        if written >= total:
            break
        shard = torch.load(shard_path, weights_only=False)
        part = shard[spec["shard_key"]]
        part_n = _get_packed_count(part, spec["label_key"])
        take = min(part_n, total - written)
        if take <= 0:
            continue
        for key, _, _ in spec["fields"]:
            _copy_tensor_into_memmap(mm_specs[key]["mm"], written, _take_from_part(part, key, take))
        written += take

    merged = {}
    for key, _, _ in spec["fields"]:
        info = mm_specs[key]
        info["mm"].flush()
        _close_memmap(info["mm"])
        del info["mm"]
        merged[key] = _finalize_memmap_tensor(info["path"], info["shape"], info["dtype"])

    return merged


def merge_dataset_shards(
    shard_paths,
    out_dir,
    max_dahai_samples,
    max_tsumo_samples,
    meta=None,
    cleanup_shards=False,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = out_dir / "_merge_tmp"

    dahai_spec = {
        "shard_key": "dahai",
        "label_key": "y",
        "empty_fn": pack_dahai_samples,
        "fields": [
            ("x", lambda n: (n, 31, NUM_TILES), torch.float32),
            ("mask", lambda n: (n, NUM_CALL_KINDS), torch.bool),
            ("hist", lambda n: (n, DEFAULT_HIST_LEN, 8), torch.long),
            ("hist_mask", lambda n: (n, DEFAULT_HIST_LEN), torch.bool),
            ("y", lambda n: (n,), torch.long),
        ],
    }
    tsumo_spec = {
        "shard_key": "tsumo",
        "label_key": "action_y",
        "empty_fn": pack_tsumo_samples,
        "fields": [
            ("x", lambda n: (n, 31, NUM_TILES), torch.float32),
            ("action_mask", lambda n: (n, NUM_TSUMO_ACTIONS), torch.bool),
            ("tile_mask", lambda n: (n, NUM_TILES), torch.bool),
            ("hist", lambda n: (n, DEFAULT_HIST_LEN, 8), torch.long),
            ("hist_mask", lambda n: (n, DEFAULT_HIST_LEN), torch.bool),
            ("action_y", lambda n: (n,), torch.long),
            ("tile_y", lambda n: (n,), torch.long),
        ],
    }

    dahai_data = _merge_dataset_incremental(shard_paths, dahai_spec, max_dahai_samples, temp_dir)
    tsumo_data = _merge_dataset_incremental(shard_paths, tsumo_spec, max_tsumo_samples, temp_dir)

    torch.save(dahai_data, out_dir / "dahai.pt")
    torch.save(tsumo_data, out_dir / "tsumo.pt")

    if meta is not None:
        torch.save(meta, out_dir / "meta.pt")

    print(
        f"Merged shards -> dahai={dahai_data['y'].shape[0]}, "
        f"tsumo={tsumo_data['action_y'].shape[0]}"
    )

    if cleanup_shards:
        for shard_path in shard_paths:
            try:
                Path(shard_path).unlink()
            except FileNotFoundError:
                pass

    if temp_dir.exists():
        for p in temp_dir.rglob("*"):
            if p.is_file():
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
        for p in sorted(temp_dir.rglob("*"), reverse=True):
            if p.is_dir():
                try:
                    p.rmdir()
                except OSError:
                    pass
        try:
            temp_dir.rmdir()
        except OSError:
            pass

    return dahai_data, tsumo_data


def _worker_to_shard(task):
    idx, path, per_file_dahai, per_file_tsumo, shard_dir, hist_len = task
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"shard_{idx:06d}.pt"

    try:
        dahai_samples, tsumo_samples = extract_all_from_file(
            path,
            max_dahai_samples=per_file_dahai,
            max_tsumo_samples=per_file_tsumo,
            hist_len=hist_len,
        )
        _save_shard(shard_path, dahai_samples, tsumo_samples, source_path=path)
        return {
            "ok": True,
            "shard_path": str(shard_path),
            "dahai": len(dahai_samples),
            "tsumo": len(tsumo_samples),
        }
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return {
            "ok": False,
            "shard_path": None,
            "dahai": 0,
            "tsumo": 0,
        }


def build_dataset_shards(
    root_dir,
    max_files,
    max_dahai_samples,
    max_tsumo_samples,
    num_workers=4,
    shard_dir="./processed_dataset/shards",
    hist_len=DEFAULT_HIST_LEN,
):
    files = find_gz_files(Path(root_dir), max_files)
    print(f"Found {len(files)} files, scanning with {num_workers} workers...")

    base_per_dahai = max(max_dahai_samples // max(len(files), 1) * 3, 500) if max_dahai_samples > 0 else 0
    base_per_tsumo = max(max_tsumo_samples // max(len(files), 1) * 3, 500) if max_tsumo_samples > 0 else 0

    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_paths = []
    total_dahai = 0
    total_tsumo = 0

    def dahai_done():
        return max_dahai_samples > 0 and total_dahai >= max_dahai_samples

    def tsumo_done():
        return max_tsumo_samples > 0 and total_tsumo >= max_tsumo_samples

    def both_done():
        d_ok = max_dahai_samples <= 0 or total_dahai >= max_dahai_samples
        t_ok = max_tsumo_samples <= 0 or total_tsumo >= max_tsumo_samples
        return d_ok and t_ok

    def make_task(file_idx):
        remaining_dahai = max(0, max_dahai_samples - total_dahai) if max_dahai_samples > 0 else 0
        remaining_tsumo = max(0, max_tsumo_samples - total_tsumo) if max_tsumo_samples > 0 else 0

        per_dahai = 0 if dahai_done() else min(base_per_dahai, remaining_dahai)
        per_tsumo = 0 if tsumo_done() else min(base_per_tsumo, remaining_tsumo)
        return (file_idx, files[file_idx], per_dahai, per_tsumo, str(shard_dir), hist_len)

    if num_workers <= 1:
        scanned = 0
        for file_idx in range(len(files)):
            if both_done():
                print("Reached requested sample targets. Stopping early.")
                break
            result = _worker_to_shard(make_task(file_idx))
            scanned += 1

            if not result["ok"]:
                continue
            shard_paths.append(result["shard_path"])
            total_dahai += result["dahai"]
            total_tsumo += result["tsumo"]

            if scanned % 100 == 0:
                print(f"Scanned {scanned}/{len(files)} -> dahai={total_dahai}, tsumo={total_tsumo}")
    else:
        pool = Pool(num_workers)
        pending = []
        next_file_idx = 0
        scanned = 0

        try:
            while next_file_idx < len(files) and len(pending) < num_workers and not both_done():
                pending.append(pool.apply_async(_worker_to_shard, (make_task(next_file_idx),)))
                next_file_idx += 1

            while pending:
                new_pending = []
                for job in pending:
                    result = job.get()
                    scanned += 1

                    if result["ok"]:
                        shard_paths.append(result["shard_path"])
                        total_dahai += result["dahai"]
                        total_tsumo += result["tsumo"]

                    if scanned % 100 == 0:
                        print(f"Scanned {scanned}/{len(files)} -> dahai={total_dahai}, tsumo={total_tsumo}")

                    if not both_done() and next_file_idx < len(files):
                        new_pending.append(pool.apply_async(_worker_to_shard, (make_task(next_file_idx),)))
                        next_file_idx += 1

                pending = new_pending

                if both_done():
                    print("Reached requested sample targets. Stopping worker pool early.")
                    break
        finally:
            pool.terminate()
            pool.join()

    print(f"Shard build complete: dahai={total_dahai}, tsumo={total_tsumo}")
    return shard_paths


def build_and_save_dataset(
    root_dir,
    max_files,
    max_dahai_samples,
    max_tsumo_samples,
    num_workers=4,
    out_dir="./processed_dataset",
    shard_subdir="shards",
    meta=None,
    cleanup_shards=False,
    hist_len=DEFAULT_HIST_LEN,
):
    out_dir = Path(out_dir)
    shard_dir = out_dir / shard_subdir

    shard_paths = build_dataset_shards(
        root_dir=root_dir,
        max_files=max_files,
        max_dahai_samples=max_dahai_samples,
        max_tsumo_samples=max_tsumo_samples,
        num_workers=num_workers,
        shard_dir=shard_dir,
        hist_len=hist_len,
    )
    # shard_paths = [f"C:/Users/houbo/Desktop/cs566-project/processed_dataset/shards/shard_{i:06}.pt" for i in range(7131)]

    dahai, tsumo = merge_dataset_shards(
        shard_paths=shard_paths,
        out_dir=out_dir,
        max_dahai_samples=max_dahai_samples,
        max_tsumo_samples=max_tsumo_samples,
        meta=meta,
        cleanup_shards=cleanup_shards,
    )

    return dahai, tsumo, shard_paths


def save_processed_dataset(out_dir, dahai_samples, tsumo_samples, meta=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dahai_data = pack_dahai_samples(dahai_samples)
    tsumo_data = pack_tsumo_samples(tsumo_samples)

    torch.save(dahai_data, out_dir / "dahai.pt")
    torch.save(tsumo_data, out_dir / "tsumo.pt")

    if meta is not None:
        torch.save(meta, out_dir / "meta.pt")

    print(f"Saved processed dataset to {out_dir}")


def load_processed_dahai_dataset(data_dir):
    data_dir = Path(data_dir)
    data = torch.load(data_dir / "dahai.pt", weights_only=False)
    meta_path = data_dir / "meta.pt"
    meta = torch.load(meta_path, weights_only=False) if meta_path.exists() else None
    return data, meta


def load_processed_tsumo_dataset(data_dir):
    data_dir = Path(data_dir)
    data = torch.load(data_dir / "tsumo.pt", weights_only=False)
    meta_path = data_dir / "meta.pt"
    meta = torch.load(meta_path, weights_only=False) if meta_path.exists() else None
    return data, meta


class MahjongDahaiDataset(Dataset):
    def __init__(self, packed):
        self.x = packed["x"]
        self.mask = packed["mask"]
        self.hist = packed["hist"]
        self.hist_mask = packed["hist_mask"]
        self.y = packed["y"]

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.mask[idx], self.hist[idx], self.hist_mask[idx], self.y[idx]


class MahjongTsumoDataset(Dataset):
    def __init__(self, packed):
        self.x = packed["x"]
        self.action_mask = packed["action_mask"]
        self.tile_mask = packed["tile_mask"]
        self.hist = packed["hist"]
        self.hist_mask = packed["hist_mask"]
        self.action_y = packed["action_y"]
        self.tile_y = packed["tile_y"]

    def __len__(self):
        return self.action_y.shape[0]

    def __getitem__(self, idx):
        return (
            self.x[idx],
            self.action_mask[idx],
            self.tile_mask[idx],
            self.hist[idx],
            self.hist_mask[idx],
            self.action_y[idx],
            self.tile_y[idx],
        )
