from pathlib import Path
import torch
import gzip
import random
import json
import shutil
from torch.utils.data import Dataset
from multiprocessing import Pool
from functools import partial

from gamestate import ToyRoundState, pai_to_idx, is_tenpai


def open_mjson(path):
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt", encoding="utf-8")
    else:
        return open(path, "r", encoding="utf-8")


def _can_pon(hand_counts, tile_idx):
    return hand_counts[tile_idx] >= 2


def _can_chi(hand_counts, tile_idx):
    if tile_idx >= 27:
        return False
    suit_start = (tile_idx // 9) * 9
    pos = tile_idx - suit_start
    if pos >= 2 and hand_counts[suit_start + pos - 2] > 0 and hand_counts[suit_start + pos - 1] > 0:
        return True
    if pos >= 1 and pos <= 7 and hand_counts[suit_start + pos - 1] > 0 and hand_counts[suit_start + pos + 1] > 0:
        return True
    if pos <= 6 and hand_counts[suit_start + pos + 1] > 0 and hand_counts[suit_start + pos + 2] > 0:
        return True
    return False


def _make_call_feature(state, actor, called_tile_idx):
    feat = state.to_feature(actor)
    called_plane = torch.zeros(1, 34, dtype=torch.float32)
    called_plane[0, called_tile_idx] = 1.0
    return torch.cat([feat, called_plane], dim=0)


def _make_discard_feature_and_mask(state, actor):
    feat = state.to_feature(actor)
    mask = state.legal_discard_mask(actor)
    return feat, mask


def extract_all_from_file(path, max_d=500, max_c=200, max_r=100, max_pass_per_opp=1):
    d_samples = []
    c_samples = []
    r_samples = []

    state = ToyRoundState()
    discard_pending = None
    call_pending_dahai = None
    riichi_pending_actor = None

    def _d_full():
        return max_d <= 0 or len(d_samples) >= max_d

    def _c_full():
        return max_c <= 0 or len(c_samples) >= max_c

    def _r_full():
        return max_r <= 0 or len(r_samples) >= max_r

    with open_mjson(path) as f:
        for line in f:
            if _d_full() and _c_full() and _r_full():
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
                discard_pending = None
                call_pending_dahai = None
                riichi_pending_actor = None
                state.apply_event(event)
                continue

            if etype == "tsumo":
                actor = event.get("actor")

                if call_pending_dahai is not None and not _c_full():
                    _add_pass_samples(state, call_pending_dahai, c_samples, max_pass_per_opp)
                call_pending_dahai = None

                state.apply_event(event)

                if actor is not None and not _d_full():
                    feat, mask = _make_discard_feature_and_mask(state, actor)
                    discard_pending = (actor, feat, mask)
                else:
                    discard_pending = None

                riichi_pending_actor = None
                if actor is not None and not _r_full():
                    if state.is_menzen(actor) and not state.riichi[actor] and state.can_riichi(actor):
                        riichi_pending_actor = actor
                continue

            if etype == "reach":
                actor = event.get("actor")
                if actor is not None and actor == riichi_pending_actor and not _r_full():
                    feat = state.to_feature(actor)
                    r_samples.append((feat, 1))
                riichi_pending_actor = None
                discard_pending = None
                state.apply_event(event)
                continue

            if etype == "dahai":
                actor = event.get("actor")
                pai = event.get("pai")

                if riichi_pending_actor is not None and actor == riichi_pending_actor and not _r_full():
                    feat = state.to_feature(actor)
                    r_samples.append((feat, 0))
                riichi_pending_actor = None

                if discard_pending is not None and not _d_full():
                    p_actor, feat, mask = discard_pending
                    if actor == p_actor and pai is not None:
                        label = pai_to_idx(pai)
                        if mask[label]:
                            d_samples.append((feat, mask, label))
                        else:
                            raise ValueError(
                                f"Discard label {pai} / idx={label} not in mask for actor={actor} "
                                f"while processing {path}"
                            )
                discard_pending = None

                if call_pending_dahai is not None and not _c_full():
                    _add_pass_samples(state, call_pending_dahai, c_samples, max_pass_per_opp)

                state.apply_event(event)

                if actor is not None and pai is not None and not _c_full():
                    call_pending_dahai = {
                        "actor": actor,
                        "pai": pai,
                        "pai_idx": pai_to_idx(pai),
                    }
                else:
                    call_pending_dahai = None
                continue

            if etype == "pon":
                actor = event.get("actor")
                if call_pending_dahai is not None and actor is not None and not _c_full():
                    feat = _make_call_feature(state, actor, call_pending_dahai["pai_idx"])
                    c_samples.append((feat, 1))

                call_pending_dahai = None
                discard_pending = None
                riichi_pending_actor = None

                state.apply_event(event)

                # after pon, actor must discard immediately
                if actor is not None and not _d_full():
                    feat, mask = _make_discard_feature_and_mask(state, actor)
                    discard_pending = (actor, feat, mask)
                continue

            if etype == "chi":
                actor = event.get("actor")
                if call_pending_dahai is not None and actor is not None and not _c_full():
                    feat = _make_call_feature(state, actor, call_pending_dahai["pai_idx"])
                    c_samples.append((feat, 2))

                call_pending_dahai = None
                discard_pending = None
                riichi_pending_actor = None

                state.apply_event(event)

                # after chi, actor must discard immediately
                if actor is not None and not _d_full():
                    feat, mask = _make_discard_feature_and_mask(state, actor)
                    discard_pending = (actor, feat, mask)
                continue

            if etype in {"daiminkan", "ankan", "kakan", "hora", "ryukyoku", "end_kyoku"}:
                discard_pending = None
                call_pending_dahai = None
                riichi_pending_actor = None
                state.apply_event(event)
                continue

            state.apply_event(event)

    return d_samples, c_samples, r_samples


def _add_pass_samples(state, pending_dahai, samples, max_pass):
    discarder = pending_dahai["actor"]
    pai_idx = pending_dahai["pai_idx"]
    added = 0
    for player in range(4):
        if player == discarder:
            continue
        if added >= max_pass:
            break
        hand_cnts = state.hand_counts(player)
        can_p = _can_pon(hand_cnts, pai_idx)
        can_c = (player == (discarder + 1) % 4) and _can_chi(hand_cnts, pai_idx)
        if can_p or can_c:
            feat = _make_call_feature(state, player, pai_idx)
            samples.append((feat, 0))
            added += 1


# ============================================================
# Backward-compatible single-type extractors
# ============================================================

def extract_discard_samples(path, max_samples):
    d, _, _ = extract_all_from_file(path, max_d=max_samples, max_c=0, max_r=0)
    return d


def extract_call_samples(path, max_samples, max_pass_per_opportunity=1):
    _, c, _ = extract_all_from_file(
        path,
        max_d=0,
        max_c=max_samples,
        max_r=0,
        max_pass_per_opp=max_pass_per_opportunity,
    )
    return c


def extract_riichi_samples(path, max_samples):
    _, _, r = extract_all_from_file(path, max_d=0, max_c=0, max_r=max_samples)
    return r


# ============================================================
# File discovery
# ============================================================

def find_gz_files(root_dir: Path, years, max_files):
    files = []
    for year in years:
        year_dir = root_dir / str(year)
        if not year_dir.exists():
            continue
        year_files = list(year_dir.rglob("*.mjson"))
        random.shuffle(year_files)
        files.extend(year_files)
    random.shuffle(files)
    return files[:max_files]


# ============================================================
# Packing helpers
# ============================================================

def pack_discard_samples(samples):
    if not samples:
        return {
            "x": torch.empty((0, 0, 34), dtype=torch.float32),
            "mask": torch.empty((0, 34), dtype=torch.bool),
            "y": torch.empty((0,), dtype=torch.long),
        }

    x = torch.stack([s[0] for s in samples]).float()
    mask = torch.stack([s[1] for s in samples]).bool()
    y = torch.tensor([s[2] for s in samples], dtype=torch.long)
    return {"x": x, "mask": mask, "y": y}


def pack_call_samples(samples):
    if not samples:
        return {
            "x": torch.empty((0, 0, 34), dtype=torch.float32),
            "y": torch.empty((0,), dtype=torch.long),
        }

    x = torch.stack([s[0] for s in samples]).float()
    y = torch.tensor([s[1] for s in samples], dtype=torch.long)
    return {"x": x, "y": y}


def pack_riichi_samples(samples):
    if not samples:
        return {
            "x": torch.empty((0, 0, 34), dtype=torch.float32),
            "y": torch.empty((0,), dtype=torch.long),
        }

    x = torch.stack([s[0] for s in samples]).float()
    y = torch.tensor([s[1] for s in samples], dtype=torch.long)
    return {"x": x, "y": y}


# ============================================================
# Shard save/load/merge
# ============================================================

def _save_shard(shard_path, d_samples, c_samples, r_samples, source_path=None):
    shard = {
        "discard": pack_discard_samples(d_samples),
        "call": pack_call_samples(c_samples),
        "riichi": pack_riichi_samples(r_samples),
        "counts": {
            "discard": len(d_samples),
            "call": len(c_samples),
            "riichi": len(r_samples),
        },
        "source_path": str(source_path) if source_path is not None else None,
    }
    torch.save(shard, shard_path)


def _concat_or_empty(tensors, empty_shape, dtype):
    if not tensors:
        return torch.empty(empty_shape, dtype=dtype)
    return torch.cat(tensors, dim=0)


def _merge_packed_discard(parts, max_samples):
    xs, masks, ys = [], [], []
    total = 0

    for part in parts:
        if total >= max_samples:
            break
        x = part["x"]
        mask = part["mask"]
        y = part["y"]
        take = min(x.shape[0], max_samples - total)
        if take <= 0:
            break
        xs.append(x[:take])
        masks.append(mask[:take])
        ys.append(y[:take])
        total += take

    if xs:
        return {
            "x": torch.cat(xs, dim=0),
            "mask": torch.cat(masks, dim=0),
            "y": torch.cat(ys, dim=0),
        }

    return {
        "x": torch.empty((0, 0, 34), dtype=torch.float32),
        "mask": torch.empty((0, 34), dtype=torch.bool),
        "y": torch.empty((0,), dtype=torch.long),
    }


def _merge_packed_simple(parts, max_samples):
    xs, ys = [], []
    total = 0

    for part in parts:
        if total >= max_samples:
            break
        x = part["x"]
        y = part["y"]
        take = min(x.shape[0], max_samples - total)
        if take <= 0:
            break
        xs.append(x[:take])
        ys.append(y[:take])
        total += take

    if xs:
        return {
            "x": torch.cat(xs, dim=0),
            "y": torch.cat(ys, dim=0),
        }

    return {
        "x": torch.empty((0, 0, 34), dtype=torch.float32),
        "y": torch.empty((0,), dtype=torch.long),
    }


def merge_dataset_shards(
    shard_paths,
    out_dir,
    max_discard_samples,
    max_call_samples,
    max_riichi_samples,
    meta=None,
    cleanup_shards=False,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    discard_parts = []
    call_parts = []
    riichi_parts = []

    for shard_path in shard_paths:
        shard = torch.load(shard_path, weights_only=False)
        discard_parts.append(shard["discard"])
        call_parts.append(shard["call"])
        riichi_parts.append(shard["riichi"])

    discard_data = _merge_packed_discard(discard_parts, max_discard_samples)
    call_data = _merge_packed_simple(call_parts, max_call_samples)
    riichi_data = _merge_packed_simple(riichi_parts, max_riichi_samples)

    torch.save(discard_data, out_dir / "discard.pt")
    torch.save(call_data, out_dir / "call.pt")
    torch.save(riichi_data, out_dir / "riichi.pt")

    if meta is not None:
        torch.save(meta, out_dir / "meta.pt")

    print(
        f"Merged shards -> discard={discard_data['y'].shape[0]}, "
        f"call={call_data['y'].shape[0]}, "
        f"riichi={riichi_data['y'].shape[0]}"
    )

    if cleanup_shards:
        for shard_path in shard_paths:
            try:
                Path(shard_path).unlink()
            except FileNotFoundError:
                pass

    return discard_data, call_data, riichi_data


# ============================================================
# Multiprocessing worker: scan one file and save one shard
# ============================================================

def _worker_to_shard(task):
    idx, path, per_file_d, per_file_c, per_file_r, shard_dir = task
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"shard_{idx:06d}.pt"

    try:
        d_samples, c_samples, r_samples = extract_all_from_file(
            path,
            max_d=per_file_d,
            max_c=per_file_c,
            max_r=per_file_r,
        )
        _save_shard(shard_path, d_samples, c_samples, r_samples, source_path=path)
        return {
            "ok": True,
            "shard_path": str(shard_path),
            "discard": len(d_samples),
            "call": len(c_samples),
            "riichi": len(r_samples),
        }
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return {
            "ok": False,
            "shard_path": None,
            "discard": 0,
            "call": 0,
            "riichi": 0,
        }


# ============================================================
# Main dataset builder: workers write shards, parent merges
# ============================================================

def build_dataset_shards(
    root_dir,
    years,
    max_files,
    max_discard_samples,
    max_call_samples,
    max_riichi_samples=0,
    num_workers=4,
    shard_dir="./processed_dataset/shards",
):
    """
    Workers scan files and write shard .pt files.
    Parent only tracks shard paths and counts.
    """
    files = find_gz_files(Path(root_dir), years, max_files)
    print(f"Found {len(files)} files, scanning with {num_workers} workers...")

    per_d = max(max_discard_samples // max(len(files), 1) * 3, 400) if max_discard_samples > 0 else 0
    per_c = max(max_call_samples // max(len(files), 1) * 3, 200) if max_call_samples > 0 else 0
    per_r = max(max_riichi_samples // max(len(files), 1) * 3, 100) if max_riichi_samples > 0 else 0

    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (i, path, per_d, per_c, per_r, str(shard_dir))
        for i, path in enumerate(files)
    ]

    shard_paths = []
    total_d = 0
    total_c = 0
    total_r = 0

    if num_workers <= 1:
        iterator = map(_worker_to_shard, tasks)
        pool = None
    else:
        pool = Pool(num_workers)
        iterator = pool.imap_unordered(_worker_to_shard, tasks)

    try:
        for i, result in enumerate(iterator, 1):
            if not result["ok"]:
                continue

            shard_paths.append(result["shard_path"])
            total_d += result["discard"]
            total_c += result["call"]
            total_r += result["riichi"]

            d_done = (max_discard_samples <= 0) or (total_d >= max_discard_samples)
            c_done = (max_call_samples <= 0) or (total_c >= max_call_samples)
            r_done = (max_riichi_samples <= 0) or (total_r >= max_riichi_samples)

            if i % 100 == 0:
                parts = [f"d={total_d}", f"c={total_c}"]
                if max_riichi_samples > 0:
                    parts.append(f"r={total_r}")
                print(f"Scanned {i}/{len(files)} -> {', '.join(parts)}")

            if d_done and c_done and r_done:
                print("Reached requested sample targets. Stopping worker pool early.")
                break
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()

    print(f"Shard build complete: discard={total_d}, call={total_c}, riichi={total_r}")
    return shard_paths


def build_and_save_dataset(
    root_dir,
    years,
    max_files,
    max_discard_samples,
    max_call_samples,
    max_riichi_samples=0,
    num_workers=4,
    out_dir="./processed_dataset",
    shard_subdir="shards",
    meta=None,
    cleanup_shards=False,
):
    out_dir = Path(out_dir)
    shard_dir = out_dir / shard_subdir

    shard_paths = build_dataset_shards(
        root_dir=root_dir,
        years=years,
        max_files=max_files,
        max_discard_samples=max_discard_samples,
        max_call_samples=max_call_samples,
        max_riichi_samples=max_riichi_samples,
        num_workers=num_workers,
        shard_dir=shard_dir,
    )

    discard, call, riichi = merge_dataset_shards(
        shard_paths=shard_paths,
        out_dir=out_dir,
        max_discard_samples=max_discard_samples,
        max_call_samples=max_call_samples,
        max_riichi_samples=max_riichi_samples,
        meta=meta,
        cleanup_shards=cleanup_shards,
    )

    return discard, call, riichi, shard_paths


# backward compat
def build_dataset(root_dir, years, max_files,
                  max_discard_samples, max_call_samples, max_riichi_samples=0,
                  num_workers=4):
    """
    Old API kept for compatibility.
    This now builds temp shards, merges them, then returns unpacked sample tuples.
    Avoid using this for very large datasets if you want lower RAM usage.
    """
    tmp_out = Path("./_tmp_processed_dataset_compat")
    discard, call, riichi, _ = build_and_save_dataset(
        root_dir=root_dir,
        years=years,
        max_files=max_files,
        max_discard_samples=max_discard_samples,
        max_call_samples=max_call_samples,
        max_riichi_samples=max_riichi_samples,
        num_workers=num_workers,
        out_dir=tmp_out,
        cleanup_shards=True,
    )

    d_samples = list(zip(discard["x"], discard["mask"], discard["y"].tolist()))
    c_samples = list(zip(call["x"], call["y"].tolist()))
    r_samples = list(zip(riichi["x"], riichi["y"].tolist()))
    return d_samples, c_samples, r_samples


def build_toy_dataset(root_dir, years, max_files, max_samples):
    d, _, _ = build_dataset(root_dir, years, max_files, max_samples, 0, 0, num_workers=1)
    return d


# ============================================================
# Save/load processed dataset
# ============================================================

def save_processed_dataset(
    out_dir,
    discard_samples,
    call_samples,
    riichi_samples,
    meta=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    discard_data = pack_discard_samples(discard_samples)
    call_data = pack_call_samples(call_samples)
    riichi_data = pack_riichi_samples(riichi_samples)

    torch.save(discard_data, out_dir / "discard.pt")
    torch.save(call_data, out_dir / "call.pt")
    torch.save(riichi_data, out_dir / "riichi.pt")

    if meta is not None:
        torch.save(meta, out_dir / "meta.pt")

    print(f"Saved processed dataset to {out_dir}")


def load_processed_dataset(data_dir):
    data_dir = Path(data_dir)

    discard = torch.load(data_dir / "discard.pt", weights_only=False)
    call = torch.load(data_dir / "call.pt", weights_only=False)
    riichi = torch.load(data_dir / "riichi.pt", weights_only=False)

    meta_path = data_dir / "meta.pt"
    meta = torch.load(meta_path, weights_only=False) if meta_path.exists() else None

    return discard, call, riichi, meta


# ============================================================
# PyTorch datasets
# ============================================================

class MahjongDiscardDataset(Dataset):
    def __init__(self, samples):
        self.x = [s[0] for s in samples]
        self.mask = [s[1] for s in samples]
        self.y = [s[2] for s in samples]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.mask[idx], self.y[idx]


class MahjongCallDataset(Dataset):
    def __init__(self, samples):
        self.x = [s[0] for s in samples]
        self.y = [s[1] for s in samples]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MahjongRiichiDataset(Dataset):
    def __init__(self, samples):
        self.x = [s[0] for s in samples]
        self.y = [s[1] for s in samples]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class PackedMahjongDiscardDataset(Dataset):
    def __init__(self, packed):
        self.x = packed["x"]
        self.mask = packed["mask"]
        self.y = packed["y"]

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.mask[idx], self.y[idx]


class PackedMahjongCallDataset(Dataset):
    def __init__(self, packed):
        self.x = packed["x"]
        self.y = packed["y"]

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class PackedMahjongRiichiDataset(Dataset):
    def __init__(self, packed):
        self.x = packed["x"]
        self.y = packed["y"]

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


MahjongToyDataset = MahjongDiscardDataset
