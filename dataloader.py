from pathlib import Path
import torch
import gzip
import random
import json
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


# ============================================================
# Single-pass extraction: discard + call + riichi from one file
# ============================================================

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


def extract_all_from_file(path, max_d=500, max_c=200, max_r=100, max_pass_per_opp=1):
    """
    Single pass through one mjai log file.
    Extracts discard, call, and riichi samples simultaneously.
    ~3x faster than three separate passes.
    """
    d_samples = []
    c_samples = []
    r_samples = []

    state = ToyRoundState()
    discard_pending = None
    call_pending_dahai = None
    riichi_pending_actor = None

    def _d_full():
        return len(d_samples) >= max_d
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

                # call: previous dahai not called -> pass samples
                if call_pending_dahai is not None and not _c_full():
                    _add_pass_samples(state, call_pending_dahai, c_samples, max_pass_per_opp)
                call_pending_dahai = None

                state.apply_event(event)

                # discard: prepare pending
                if actor is not None and not _d_full():
                    feat = state.to_feature(actor)
                    mask = feat[0] > 0
                    discard_pending = (actor, feat, mask)
                else:
                    discard_pending = None

                # riichi: check eligibility
                riichi_pending_actor = None
                if actor is not None and not _r_full():
                    if (state.is_menzen(actor) and
                            not state.riichi[actor] and
                            state.can_riichi(actor)):
                        riichi_pending_actor = actor
                continue

            if etype == "reach":
                actor = event.get("actor")
                if (actor is not None and
                        actor == riichi_pending_actor and
                        not _r_full()):
                    feat = state.to_feature(actor)
                    r_samples.append((feat, 1))
                riichi_pending_actor = None
                discard_pending = None
                state.apply_event(event)
                continue

            if etype == "dahai":
                actor = event.get("actor")
                pai = event.get("pai")

                # riichi negative: could but didn't
                if (riichi_pending_actor is not None and
                        actor == riichi_pending_actor and
                        not _r_full()):
                    feat = state.to_feature(actor)
                    r_samples.append((feat, 0))
                riichi_pending_actor = None

                # discard sample
                if discard_pending is not None and not _d_full():
                    p_actor, feat, mask = discard_pending
                    if actor == p_actor and pai is not None:
                        label = pai_to_idx(pai)
                        if label is not None:
                            d_samples.append((feat, mask, label))
                discard_pending = None

                # call: pass samples for previous dahai
                if call_pending_dahai is not None and not _c_full():
                    _add_pass_samples(state, call_pending_dahai, c_samples, max_pass_per_opp)

                state.apply_event(event)

                # call: new pending
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
                if (call_pending_dahai is not None and
                        actor is not None and not _c_full()):
                    feat = _make_call_feature(state, actor, call_pending_dahai["pai_idx"])
                    if feat is not None:
                        c_samples.append((feat, 1))
                call_pending_dahai = None
                discard_pending = None
                riichi_pending_actor = None
                state.apply_event(event)
                continue

            if etype == "chi":
                actor = event.get("actor")
                if (call_pending_dahai is not None and
                        actor is not None and not _c_full()):
                    feat = _make_call_feature(state, actor, call_pending_dahai["pai_idx"])
                    if feat is not None:
                        c_samples.append((feat, 2))
                call_pending_dahai = None
                discard_pending = None
                riichi_pending_actor = None
                state.apply_event(event)
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
            if feat is not None:
                samples.append((feat, 0))
                added += 1


# ============================================================
# Backward-compatible single-type extractors (for testing)
# ============================================================

def extract_discard_samples(path, max_samples):
    d, _, _ = extract_all_from_file(path, max_d=max_samples, max_c=0, max_r=0)
    return d

def extract_call_samples(path, max_samples, max_pass_per_opportunity=1):
    _, c, _ = extract_all_from_file(path, max_d=0, max_c=max_samples, max_r=0,
                                     max_pass_per_opp=max_pass_per_opportunity)
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
# Multiprocessing worker
# ============================================================

def _worker(path, per_file_d, per_file_c, per_file_r):
    try:
        return extract_all_from_file(path, per_file_d, per_file_c, per_file_r)
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return [], [], []


# ============================================================
# Main dataset builder: single-pass + multiprocessing
# ============================================================

def build_dataset(root_dir, years, max_files,
                  max_discard_samples, max_call_samples, max_riichi_samples=0,
                  num_workers=4):
    """
    Build discard, call, and riichi datasets.
    Uses single-pass extraction + multiprocessing for speed.

    On M1 Mac: num_workers=6 recommended
    On Colab CPU: num_workers=2
    On Colab GPU: num_workers=2
    """
    files = find_gz_files(Path(root_dir), years, max_files)
    print(f"Found {len(files)} files, scanning with {num_workers} workers...")

    # per-file caps
    per_d = max(max_discard_samples // max(len(files), 1) * 3, 500)
    per_c = max(max_call_samples // max(len(files), 1) * 3, 200)
    per_r = max(max_riichi_samples // max(len(files), 1) * 3, 100) if max_riichi_samples > 0 else 0

    worker = partial(_worker, per_file_d=per_d, per_file_c=per_c, per_file_r=per_r)

    all_d, all_c, all_r = [], [], []

    if num_workers <= 1:
        iterator = map(worker, files)
    else:
        pool = Pool(num_workers)
        iterator = pool.imap_unordered(worker, files)

    try:
        for i, (d, c, r) in enumerate(iterator, 1):
            all_d.extend(d)
            all_c.extend(c)
            all_r.extend(r)

            d_done = len(all_d) >= max_discard_samples
            c_done = len(all_c) >= max_call_samples
            r_done = (max_riichi_samples == 0) or (len(all_r) >= max_riichi_samples)

            if d_done and c_done and r_done:
                break

            if i % 100 == 0:
                parts = [f"d={len(all_d)}", f"c={len(all_c)}"]
                if max_riichi_samples > 0:
                    parts.append(f"r={len(all_r)}")
                print(f"Scanned {i}/{len(files)} -> {', '.join(parts)}")
    finally:
        if num_workers > 1:
            pool.terminate()
            pool.join()

    all_d = all_d[:max_discard_samples]
    all_c = all_c[:max_call_samples]
    if max_riichi_samples > 0:
        all_r = all_r[:max_riichi_samples]

    print(f"Done: discard={len(all_d)}, call={len(all_c)}, riichi={len(all_r)}")
    return all_d, all_c, all_r


# backward compat
def build_toy_dataset(root_dir, years, max_files, max_samples):
    d, _, _ = build_dataset(root_dir, years, max_files, max_samples, 0, 0, num_workers=1)
    return d


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


MahjongToyDataset = MahjongDiscardDataset