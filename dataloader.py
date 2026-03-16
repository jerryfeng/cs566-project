from pathlib import Path
import torch
import gzip
import random
import json
from torch.utils.data import Dataset

from gamestate import ToyRoundState, pai_to_idx, is_tenpai


def open_mjson(path):
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt", encoding="utf-8")
    else:
        return open(path, "r", encoding="utf-8")


# ============================================================
# Extract discard samples (tsumo -> dahai)
# ============================================================

def extract_discard_samples(path: Path, max_samples: int):
    samples = []
    state = ToyRoundState()
    pending = None

    with open_mjson(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type")

            if etype == "start_kyoku":
                pending = None
                state.apply_event(event)
                continue

            if etype in {"chi", "pon", "daiminkan", "ankan", "kakan",
                         "hora", "ryukyoku", "end_kyoku"}:
                pending = None
                state.apply_event(event)
                continue

            if etype == "tsumo":
                actor = event.get("actor")
                state.apply_event(event)
                if actor is not None:
                    feat = state.to_feature(actor)
                    mask = feat[0] > 0
                    pending = (actor, feat, mask)
                else:
                    pending = None
                continue

            if etype == "dahai":
                actor = event.get("actor")
                pai = event.get("pai")

                if pending is not None:
                    p_actor, feat, mask = pending
                    if actor == p_actor and pai is not None:
                        label = pai_to_idx(pai)
                        if label is not None:
                            samples.append((feat, mask, label))
                            if len(samples) >= max_samples:
                                break

                state.apply_event(event)
                pending = None
                continue

            state.apply_event(event)

    return samples


# ============================================================
# Extract call samples (opponent dahai -> pon/chi/pass)
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


def extract_call_samples(path: Path, max_samples: int, max_pass_per_opportunity: int = 1):
    samples = []
    state = ToyRoundState()
    pending_dahai = None

    with open_mjson(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type")

            if etype == "start_kyoku":
                pending_dahai = None
                state.apply_event(event)
                continue

            if etype == "tsumo":
                if pending_dahai is not None:
                    _add_pass_samples(state, pending_dahai, samples, max_pass_per_opportunity)
                    pending_dahai = None
                state.apply_event(event)
                continue

            if etype == "dahai":
                if pending_dahai is not None:
                    _add_pass_samples(state, pending_dahai, samples, max_pass_per_opportunity)

                actor = event.get("actor")
                pai = event.get("pai")
                state.apply_event(event)

                if actor is not None and pai is not None:
                    pending_dahai = {
                        "actor": actor,
                        "pai": pai,
                        "pai_idx": pai_to_idx(pai),
                    }
                else:
                    pending_dahai = None

                if len(samples) >= max_samples:
                    break
                continue

            if etype == "pon":
                actor = event.get("actor")
                if pending_dahai is not None and actor is not None:
                    feat = _make_call_feature(state, actor, pending_dahai["pai_idx"])
                    if feat is not None:
                        samples.append((feat, 1))
                pending_dahai = None
                state.apply_event(event)
                if len(samples) >= max_samples:
                    break
                continue

            if etype == "chi":
                actor = event.get("actor")
                if pending_dahai is not None and actor is not None:
                    feat = _make_call_feature(state, actor, pending_dahai["pai_idx"])
                    if feat is not None:
                        samples.append((feat, 2))
                pending_dahai = None
                state.apply_event(event)
                if len(samples) >= max_samples:
                    break
                continue

            if etype in {"daiminkan", "ankan", "kakan", "hora", "ryukyoku", "end_kyoku"}:
                pending_dahai = None
                state.apply_event(event)
                continue

            state.apply_event(event)

    return samples


def _make_call_feature(state, actor, called_tile_idx):
    feat = state.to_feature(actor)
    called_plane = torch.zeros(1, 34, dtype=torch.float32)
    called_plane[0, called_tile_idx] = 1.0
    return torch.cat([feat, called_plane], dim=0)


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
# Extract riichi samples
# ============================================================

def extract_riichi_samples(path: Path, max_samples: int):
    """
    Extract riichi decision samples.

    Positive: player declared reach (menzen + tenpai and chose to riichi)
    Negative: player was menzen + tenpai after tsumo but did NOT riichi (dama / kept playing)

    Returns list of (feature[16,34], label):
    - label 0 = did not riichi (dama or continued)
    - label 1 = declared riichi
    """
    samples = []
    state = ToyRoundState()

    # track: after tsumo, did the player declare reach before dahai?
    pending_tsumo_actor = None

    with open_mjson(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type")

            if etype == "start_kyoku":
                pending_tsumo_actor = None
                state.apply_event(event)
                continue

            if etype == "tsumo":
                actor = event.get("actor")
                state.apply_event(event)

                # check if this player could riichi
                if actor is not None and state.is_menzen(actor) and not state.riichi[actor]:
                    if state.can_riichi(actor):
                        pending_tsumo_actor = actor
                    else:
                        pending_tsumo_actor = None
                else:
                    pending_tsumo_actor = None
                continue

            if etype == "reach":
                # positive sample: player chose to riichi
                actor = event.get("actor")
                if actor is not None and actor == pending_tsumo_actor:
                    feat = state.to_feature(actor)
                    samples.append((feat, 1))  # 1 = riichi
                pending_tsumo_actor = None
                state.apply_event(event)
                if len(samples) >= max_samples:
                    break
                continue

            if etype == "dahai":
                # if pending_tsumo_actor set but no reach came -> negative sample
                actor = event.get("actor")
                if pending_tsumo_actor is not None and actor == pending_tsumo_actor:
                    feat = state.to_feature(actor)
                    samples.append((feat, 0))  # 0 = no riichi (dama)
                pending_tsumo_actor = None
                state.apply_event(event)
                if len(samples) >= max_samples:
                    break
                continue

            if etype in {"chi", "pon", "daiminkan", "ankan", "kakan",
                         "hora", "ryukyoku", "end_kyoku"}:
                pending_tsumo_actor = None
                state.apply_event(event)
                continue

            state.apply_event(event)

    return samples


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


def build_dataset(root_dir, years, max_files,
                  max_discard_samples, max_call_samples, max_riichi_samples=0):
    """
    Build discard, call, and riichi datasets from mjai logs.
    Returns (discard_samples, call_samples, riichi_samples).
    """
    files = find_gz_files(Path(root_dir), years, max_files)
    print(f"Found {len(files)} files to scan.")

    all_discard = []
    all_call = []
    all_riichi = []

    for i, path in enumerate(files, 1):
        d_rem = max_discard_samples - len(all_discard)
        c_rem = max_call_samples - len(all_call)
        r_rem = max_riichi_samples - len(all_riichi) if max_riichi_samples > 0 else 0

        if d_rem <= 0 and c_rem <= 0 and r_rem <= 0:
            break

        try:
            if d_rem > 0:
                all_discard.extend(extract_discard_samples(path, d_rem))
            if c_rem > 0:
                all_call.extend(extract_call_samples(path, c_rem))
            if r_rem > 0:
                all_riichi.extend(extract_riichi_samples(path, r_rem))
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue

        if i % 50 == 0 or i == len(files):
            parts = [f"discard={len(all_discard)}", f"call={len(all_call)}"]
            if max_riichi_samples > 0:
                parts.append(f"riichi={len(all_riichi)}")
            print(f"Scanned {i}/{len(files)} -> {', '.join(parts)}")

    return all_discard, all_call, all_riichi


# backward compat
def build_toy_dataset(root_dir, years, max_files, max_samples):
    files = find_gz_files(Path(root_dir), years, max_files)
    print(f"Found {len(files)} files to scan.")
    all_samples = []
    for i, path in enumerate(files, 1):
        remaining = max_samples - len(all_samples)
        if remaining <= 0:
            break
        try:
            all_samples.extend(extract_discard_samples(path, remaining))
        except Exception as e:
            print(f"Error processing {path}: {e}")
        if i % 50 == 0 or i == len(files):
            print(f"Scanned {i}/{len(files)} -> {len(all_samples)} samples")
    return all_samples


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
    """Dataset for riichi decisions: (feat[16,34], label 0/1)"""
    def __init__(self, samples):
        self.x = [s[0] for s in samples]
        self.y = [s[1] for s in samples]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


MahjongToyDataset = MahjongDiscardDataset