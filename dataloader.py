from pathlib import Path
import torch
import gzip
import random
import json
from torch.utils.data import Dataset

from gamestate import ToyRoundState, pai_to_idx


def open_mjson(path):
    with open(path, "rb") as f:
        magic = f.read(2)

    if magic == b"\x1f\x8b":  # gzip
        return gzip.open(path, "rt", encoding="utf-8")
    else:
        return open(path, "r", encoding="utf-8")


# ============================================================
# Extract discard samples (tsumo -> dahai)
# ============================================================

def extract_discard_samples(path: Path, max_samples: int):
    """
    Extract (feature, mask, label) for discard decisions.
    feature: [16, 34], mask: [34] bool, label: int (0-33)
    """
    samples = []
    state = ToyRoundState()
    pending = None  # (actor, feat, mask)

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
    """Check if player has >= 2 of the tile (can pon)."""
    return hand_counts[tile_idx] >= 2


def _can_chi(hand_counts, tile_idx):
    """
    Check if player can chi (form a sequence) with the tile.
    Only works for number tiles (idx 0-26), not honors (27-33).
    """
    if tile_idx >= 27:
        return False

    suit_start = (tile_idx // 9) * 9
    pos = tile_idx - suit_start  # 0-8 within suit

    # check three possible sequences: (pos-2,pos-1,pos), (pos-1,pos,pos+1), (pos,pos+1,pos+2)
    if pos >= 2 and hand_counts[suit_start + pos - 2] > 0 and hand_counts[suit_start + pos - 1] > 0:
        return True
    if pos >= 1 and pos <= 7 and hand_counts[suit_start + pos - 1] > 0 and hand_counts[suit_start + pos + 1] > 0:
        return True
    if pos <= 6 and hand_counts[suit_start + pos + 1] > 0 and hand_counts[suit_start + pos + 2] > 0:
        return True

    return False


def extract_call_samples(path: Path, max_samples: int, max_pass_per_opportunity: int = 1):
    """
    Extract call decision samples from one mjai log.

    Returns list of (feature_with_called_tile, label):
    - feature_with_called_tile: [17, 34] tensor (16 game channels + 1 called tile one-hot)
    - label: 0=pass, 1=pon, 2=chi

    To control class imbalance, max_pass_per_opportunity limits how many
    pass samples we generate per discard event (default 1).
    """
    samples = []
    state = ToyRoundState()
    pending_dahai = None  # {"actor": int, "pai": str, "pai_idx": int}

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
                # someone drew a tile -> previous discard was not called
                # generate pass samples for players who could have called
                if pending_dahai is not None:
                    _add_pass_samples(state, pending_dahai, samples, max_pass_per_opportunity)
                    pending_dahai = None

                state.apply_event(event)
                continue

            if etype == "dahai":
                # process previous pending first
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
                # positive sample: actor chose to pon
                actor = event.get("actor")
                if pending_dahai is not None and actor is not None:
                    feat = _make_call_feature(state, actor, pending_dahai["pai_idx"])
                    if feat is not None:
                        samples.append((feat, 1))  # 1 = pon

                pending_dahai = None
                state.apply_event(event)

                if len(samples) >= max_samples:
                    break
                continue

            if etype == "chi":
                # positive sample: actor chose to chi
                actor = event.get("actor")
                if pending_dahai is not None and actor is not None:
                    feat = _make_call_feature(state, actor, pending_dahai["pai_idx"])
                    if feat is not None:
                        samples.append((feat, 2))  # 2 = chi

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


def _make_call_feature(state: ToyRoundState, actor: int, called_tile_idx: int):
    """
    Build [17, 34] feature: 16 game channels + 1 called tile one-hot.
    """
    feat = state.to_feature(actor)  # [16, 34]
    called_plane = torch.zeros(1, 34, dtype=torch.float32)
    called_plane[0, called_tile_idx] = 1.0
    return torch.cat([feat, called_plane], dim=0)  # [17, 34]


def _add_pass_samples(state, pending_dahai, samples, max_pass):
    """
    For each player who could have called (pon/chi) but didn't, add a pass sample.
    """
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
        # chi: only kamicha (previous seat) can chi
        can_c = (player == (discarder + 1) % 4) and _can_chi(hand_cnts, pai_idx)

        if can_p or can_c:
            feat = _make_call_feature(state, player, pai_idx)
            if feat is not None:
                samples.append((feat, 0))  # 0 = pass
                added += 1


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


def build_dataset(root_dir, years, max_files, max_discard_samples, max_call_samples):
    """
    Build both discard and call datasets from mjai logs.
    Returns (discard_samples, call_samples).
    """
    files = find_gz_files(Path(root_dir), years, max_files)
    print(f"Found {len(files)} files to scan.")

    all_discard = []
    all_call = []

    for i, path in enumerate(files, 1):
        d_remaining = max_discard_samples - len(all_discard)
        c_remaining = max_call_samples - len(all_call)

        if d_remaining <= 0 and c_remaining <= 0:
            break

        try:
            if d_remaining > 0:
                d_samples = extract_discard_samples(path, d_remaining)
                all_discard.extend(d_samples)

            if c_remaining > 0:
                c_samples = extract_call_samples(path, c_remaining)
                all_call.extend(c_samples)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue

        if i % 50 == 0 or i == len(files):
            print(f"Scanned {i}/{len(files)} -> discard={len(all_discard)}, call={len(all_call)}")

    return all_discard, all_call


# For backward compatibility
def build_toy_dataset(root_dir, years, max_files, max_samples):
    files = find_gz_files(Path(root_dir), years, max_files)
    print(f"Found {len(files)} files to scan.")
    all_samples = []
    for i, path in enumerate(files, 1):
        remaining = max_samples - len(all_samples)
        if remaining <= 0:
            break
        try:
            samples = extract_discard_samples(path, remaining)
            all_samples.extend(samples)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
        if i % 50 == 0 or i == len(files):
            print(f"Scanned {i}/{len(files)} -> {len(all_samples)} samples")
    return all_samples


# ============================================================
# PyTorch datasets
# ============================================================

class MahjongDiscardDataset(Dataset):
    """Dataset for discard decisions: (feat[16,34], mask[34], label int)"""
    def __init__(self, samples):
        self.x = [s[0] for s in samples]
        self.mask = [s[1] for s in samples]
        self.y = [s[2] for s in samples]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.mask[idx], self.y[idx]


class MahjongCallDataset(Dataset):
    """Dataset for call decisions: (feat[17,34], label int 0/1/2)"""
    def __init__(self, samples):
        self.x = [s[0] for s in samples]
        self.y = [s[1] for s in samples]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# backward compat alias
MahjongToyDataset = MahjongDiscardDataset