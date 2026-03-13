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

    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt", encoding="utf-8")
    else:
        return open(path, "r", encoding="utf-8")


# ============================================================
# Parse a single mjai log and extract toy samples
# ============================================================

def extract_samples_from_gz(path: Path, max_samples_remaining: int):
    samples = []
    state = ToyRoundState()

    # pending sample after a tsumo:
    # (actor, feature, mask)
    pending = None

    with open_mjson(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            event = json.loads(line)
            etype = event.get("type")

            if etype == "start_kyoku":
                pending = None
                state.apply_event(event)
                continue

            # If calls / win / draw interrupt the tsumo->dahai chain, drop pending
            if etype in {"chi", "pon", "daiminkan", "ankan", "kakan", "hora", "ryukyoku", "end_kyoku"}:
                pending = None
                state.apply_event(event)
                continue

            if etype == "tsumo":
                actor = event.get("actor")

                # update state first so hand includes drawn tile
                state.apply_event(event)

                if actor is not None:
                    feat = state.to_feature(actor)
                    mask = feat[0] > 0  # plane 0 = hand counts
                    pending = (actor, feat, mask)
                else:
                    pending = None

                continue

            if etype == "dahai":
                actor = event.get("actor")
                pai = event.get("pai")

                if pending is not None:
                    pending_actor, feat, mask = pending

                    if actor == pending_actor and pai is not None:
                        label = pai_to_idx(pai)
                        samples.append((feat, mask, label))

                        if len(samples) >= max_samples_remaining:
                            break

                # now update state with the discard
                state.apply_event(event)
                pending = None
                continue

            # all other events: just apply if supported
            state.apply_event(event)

    return samples


# ============================================================
# Collect a tiny dataset
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


def build_toy_dataset(root_dir: Path, years, max_files, max_samples):
    files = find_gz_files(root_dir, years, max_files)
    print(f"Found {len(files)} files to scan.")

    all_samples = []
    for i, path in enumerate(files, 1):
        remaining = max_samples - len(all_samples)
        if remaining <= 0:
            break

        samples = extract_samples_from_gz(path, remaining)
        all_samples.extend(samples)

        if i % 5 == 0 or i == len(files):
            print(f"Scanned {i}/{len(files)} files -> {len(all_samples)} samples")

    return all_samples


# ============================================================
# PyTorch dataset
# ============================================================

class MahjongToyDataset(Dataset):
    def __init__(self, samples):
        self.x = [s[0] for s in samples]
        self.mask = [s[1] for s in samples]
        self.y = [s[2] for s in samples]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.mask[idx], self.y[idx]
