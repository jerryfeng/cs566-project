from pathlib import Path
from collections import defaultdict
import torch
import gzip
import random
import json
from torch.utils.data import Dataset
from .gamestate import ToyRoundState, pai_to_idx

def open_mjson(path):
    with open(path, "rb") as f:
        magic = f.read(2)

    if magic == b"\x1f\x8b":  # gzip magic number
        return gzip.open(path, "rt", encoding="utf-8")
    else:
        return open(path, "r", encoding="utf-8")

# ============================================================
# Parse a single mjai log and extract toy samples
# ============================================================
def extract_samples_from_gz(path: Path, max_samples_remaining: int):
    samples = []
    state = ToyRoundState()
    last_tsumo_actor = None

    with open_mjson(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            event = json.loads(line)
            etype = event.get("type")

            if etype == "start_kyoku":
                state.reset_round(event)
                last_tsumo_actor = None

            elif etype == "dora":
                marker = event.get("dora_marker")
                idx = tile_to_idx(marker)
                if idx is not None:
                    state.dora_indicators.append(idx)

            elif etype == "reach_accepted":
                actor = event.get("actor")
                if actor is not None:
                    state.riichi[actor] = 1

            elif etype == "tsumo":
                actor = event.get("actor")
                pai = event.get("pai")
                if actor is not None and pai is not None:
                    state.add_tile_to_hand(actor, pai)
                    last_tsumo_actor = actor
                else:
                    last_tsumo_actor = None

            elif etype == "dahai":
                actor = event.get("actor")
                pai = event.get("pai")
                tsumogiri = bool(event.get("tsumogiri", False))
                label = pai_to_idx(pai)

                if actor is not None and pai is not None:
                    # Create training sample only if this discard follows own tsumo
                    # This keeps the task simpler.
                    if actor == last_tsumo_actor and label is not None:
                        feat = state.to_feature(actor)
                        hand_counts = feat[0]
                        # create a mask that would be valid for discard
                        mask = hand_counts > 0
                        samples.append((feat, mask, label))

                        if len(samples) >= max_samples_remaining:
                            break

                    state.remove_tile_from_hand(actor, pai)
                    state.add_discard(actor, pai)

                last_tsumo_actor = None

            elif etype in {"pon", "chi", "daiminkan", "ankan", "kakan"}:
                # For this toy script, we are NOT updating meld state properly.
                # We only invalidate the "tsumo -> discard" chain.
                last_tsumo_actor = None

            else:
                # ignore everything else for the toy baseline
                pass

    

    return samples


# ============================================================
# Collect a tiny dataset
# ============================================================
def find_gz_files(root_dir: Path, years, max_files):
    files = []
    for year in years:
        year_dir = root_dir / year
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
