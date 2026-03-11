from pathlib import Path
from collections import defaultdict
import torch
import gzip
import random
import json
from torch.utils.data import Dataset

# ============================================================
# Tile mapping: 34 classes
# ============================================================
TILE_STRS = (
    [f"{i}m" for i in range(1, 10)] +
    [f"{i}p" for i in range(1, 10)] +
    [f"{i}s" for i in range(1, 10)] +
    [f"{i}z" for i in range(1, 8)]
)
TILE2IDX = {t: i for i, t in enumerate(TILE_STRS)}

def normalize_tile(tile: str) -> str:
    """
    Convert red fives like '5mr', '5pr', '5sr' into '5m', '5p', '5s'.
    If tile is already normal, return as-is.
    """
    if not isinstance(tile, str):
        return tile
    if tile.endswith("r") and len(tile) >= 2:
        return tile[:2]
    return tile


def tile_to_idx(tile: str):
    tile = normalize_tile(tile)
    return TILE2IDX.get(tile, None)

# ============================================================
# Tiny game state tracker
# ============================================================
class ToyRoundState:
    def __init__(self):
        self.hands = [defaultdict(int) for _ in range(4)]
        self.discards = [defaultdict(int) for _ in range(4)]
        self.riichi = [0, 0, 0, 0]
        self.dora_indicators = []
        self.bakaze = "E"
        self.kyoku = 1
        self.honba = 0
        self.kyotaku = 0

    def reset_round(self, event):
        self.hands = [defaultdict(int) for _ in range(4)]
        self.discards = [defaultdict(int) for _ in range(4)]
        self.riichi = [0, 0, 0, 0]
        self.dora_indicators = []

        self.bakaze = event.get("bakaze", "E")
        self.kyoku = int(event.get("kyoku", 1))
        self.honba = int(event.get("honba", 0))
        self.kyotaku = int(event.get("kyotaku", 0))

        # Most mjai logs have "tehais" at start_kyoku
        tehais = event.get("tehais")
        if isinstance(tehais, list) and len(tehais) == 4:
            for actor in range(4):
                for tile in tehais[actor]:
                    idx = tile_to_idx(tile)
                    if idx is not None:
                        self.hands[actor][idx] += 1

        dora = event.get("dora_marker")
        if dora is not None:
            idx = tile_to_idx(dora)
            if idx is not None:
                self.dora_indicators.append(idx)

    def add_tile_to_hand(self, actor, tile):
        idx = tile_to_idx(tile)
        if idx is not None:
            self.hands[actor][idx] += 1

    def remove_tile_from_hand(self, actor, tile):
        idx = tile_to_idx(tile)
        if idx is not None:
            self.hands[actor][idx] -= 1
            if self.hands[actor][idx] < 0:
                self.hands[actor][idx] = 0

    def add_discard(self, actor, tile):
        idx = tile_to_idx(tile)
        if idx is not None:
            self.discards[actor][idx] += 1

    def to_feature(self, actor):
        """
        Return tensor shape [6, 34]
        """
        x = torch.zeros(10, 34, dtype=torch.float32)

        # plane 0: current player's hand counts
        for idx, cnt in self.hands[actor].items():
            x[0, idx] = cnt

        # plane 1: current player's discards
        for idx, cnt in self.discards[actor].items():
            x[1, idx] = cnt

        # plane 2-4: opponents' discards combined
        i = 0
        for other in range(4):
            if other == actor:
                continue
            for idx, cnt in self.discards[other].items():
                x[2 + i, idx] += cnt
            i += 1

        # plane 5: dora indicators
        for idx in self.dora_indicators:
            x[5, idx] = 1.0

        # plane 6-8: riichi flags broadcast
        i = 0
        for other in range(4):
            if other == actor:
                continue
            x[6 + i, :] = float(self.riichi[other])
            i += 1

        # plane 9: simple round summary broadcast
        bakaze_val = {"E": 0.0, "S": 1.0, "W": 2.0, "N": 3.0}.get(self.bakaze, 0.0)
        summary = bakaze_val * 0.1 + self.kyoku * 0.1 + self.honba * 0.01 + self.kyotaku * 0.01
        x[9, :] = summary

        return x

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
                label = tile_to_idx(pai)

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
