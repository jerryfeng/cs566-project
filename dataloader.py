from pathlib import Path
import torch
import gzip
import random
import json
from torch.utils.data import Dataset
from multiprocessing import Pool

from gamestate import RoundState, pai_to_idx

# ============================================================
# Config
# ============================================================

DEFAULT_HIST_LEN = 64

# ============================================================
# Action space for merged "call" dataset
# ============================================================

CALL_PASS = 0
CALL_CHI = 1
CALL_PON = 2
CALL_HORA = 3
CALL_DAIMINKAN = 4
CALL_ANKAN = 5
CALL_KAKAN = 6
CALL_RIICHI = 7

NUM_CALL_ACTIONS = 8

CALL_ACTION_NAMES = {
    CALL_PASS: "none",
    CALL_CHI: "chi",
    CALL_PON: "pon",
    CALL_HORA: "hora",
    CALL_DAIMINKAN: "daiminkan",
    CALL_ANKAN: "ankan",
    CALL_KAKAN: "kakan",
    CALL_RIICHI: "riichi",
}


def open_mjson(path):
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt", encoding="utf-8")
    else:
        return open(path, "r", encoding="utf-8")


# ============================================================
# Call legality helpers
# ============================================================

def _can_pon(hand_counts, tile_idx):
    return hand_counts[tile_idx] >= 2


def _can_daiminkan(hand_counts, tile_idx):
    return hand_counts[tile_idx] >= 3


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


def _can_ankan(state: RoundState, actor: int) -> bool:
    hand_counts = state.hand_counts(actor)
    return any(c >= 4 for c in hand_counts)


def _can_kakan(state: RoundState, actor: int) -> bool:
    # Approximation based on current state representation:
    # if player has 1 in hand and already has 3 copies exposed in meld tiles,
    # we treat it as a possible kakan.
    hand_counts = state.hand_counts(actor)
    meld_counts = state.melds[actor]
    for i in range(34):
        if hand_counts[i] >= 1 and meld_counts[i] == 3:
            return True
    return False


# ============================================================
# Feature builders
# ============================================================

def _make_call_feature(state, actor, called_tile_idx=None):
    feat = state.to_feature(actor)
    called_plane = torch.zeros(1, 34, dtype=torch.float32)
    if called_tile_idx is not None:
        called_plane[0, called_tile_idx] = 1.0
    return torch.cat([feat, called_plane], dim=0)


def _make_discard_feature_and_mask(state, actor):
    feat = state.to_feature(actor)
    mask = state.legal_discard_mask(actor)
    return feat, mask


def _make_discard_sample(state, actor, feat, discard_mask, label, hist_len=DEFAULT_HIST_LEN):
    hist, hist_mask = state.get_history(observer=actor, max_len=hist_len)
    return feat, discard_mask.clone(), hist, hist_mask, int(label)


def _make_call_sample(state, actor, action_mask, label, called_tile_idx=None, hist_len=DEFAULT_HIST_LEN):
    feat = _make_call_feature(state, actor, called_tile_idx)
    hist, hist_mask = state.get_history(observer=actor, max_len=hist_len)
    return feat, action_mask.clone(), hist, hist_mask, int(label)


def _build_discard_reaction_mask(state: RoundState, player: int, discarder: int, pai: str, pai_idx: int):
    hand_counts = state.hand_counts(player)
    mask = torch.zeros(NUM_CALL_ACTIONS, dtype=torch.bool)

    can_hora = state.check_ron(player, pai)
    can_pon = _can_pon(hand_counts, pai_idx)
    can_daiminkan = _can_daiminkan(hand_counts, pai_idx)
    can_chi = (player == (discarder + 1) % 4) and _can_chi(hand_counts, pai_idx)

    mask[CALL_CHI] = can_chi
    mask[CALL_PON] = can_pon
    mask[CALL_HORA] = can_hora
    mask[CALL_DAIMINKAN] = can_daiminkan

    if mask.any():
        mask[CALL_PASS] = True

    return mask


def _build_self_action_mask(state: RoundState, actor: int):
    mask = torch.zeros(NUM_CALL_ACTIONS, dtype=torch.bool)

    mask[CALL_HORA] = state.check_tsumo_agari(actor)
    mask[CALL_ANKAN] = _can_ankan(state, actor)
    mask[CALL_KAKAN] = _can_kakan(state, actor)
    mask[CALL_RIICHI] = state.can_riichi(actor)

    if mask.any():
        mask[CALL_PASS] = True

    return mask


# ============================================================
# Main extraction
# ============================================================

def extract_all_from_file(path, max_d=500, max_c=300, max_pass_per_opp=1, hist_len=DEFAULT_HIST_LEN):
    """
    Returns:
      d_samples: list[(feat, discard_mask, hist, hist_mask, discard_label)]
      c_samples: list[(feat, call_action_mask, hist, hist_mask, call_label)]
    """
    d_samples = []
    c_samples = []

    state = RoundState()
    discard_pending = None

    call_pending_dahai = None
    pending_self_action = None

    def _d_full():
        return max_d <= 0 or len(d_samples) >= max_d

    def _c_full():
        return max_c <= 0 or len(c_samples) >= max_c

    def _flush_pending_discard_reactions():
        nonlocal call_pending_dahai
        if call_pending_dahai is None or _c_full():
            call_pending_dahai = None
            return

        _add_discard_pass_samples(
            state=state,
            pending_dahai=call_pending_dahai,
            samples=c_samples,
            max_pass=max_pass_per_opp,
            hist_len=hist_len,
        )
        call_pending_dahai = None

    def _flush_pending_self_action_as_pass():
        nonlocal pending_self_action
        if pending_self_action is None or _c_full():
            pending_self_action = None
            return

        actor = pending_self_action["actor"]
        mask = pending_self_action["mask"]
        if mask[CALL_PASS]:
            c_samples.append(
                _make_call_sample(
                    state, actor, mask, CALL_PASS,
                    called_tile_idx=None,
                    hist_len=hist_len,
                )
            )
        pending_self_action = None

    with open_mjson(path) as f:
        for line in f:
            if _d_full() and _c_full():
                break

            line = line.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type")

            # ---------------------------------------------------
            # start_kyoku
            # ---------------------------------------------------
            if etype == "start_kyoku":
                _flush_pending_discard_reactions()
                _flush_pending_self_action_as_pass()

                discard_pending = None
                call_pending_dahai = None
                pending_self_action = None
                state.apply_event(event)
                continue

            # ---------------------------------------------------
            # tsumo
            # ---------------------------------------------------
            if etype == "tsumo":
                actor = event.get("actor")

                _flush_pending_discard_reactions()
                _flush_pending_self_action_as_pass()

                state.apply_event(event)

                if actor is not None and not _d_full():
                    feat, mask = _make_discard_feature_and_mask(state, actor)
                    discard_pending = (actor, feat, mask)
                else:
                    discard_pending = None

                pending_self_action = None
                if actor is not None and not _c_full():
                    self_mask = _build_self_action_mask(state, actor)
                    if self_mask.any():
                        pending_self_action = {
                            "actor": actor,
                            "mask": self_mask,
                        }
                continue

            # ---------------------------------------------------
            # reach -> merged into call actions
            # ---------------------------------------------------
            if etype == "reach":
                actor = event.get("actor")

                if pending_self_action is not None and actor == pending_self_action["actor"] and not _c_full():
                    mask = pending_self_action["mask"]
                    if mask[CALL_RIICHI]:
                        c_samples.append(
                            _make_call_sample(
                                state, actor, mask, CALL_RIICHI,
                                called_tile_idx=None,
                                hist_len=hist_len,
                            )
                        )
                    elif mask[CALL_PASS]:
                        c_samples.append(
                            _make_call_sample(
                                state, actor, mask, CALL_PASS,
                                called_tile_idx=None,
                                hist_len=hist_len,
                            )
                        )

                pending_self_action = None
                discard_pending = None
                state.apply_event(event)
                continue

            # ---------------------------------------------------
            # dahai
            # ---------------------------------------------------
            if etype == "dahai":
                actor = event.get("actor")
                pai = event.get("pai")

                if pending_self_action is not None and actor == pending_self_action["actor"] and not _c_full():
                    mask = pending_self_action["mask"]
                    if mask[CALL_PASS]:
                        c_samples.append(
                            _make_call_sample(
                                state, actor, mask, CALL_PASS,
                                called_tile_idx=None,
                                hist_len=hist_len,
                            )
                        )
                    pending_self_action = None
                else:
                    _flush_pending_self_action_as_pass()

                if discard_pending is not None and not _d_full():
                    p_actor, feat, mask = discard_pending
                    if actor == p_actor and pai is not None:
                        label = pai_to_idx(pai)
                        if mask[label]:
                            d_samples.append(
                                _make_discard_sample(
                                    state, actor, feat, mask, label,
                                    hist_len=hist_len,
                                )
                            )
                        else:
                            raise ValueError(
                                f"Discard label {pai} / idx={label} not in mask for actor={actor} "
                                f"while processing {path}"
                            )
                discard_pending = None

                _flush_pending_discard_reactions()

                state.apply_event(event)

                if actor is not None and pai is not None and not _c_full():
                    call_pending_dahai = {
                        "actor": actor,
                        "pai": pai,
                        "pai_idx": pai_to_idx(pai),
                        "resolved": set(),
                    }
                else:
                    call_pending_dahai = None
                continue

            # ---------------------------------------------------
            # chi / pon / daiminkan from another player's discard
            # ---------------------------------------------------
            if etype in {"chi", "pon", "daiminkan"}:
                actor = event.get("actor")

                if call_pending_dahai is not None and actor is not None and not _c_full():
                    discarder = call_pending_dahai["actor"]
                    pai = call_pending_dahai["pai"]
                    pai_idx = call_pending_dahai["pai_idx"]

                    mask = _build_discard_reaction_mask(state, actor, discarder, pai, pai_idx)

                    label = {
                        "chi": CALL_CHI,
                        "pon": CALL_PON,
                        "daiminkan": CALL_DAIMINKAN,
                    }[etype]

                    if mask[label]:
                        c_samples.append(
                            _make_call_sample(
                                state, actor, mask, label,
                                called_tile_idx=pai_idx,
                                hist_len=hist_len,
                            )
                        )
                        call_pending_dahai["resolved"].add(actor)

                    _add_discard_pass_samples(
                        state=state,
                        pending_dahai=call_pending_dahai,
                        samples=c_samples,
                        max_pass=max_pass_per_opp,
                        hist_len=hist_len,
                    )

                call_pending_dahai = None
                discard_pending = None
                pending_self_action = None

                state.apply_event(event)

                if etype in {"chi", "pon"} and actor is not None and not _d_full():
                    feat, mask = _make_discard_feature_and_mask(state, actor)
                    discard_pending = (actor, feat, mask)

                continue

            # ---------------------------------------------------
            # ankan / kakan from self draw
            # ---------------------------------------------------
            if etype in {"ankan", "kakan"}:
                actor = event.get("actor")

                if pending_self_action is not None and actor == pending_self_action["actor"] and not _c_full():
                    mask = pending_self_action["mask"]
                    label = CALL_ANKAN if etype == "ankan" else CALL_KAKAN
                    if mask[label]:
                        c_samples.append(
                            _make_call_sample(
                                state, actor, mask, label,
                                called_tile_idx=None,
                                hist_len=hist_len,
                            )
                        )
                    elif mask[CALL_PASS]:
                        c_samples.append(
                            _make_call_sample(
                                state, actor, mask, CALL_PASS,
                                called_tile_idx=None,
                                hist_len=hist_len,
                            )
                        )

                pending_self_action = None
                discard_pending = None

                state.apply_event(event)
                continue

            # ---------------------------------------------------
            # hora: either ron on a discard, or tsumo on self draw
            # ---------------------------------------------------
            if etype == "hora":
                actor = event.get("actor")

                handled = False

                if (
                    call_pending_dahai is not None
                    and actor is not None
                    and actor != call_pending_dahai["actor"]
                    and not _c_full()
                ):
                    discarder = call_pending_dahai["actor"]
                    pai = call_pending_dahai["pai"]
                    pai_idx = call_pending_dahai["pai_idx"]

                    mask = _build_discard_reaction_mask(state, actor, discarder, pai, pai_idx)
                    if mask[CALL_HORA]:
                        c_samples.append(
                            _make_call_sample(
                                state, actor, mask, CALL_HORA,
                                called_tile_idx=pai_idx,
                                hist_len=hist_len,
                            )
                        )
                        call_pending_dahai["resolved"].add(actor)
                        handled = True

                if (
                    not handled
                    and pending_self_action is not None
                    and actor is not None
                    and actor == pending_self_action["actor"]
                    and not _c_full()
                ):
                    mask = pending_self_action["mask"]
                    if mask[CALL_HORA]:
                        c_samples.append(
                            _make_call_sample(
                                state, actor, mask, CALL_HORA,
                                called_tile_idx=None,
                                hist_len=hist_len,
                            )
                        )
                    elif mask[CALL_PASS]:
                        c_samples.append(
                            _make_call_sample(
                                state, actor, mask, CALL_PASS,
                                called_tile_idx=None,
                                hist_len=hist_len,
                            )
                        )
                    pending_self_action = None
                    discard_pending = None
                    handled = True

                if (
                    not handled
                    and pending_self_action is not None
                    and actor is not None
                    and actor == pending_self_action["actor"]
                    and not _c_full()
                ):
                    mask = pending_self_action["mask"]
                    if mask[CALL_HORA]:
                        c_samples.append(
                            _make_call_sample(
                                state, actor, mask, CALL_HORA,
                                called_tile_idx=None,
                                hist_len=hist_len,
                            )
                        )
                    pending_self_action = None
                    discard_pending = None

                state.apply_event(event)
                continue

            # ---------------------------------------------------
            # end_kyoku
            # ---------------------------------------------------
            if etype == "end_kyoku":
                _flush_pending_self_action_as_pass()
                _flush_pending_discard_reactions()

                discard_pending = None
                state.apply_event(event)
                continue

            # ---------------------------------------------------
            # other events
            # ---------------------------------------------------
            state.apply_event(event)

    if not _c_full():
        if pending_self_action is not None:
            actor = pending_self_action["actor"]
            mask = pending_self_action["mask"]
            if mask[CALL_PASS]:
                c_samples.append(
                    _make_call_sample(
                        state, actor, mask, CALL_PASS,
                        called_tile_idx=None,
                        hist_len=hist_len,
                    )
                )

        if call_pending_dahai is not None:
            _add_discard_pass_samples(
                state=state,
                pending_dahai=call_pending_dahai,
                samples=c_samples,
                max_pass=max_pass_per_opp,
                hist_len=hist_len,
            )

    return d_samples, c_samples


def _add_discard_pass_samples(state, pending_dahai, samples, max_pass, hist_len=DEFAULT_HIST_LEN):
    discarder = pending_dahai["actor"]
    pai = pending_dahai["pai"]
    pai_idx = pending_dahai["pai_idx"]
    resolved = pending_dahai.get("resolved", set())

    added = 0
    for player in range(4):
        if player == discarder:
            continue
        if player in resolved:
            continue
        if added >= max_pass:
            break

        mask = _build_discard_reaction_mask(state, player, discarder, pai, pai_idx)
        if mask[CALL_PASS]:
            samples.append(
                _make_call_sample(
                    state, player, mask, CALL_PASS,
                    called_tile_idx=pai_idx,
                    hist_len=hist_len,
                )
            )
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


# ============================================================
# Packing helpers
# ============================================================

def pack_discard_samples(samples):
    if not samples:
        return {
            "x": torch.empty((0, 0, 34), dtype=torch.float32),
            "mask": torch.empty((0, 34), dtype=torch.bool),
            "hist": torch.empty((0, DEFAULT_HIST_LEN, 8), dtype=torch.long),
            "hist_mask": torch.empty((0, DEFAULT_HIST_LEN), dtype=torch.bool),
            "y": torch.empty((0,), dtype=torch.long),
        }

    x = torch.stack([s[0] for s in samples]).float()
    mask = torch.stack([s[1] for s in samples]).bool()
    hist = torch.stack([s[2] for s in samples]).long()
    hist_mask = torch.stack([s[3] for s in samples]).bool()
    y = torch.tensor([s[4] for s in samples], dtype=torch.long)
    return {"x": x, "mask": mask, "hist": hist, "hist_mask": hist_mask, "y": y}


def pack_call_samples(samples):
    if not samples:
        return {
            "x": torch.empty((0, 0, 34), dtype=torch.float32),
            "mask": torch.empty((0, NUM_CALL_ACTIONS), dtype=torch.bool),
            "hist": torch.empty((0, DEFAULT_HIST_LEN, 8), dtype=torch.long),
            "hist_mask": torch.empty((0, DEFAULT_HIST_LEN), dtype=torch.bool),
            "y": torch.empty((0,), dtype=torch.long),
        }

    x = torch.stack([s[0] for s in samples]).float()
    mask = torch.stack([s[1] for s in samples]).bool()
    hist = torch.stack([s[2] for s in samples]).long()
    hist_mask = torch.stack([s[3] for s in samples]).bool()
    y = torch.tensor([s[4] for s in samples], dtype=torch.long)
    return {"x": x, "mask": mask, "hist": hist, "hist_mask": hist_mask, "y": y}


# ============================================================
# Shard save/load/merge
# ============================================================

def _save_shard(shard_path, d_samples, c_samples, source_path=None):
    shard = {
        "discard": pack_discard_samples(d_samples),
        "call": pack_call_samples(c_samples),
        "counts": {
            "discard": len(d_samples),
            "call": len(c_samples),
        },
        "source_path": str(source_path) if source_path is not None else None,
    }
    torch.save(shard, shard_path)


def _merge_packed_discard(parts, max_samples):
    xs, masks, hists, hist_masks, ys = [], [], [], [], []
    total = 0

    for part in parts:
        if total >= max_samples:
            break
        x = part["x"]
        mask = part["mask"]
        hist = part["hist"]
        hist_mask = part["hist_mask"]
        y = part["y"]

        take = min(x.shape[0], max_samples - total)
        if take <= 0:
            break

        xs.append(x[:take])
        masks.append(mask[:take])
        hists.append(hist[:take])
        hist_masks.append(hist_mask[:take])
        ys.append(y[:take])
        total += take

    if xs:
        return {
            "x": torch.cat(xs, dim=0),
            "mask": torch.cat(masks, dim=0),
            "hist": torch.cat(hists, dim=0),
            "hist_mask": torch.cat(hist_masks, dim=0),
            "y": torch.cat(ys, dim=0),
        }

    return {
        "x": torch.empty((0, 0, 34), dtype=torch.float32),
        "mask": torch.empty((0, 34), dtype=torch.bool),
        "hist": torch.empty((0, DEFAULT_HIST_LEN, 8), dtype=torch.long),
        "hist_mask": torch.empty((0, DEFAULT_HIST_LEN), dtype=torch.bool),
        "y": torch.empty((0,), dtype=torch.long),
    }


def _merge_packed_call(parts, max_samples):
    xs, masks, hists, hist_masks, ys = [], [], [], [], []
    total = 0

    for part in parts:
        if total >= max_samples:
            break
        x = part["x"]
        mask = part["mask"]
        hist = part["hist"]
        hist_mask = part["hist_mask"]
        y = part["y"]

        take = min(x.shape[0], max_samples - total)
        if take <= 0:
            break

        xs.append(x[:take])
        masks.append(mask[:take])
        hists.append(hist[:take])
        hist_masks.append(hist_mask[:take])
        ys.append(y[:take])
        total += take

    if xs:
        return {
            "x": torch.cat(xs, dim=0),
            "mask": torch.cat(masks, dim=0),
            "hist": torch.cat(hists, dim=0),
            "hist_mask": torch.cat(hist_masks, dim=0),
            "y": torch.cat(ys, dim=0),
        }

    return {
        "x": torch.empty((0, 0, 34), dtype=torch.float32),
        "mask": torch.empty((0, NUM_CALL_ACTIONS), dtype=torch.bool),
        "hist": torch.empty((0, DEFAULT_HIST_LEN, 8), dtype=torch.long),
        "hist_mask": torch.empty((0, DEFAULT_HIST_LEN), dtype=torch.bool),
        "y": torch.empty((0,), dtype=torch.long),
    }


def merge_dataset_shards(
    shard_paths,
    out_dir,
    max_discard_samples,
    max_call_samples,
    meta=None,
    cleanup_shards=False,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    discard_parts = []
    call_parts = []

    for shard_path in shard_paths:
        shard = torch.load(shard_path, weights_only=False)
        discard_parts.append(shard["discard"])
        call_parts.append(shard["call"])

    discard_data = _merge_packed_discard(discard_parts, max_discard_samples)
    call_data = _merge_packed_call(call_parts, max_call_samples)

    torch.save(discard_data, out_dir / "discard.pt")
    torch.save(call_data, out_dir / "call.pt")

    if meta is not None:
        torch.save(meta, out_dir / "meta.pt")

    print(
        f"Merged shards -> discard={discard_data['y'].shape[0]}, "
        f"call={call_data['y'].shape[0]}"
    )

    if cleanup_shards:
        for shard_path in shard_paths:
            try:
                Path(shard_path).unlink()
            except FileNotFoundError:
                pass

    return discard_data, call_data


# ============================================================
# Multiprocessing worker: scan one file and save one shard
# ============================================================

def _worker_to_shard(task):
    idx, path, per_file_d, per_file_c, shard_dir, hist_len = task
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"shard_{idx:06d}.pt"

    try:
        d_samples, c_samples = extract_all_from_file(
            path,
            max_d=per_file_d,
            max_c=per_file_c,
            hist_len=hist_len,
        )
        _save_shard(shard_path, d_samples, c_samples, source_path=path)
        return {
            "ok": True,
            "shard_path": str(shard_path),
            "discard": len(d_samples),
            "call": len(c_samples),
        }
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return {
            "ok": False,
            "shard_path": None,
            "discard": 0,
            "call": 0,
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
    num_workers=4,
    shard_dir="./processed_dataset/shards",
    hist_len=DEFAULT_HIST_LEN,
):
    files = find_gz_files(Path(root_dir), years, max_files)
    print(f"Found {len(files)} files, scanning with {num_workers} workers...")

    base_per_d = (
        max(max_discard_samples // max(len(files), 1) * 3, 500)
        if max_discard_samples > 0 else 0
    )
    base_per_c = (
        max(max_call_samples // max(len(files), 1) * 3, 500)
        if max_call_samples > 0 else 0
    )

    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_paths = []
    total_d = 0
    total_c = 0

    def discard_done():
        return (max_discard_samples > 0) and (total_d >= max_discard_samples)

    def call_done():
        return (max_call_samples > 0) and (total_c >= max_call_samples)

    def both_done():
        d_ok = (max_discard_samples <= 0) or (total_d >= max_discard_samples)
        c_ok = (max_call_samples <= 0) or (total_c >= max_call_samples)
        return d_ok and c_ok

    def make_task(file_idx):
        per_d = 0 if discard_done() else base_per_d
        per_c = 0 if call_done() else base_per_c
        return (file_idx, files[file_idx], per_d, per_c, str(shard_dir), hist_len)

    if num_workers <= 1:
        scanned = 0
        for file_idx in range(len(files)):
            if both_done():
                print("Reached requested sample targets. Stopping early.")
                break

            task = make_task(file_idx)
            result = _worker_to_shard(task)
            scanned += 1

            if not result["ok"]:
                continue

            shard_paths.append(result["shard_path"])
            total_d += result["discard"]
            total_c += result["call"]

            if scanned % 100 == 0:
                print(f"Scanned {scanned}/{len(files)} -> d={total_d}, c={total_c}")

    else:
        pool = Pool(num_workers)
        pending = []
        next_file_idx = 0
        scanned = 0

        try:
            # Initial fill
            while next_file_idx < len(files) and len(pending) < num_workers and not both_done():
                task = make_task(next_file_idx)
                pending.append(pool.apply_async(_worker_to_shard, (task,)))
                next_file_idx += 1

            while pending:
                new_pending = []

                for job in pending:
                    result = job.get()
                    scanned += 1

                    if result["ok"]:
                        shard_paths.append(result["shard_path"])
                        total_d += result["discard"]
                        total_c += result["call"]

                    if scanned % 100 == 0:
                        print(f"Scanned {scanned}/{len(files)} -> d={total_d}, c={total_c}")

                    if not both_done() and next_file_idx < len(files):
                        task = make_task(next_file_idx)
                        new_pending.append(pool.apply_async(_worker_to_shard, (task,)))
                        next_file_idx += 1

                pending = new_pending

                if both_done():
                    print("Reached requested sample targets. Stopping worker pool early.")
                    break

        finally:
            pool.terminate()
            pool.join()

    print(f"Shard build complete: discard={total_d}, call={total_c}")
    return shard_paths


def build_and_save_dataset(
    root_dir,
    years,
    max_files,
    max_discard_samples,
    max_call_samples,
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
        years=years,
        max_files=max_files,
        max_discard_samples=max_discard_samples,
        max_call_samples=max_call_samples,
        num_workers=num_workers,
        shard_dir=shard_dir,
        hist_len=hist_len,
    )

    discard, call = merge_dataset_shards(
        shard_paths=shard_paths,
        out_dir=out_dir,
        max_discard_samples=max_discard_samples,
        max_call_samples=max_call_samples,
        meta=meta,
        cleanup_shards=cleanup_shards,
    )

    return discard, call, shard_paths


def build_dataset(root_dir, years, max_files,
                  max_discard_samples, max_call_samples,
                  num_workers=4, hist_len=DEFAULT_HIST_LEN):
    tmp_out = Path("./_tmp_processed_dataset_compat")
    discard, call, _ = build_and_save_dataset(
        root_dir=root_dir,
        years=years,
        max_files=max_files,
        max_discard_samples=max_discard_samples,
        max_call_samples=max_call_samples,
        num_workers=num_workers,
        out_dir=tmp_out,
        cleanup_shards=True,
        hist_len=hist_len,
    )

    d_samples = list(zip(
        discard["x"],
        discard["mask"],
        discard["hist"],
        discard["hist_mask"],
        discard["y"].tolist(),
    ))
    c_samples = list(zip(
        call["x"],
        call["mask"],
        call["hist"],
        call["hist_mask"],
        call["y"].tolist(),
    ))
    return d_samples, c_samples


# ============================================================
# Save/load processed dataset
# ============================================================

def save_processed_dataset(
    out_dir,
    discard_samples,
    call_samples,
    meta=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    discard_data = pack_discard_samples(discard_samples)
    call_data = pack_call_samples(call_samples)

    torch.save(discard_data, out_dir / "discard.pt")
    torch.save(call_data, out_dir / "call.pt")

    if meta is not None:
        torch.save(meta, out_dir / "meta.pt")

    print(f"Saved processed dataset to {out_dir}")


def load_processed_discard_dataset(data_dir):
    data_dir = Path(data_dir)

    discard = torch.load(data_dir / "discard.pt", weights_only=False)

    meta_path = data_dir / "meta.pt"
    meta = torch.load(meta_path, weights_only=False) if meta_path.exists() else None

    return discard, meta

def load_processed_call_dataset(data_dir):
    data_dir = Path(data_dir)

    call = torch.load(data_dir / "call.pt", weights_only=False)

    meta_path = data_dir / "meta.pt"
    meta = torch.load(meta_path, weights_only=False) if meta_path.exists() else None

    return call, meta


# ============================================================
# PyTorch datasets
# ============================================================

class MahjongDiscardDataset(Dataset):
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


class MahjongCallDataset(Dataset):
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
