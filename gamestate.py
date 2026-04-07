from typing import List, Optional, Tuple, Dict
import torch


# ============================================================
# Tile utilities
# ============================================================

NUM_BASE_TILES = 34
NUM_TILES = 37

RED_FIVE_MAN = 34
RED_FIVE_PIN = 35
RED_FIVE_SOU = 36

HONOR_TO_IDX = {
    "E": 27, "S": 28, "W": 29, "N": 30,
    "P": 31, "F": 32, "C": 33,
}
IDX_TO_HONOR = {v: k for k, v in HONOR_TO_IDX.items()}
BAKAZE_MAP = {"E": 0, "S": 1, "W": 2, "N": 3}

PAD_ACTOR = 4
PAD_TILE = NUM_TILES

# ============================================================
# History encoding
# ============================================================

EVENT_TYPE_TO_IDX = {
    "start_kyoku": 0,
    "tsumo": 1,
    "dahai": 2,
    "reach": 3,
    "reach_accepted": 4,
    "dora": 5,
    "chi": 6,
    "pon": 7,
    "daiminkan": 8,
    "ankan": 9,
    "kakan": 10,
    "hora": 11,
    "ryukyoku": 12,
    "end_kyoku": 13,
}
NUM_EVENT_TYPES = len(EVENT_TYPE_TO_IDX)

CALL_KIND_TO_IDX = {
    "none": 0,
    "chi_low": 1,
    "chi_mid": 2,
    "chi_high": 3,
    "pon": 4,
    "kan": 5,
    "hora": 6,
}
NUM_CALL_KINDS = len(CALL_KIND_TO_IDX)

TSUMO_ACTION_TO_IDX = {
    "none": 0,
    "dahai": 1,
    "reach": 2,
    "kan": 3,
    "hora": 4,
}
NUM_TSUMO_ACTIONS = len(TSUMO_ACTION_TO_IDX)


def is_red_pai(pai: str) -> bool:
    return isinstance(pai, str) and len(pai) == 3 and pai[2] == "r"


def pai_to_idx(pai: str) -> int:
    if pai in HONOR_TO_IDX:
        return HONOR_TO_IDX[pai]
    if len(pai) != 2 and len(pai) != 3:
        raise ValueError(f"Invalid pai: {pai}")

    num = pai[0]
    suit = pai[1]

    if num == "0":
        if suit == "m":
            return RED_FIVE_MAN
        if suit == "p":
            return RED_FIVE_PIN
        if suit == "s":
            return RED_FIVE_SOU
        raise ValueError(f"Invalid red pai: {pai}")

    if len(pai) == 3 and pai[2] == "r":
        if num != "5":
            raise ValueError(f"Unexpected red suffix on non-5 tile: {pai}")
        if suit == "m":
            return RED_FIVE_MAN
        if suit == "p":
            return RED_FIVE_PIN
        if suit == "s":
            return RED_FIVE_SOU
        raise ValueError(f"Invalid red pai: {pai}")

    n = int(num)
    if suit == "m":
        return n - 1
    if suit == "p":
        return 9 + (n - 1)
    if suit == "s":
        return 18 + (n - 1)
    raise ValueError(f"Invalid pai: {pai}")


def idx_to_pai(idx: int) -> str:
    if idx == RED_FIVE_MAN:
        return "5mr"
    if idx == RED_FIVE_PIN:
        return "5pr"
    if idx == RED_FIVE_SOU:
        return "5sr"
    if 0 <= idx <= 8:
        return f"{idx + 1}m"
    if 9 <= idx <= 17:
        return f"{idx - 9 + 1}p"
    if 18 <= idx <= 26:
        return f"{idx - 18 + 1}s"
    if 27 <= idx <= 33:
        return IDX_TO_HONOR[idx]
    raise ValueError(f"Invalid idx: {idx}")


def tile37_to_base34(idx: int) -> int:
    if idx == RED_FIVE_MAN:
        return 4
    if idx == RED_FIVE_PIN:
        return 13
    if idx == RED_FIVE_SOU:
        return 22
    if 0 <= idx < 34:
        return idx
    raise ValueError(f"Invalid 37-tile idx: {idx}")


def counts37_to_base34(counts37: List[int]) -> List[int]:
    if len(counts37) != NUM_TILES:
        raise ValueError(f"Expected {NUM_TILES} counts, got {len(counts37)}")
    out = counts37[:34]
    out[4] += counts37[RED_FIVE_MAN]
    out[13] += counts37[RED_FIVE_PIN]
    out[22] += counts37[RED_FIVE_SOU]
    return out


def normalize_pai(pai: str) -> str:
    idx = pai_to_idx(pai)
    return idx_to_pai(idx)


# ============================================================
# Win / tenpai detection (base 34 logic)
# ============================================================


def _is_winning_counts(counts: List[int], open_melds: int = 0) -> bool:
    """
    counts contains only the CLOSED tiles currently in hand, represented in base-34.
    open_melds is the number of already-open 3-tile melds outside the hand.

    A complete hand must satisfy:
        closed_tiles + 3 * open_melds == 14

    and the closed portion itself must be decomposable into:
        1 pair + (4 - open_melds) melds

    Special hands (chiitoitsu / kokushi) are only valid for fully closed hands.
    """
    total = sum(counts)
    required_closed_tiles = 14 - 3 * open_melds
    if total != required_closed_tiles:
        return False

    if open_melds == 0:
        if sum(1 for c in counts if c == 2) == 7:
            return True

        terminals = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]
        if all(counts[t] >= 1 for t in terminals):
            has_pair = any(counts[t] >= 2 for t in terminals)
            if has_pair and total == 14:
                non_terminal_count = sum(counts[i] for i in range(34) if i not in terminals)
                if non_terminal_count == 0:
                    return True

    return _check_standard_win(counts[:], 4 - open_melds)


def _check_standard_win(counts: List[int], needed_mentsu: int) -> bool:
    for pair_idx in range(34):
        if counts[pair_idx] < 2:
            continue
        counts[pair_idx] -= 2
        if _remove_mentsu(counts, needed_mentsu):
            counts[pair_idx] += 2
            return True
        counts[pair_idx] += 2
    return False


def _remove_mentsu(counts: List[int], needed: int) -> bool:
    if needed == 0:
        return all(c == 0 for c in counts)

    for i in range(34):
        if counts[i] > 0:
            break
    else:
        return False

    if counts[i] >= 3:
        counts[i] -= 3
        if _remove_mentsu(counts, needed - 1):
            counts[i] += 3
            return True
        counts[i] += 3

    if i < 27:
        suit_pos = i % 9
        if suit_pos <= 6 and counts[i + 1] >= 1 and counts[i + 2] >= 1:
            counts[i] -= 1
            counts[i + 1] -= 1
            counts[i + 2] -= 1
            if _remove_mentsu(counts, needed - 1):
                counts[i] += 1
                counts[i + 1] += 1
                counts[i + 2] += 1
                return True
            counts[i] += 1
            counts[i + 1] += 1
            counts[i + 2] += 1
    return False


def is_winning_hand(counts34: List[int], open_melds: int = 0) -> bool:
    return _is_winning_counts(counts34[:], open_melds=open_melds)


def is_tenpai(counts34: List[int], open_melds: int = 0) -> Tuple[bool, List[int]]:
    required_closed_tiles = 13 - 3 * open_melds
    assert sum(counts34) == required_closed_tiles, (
        f"Expected {required_closed_tiles} closed tiles with {open_melds} open melds, "
        f"got {sum(counts34)}"
    )
    waiting = []
    for tile in range(34):
        if counts34[tile] >= 4:
            continue
        counts34[tile] += 1
        if _is_winning_counts(counts34, open_melds=open_melds):
            waiting.append(tile)
        counts34[tile] -= 1
    return len(waiting) > 0, waiting


# ============================================================
# Enhanced game state tracker -- 31-channel features x 37 tiles
# ============================================================

class RoundState:
    NUM_FEATURE_CHANNELS = 31

    def __init__(self):
        self.reset()

    def reset(self):
        self.hands: List[List[str]] = [[] for _ in range(4)]
        self.discards: List[List[int]] = [[0] * NUM_TILES for _ in range(4)]
        self.riichi: List[int] = [0, 0, 0, 0]
        self.dora_indicators: List[int] = []
        self.bakaze: str = "E"
        self.kyoku: int = 1
        self.honba: int = 0
        self.kyotaku: int = 0
        self.oya: int = 0
        self.scores: List[int] = [25000] * 4
        self.last_draw: List[Optional[str]] = [None] * 4

        self.junme: int = 0
        self.melds: List[List[int]] = [[0] * NUM_TILES for _ in range(4)]
        self.has_called: List[bool] = [False, False, False, False]

        self.last_discard_tile: Optional[int] = None
        self.last_discard_actor: Optional[int] = None

        # (type_id, actor_id, target_id, tile_id, red_flag, tsumogiri_flag, call_kind_id, riichi_flag)
        self.history: List[Tuple[int, int, int, int, int, int, int, int]] = []

    def start_kyoku(self, event: dict):
        self.discards = [[0] * NUM_TILES for _ in range(4)]
        self.riichi = [0, 0, 0, 0]
        self.dora_indicators = [pai_to_idx(event["dora_marker"])]
        self.bakaze = event["bakaze"]
        self.kyoku = event["kyoku"]
        self.honba = event["honba"]
        self.kyotaku = event["kyotaku"]
        self.oya = event["oya"]
        self.scores = event["scores"][:]
        self.last_draw = [None] * 4

        self.junme = 0
        self.melds = [[0] * NUM_TILES for _ in range(4)]
        self.has_called = [False, False, False, False]

        self.last_discard_tile = None
        self.last_discard_actor = None
        self.history = []

        self.hands = [[] for _ in range(4)]
        for pid in range(4):
            self.hands[pid] = [p for p in event["tehais"][pid] if p != "?"]

    # ----------------------------------------------------------
    # Hand utilities
    # ----------------------------------------------------------
    def hand_counts37(self, actor: int) -> List[int]:
        counts = [0] * NUM_TILES
        for pai in self.hands[actor]:
            counts[pai_to_idx(pai)] += 1
        return counts

    def hand_counts_base34(self, actor: int) -> List[int]:
        return counts37_to_base34(self.hand_counts37(actor))

    def legal_discard_mask(self, actor: int) -> torch.Tensor:
        counts = self.hand_counts37(actor)
        return torch.tensor([c > 0 for c in counts], dtype=torch.bool)

    def is_menzen(self, actor: int) -> bool:
        return not self.has_called[actor]

    def _num_open_melds(self, actor: int) -> int:
        meld_tile_count = sum(self.melds_base34(actor))
        return meld_tile_count // 3

    def check_tsumo_agari(self, actor: int) -> bool:
        counts34 = self.hand_counts_base34(actor)
        open_melds = self._num_open_melds(actor)
        required_closed_tiles = 14 - 3 * open_melds
        if sum(counts34) != required_closed_tiles:
            return False
        return is_winning_hand(counts34, open_melds=open_melds)

    def check_ron(self, actor: int, pai: str) -> bool:
        counts34 = self.hand_counts_base34(actor)
        tile_idx34 = tile37_to_base34(pai_to_idx(pai))

        open_melds = self._num_open_melds(actor)
        expected_closed_tiles = 13 - 3 * open_melds

        if sum(counts34) != expected_closed_tiles:
            return False

        counts34[tile_idx34] += 1
        return is_winning_hand(counts34, open_melds=open_melds)

    def melds_base34(self, actor: int) -> List[int]:
        return counts37_to_base34(self.melds[actor])

    def check_tenpai(self, actor: int) -> Tuple[bool, List[int]]:
        counts34 = self.hand_counts_base34(actor)
        open_melds = self._num_open_melds(actor)
        expected_closed_tiles = 13 - 3 * open_melds

        if sum(counts34) != expected_closed_tiles:
            return False, []

        return is_tenpai(counts34, open_melds=open_melds)

    def can_riichi(self, actor: int) -> bool:
        if self.riichi[actor]:
            return False
        if not self.is_menzen(actor):
            return False
        if self.scores[actor] < 1000:
            return False

        counts34 = self.hand_counts_base34(actor)
        if sum(counts34) == 14:
            return self._has_tenpai_discard(counts34)
        if sum(counts34) == 13:
            tenpai, _ = is_tenpai(counts34)
            return tenpai
        return False

    def _has_tenpai_discard(self, counts34: List[int]) -> bool:
        counts37 = self.hand_counts37_from_base34_guess(counts34)
        for idx37 in range(NUM_TILES):
            if counts37[idx37] <= 0:
                continue
            counts37[idx37] -= 1
            if sum(counts37_to_base34(counts37)) == 13:
                tenpai, _ = is_tenpai(counts37_to_base34(counts37))
                if tenpai:
                    counts37[idx37] += 1
                    return True
            counts37[idx37] += 1
        return False

    @staticmethod
    def hand_counts37_from_base34_guess(counts34: List[int]) -> List[int]:
        counts37 = [0] * NUM_TILES
        for i in range(34):
            counts37[i] = counts34[i]
        return counts37

    def find_riichi_discards(self, actor: int) -> List[int]:
        counts37 = self.hand_counts37(actor)
        if not self.is_menzen(actor):
            return []

        valid = []
        for idx37 in range(NUM_TILES):
            if counts37[idx37] <= 0:
                continue
            counts37[idx37] -= 1
            counts34 = counts37_to_base34(counts37)
            if sum(counts34) == 13:
                tenpai, _ = is_tenpai(counts34)
                if tenpai:
                    valid.append(idx37)
            counts37[idx37] += 1
        return valid

    def find_kan_tiles(self, actor: int) -> List[int]:
        counts37 = self.hand_counts37(actor)
        counts34 = counts37_to_base34(counts37)
        meld34 = self.melds_base34(actor)

        result = set()

        # ankan
        for base34 in range(34):
            if counts34[base34] >= 4:
                for idx37 in self._base34_to_possible_37_indices(base34):
                    if counts37[idx37] > 0:
                        result.add(idx37)
                        break

        # kakan
        for base34 in range(34):
            if meld34[base34] >= 3 and counts34[base34] >= 1:
                for idx37 in self._base34_to_possible_37_indices(base34):
                    if counts37[idx37] > 0:
                        result.add(idx37)
                        break

        return sorted(result)

    @staticmethod
    def _base34_to_possible_37_indices(base34: int) -> List[int]:
        if base34 == 4:
            return [RED_FIVE_MAN, 4]
        if base34 == 13:
            return [RED_FIVE_PIN, 13]
        if base34 == 22:
            return [RED_FIVE_SOU, 22]
        return [base34]

    # ----------------------------------------------------------
    # Score / round-state helpers
    # ----------------------------------------------------------
    def _update_scores_from_event(self, event: dict):
        if "scores" in event and event["scores"] is not None:
            self.scores = event["scores"][:]
        elif "deltas" in event and event["deltas"] is not None:
            deltas = event["deltas"]
            if len(deltas) == 4:
                self.scores = [s + d for s, d in zip(self.scores, deltas)]

    def _update_round_counters_from_event(self, event: dict):
        if "honba" in event and event["honba"] is not None:
            self.honba = event["honba"]
        if "kyotaku" in event and event["kyotaku"] is not None:
            self.kyotaku = event["kyotaku"]
        if "kyoku" in event and event["kyoku"] is not None:
            self.kyoku = event["kyoku"]
        if "bakaze" in event and event["bakaze"] is not None:
            self.bakaze = event["bakaze"]
        if "oya" in event and event["oya"] is not None:
            self.oya = event["oya"]

    # ----------------------------------------------------------
    # History helpers
    # ----------------------------------------------------------
    @staticmethod
    def _safe_actor(value) -> int:
        return int(value) if value is not None else PAD_ACTOR

    @staticmethod
    def _safe_tile_from_pai(pai: Optional[str]) -> int:
        if pai is None or pai == "" or pai == "?":
            return PAD_TILE
        return pai_to_idx(pai)

    def _infer_target(self, event: dict) -> int:
        if "target" in event and event["target"] is not None:
            return int(event["target"])
        if "fromWho" in event and event["fromWho"] is not None:
            return int(event["fromWho"])
        return PAD_ACTOR

    def _infer_call_kind(self, event_type: str, event: dict) -> int:
        if event_type == "chi":
            sub = self.classify_chi_from_event(event)
            return CALL_KIND_TO_IDX[sub]
        if event_type == "pon":
            return CALL_KIND_TO_IDX["pon"]
        if event_type in {"daiminkan", "ankan", "kakan"}:
            return CALL_KIND_TO_IDX["kan"]
        if event_type == "hora":
            return CALL_KIND_TO_IDX["hora"]
        return CALL_KIND_TO_IDX["none"]

    def _encode_history_event(self, event: dict) -> Tuple[int, int, int, int, int, int, int, int]:
        event_type = event["type"]
        actor = event.get("actor", None)
        pai = event.get("pai", None)

        type_id = EVENT_TYPE_TO_IDX.get(event_type, 0)
        actor_id = self._safe_actor(actor)
        target_id = self._infer_target(event)
        tile_id = self._safe_tile_from_pai(pai)
        red_flag = int(is_red_pai(pai))
        tsumogiri_flag = int(event.get("tsumogiri", False))
        call_kind_id = self._infer_call_kind(event_type, event)

        if actor is not None and 0 <= actor < 4:
            riichi_flag = int(self.riichi[actor])
        else:
            riichi_flag = 0

        return (
            type_id,
            actor_id,
            target_id,
            tile_id,
            red_flag,
            tsumogiri_flag,
            call_kind_id,
            riichi_flag,
        )

    def get_history(self, observer: int, max_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        assert 0 <= observer < 4, f"Invalid observer: {observer}"

        hist = self.history[-max_len:]
        pad_len = max_len - len(hist)

        hist_events = torch.zeros(max_len, 8, dtype=torch.long)
        hist_pad_mask = torch.ones(max_len, dtype=torch.bool)

        start = pad_len
        for i, ev in enumerate(hist):
            type_id, actor_id, target_id, tile_id, red_flag, tsumogiri_flag, call_kind_id, riichi_flag = ev

            rel_actor = (actor_id - observer) % 4 if actor_id < 4 else actor_id
            rel_target = (target_id - observer) % 4 if target_id < 4 else target_id

            # Hide opponents' drawn tiles in history
            if type_id == EVENT_TYPE_TO_IDX["tsumo"] and actor_id != observer:
                tile_id = PAD_TILE
                red_flag = 0

            hist_events[start + i] = torch.tensor(
                [
                    type_id,
                    rel_actor,
                    rel_target,
                    tile_id,
                    red_flag,
                    tsumogiri_flag,
                    call_kind_id,
                    riichi_flag,
                ],
                dtype=torch.long,
            )
            hist_pad_mask[start + i] = False

        return hist_events, hist_pad_mask

    # ----------------------------------------------------------
    # Chi helper features
    # ----------------------------------------------------------
    @staticmethod
    def _same_suit_base34(a: int, b: int) -> bool:
        return (0 <= a < 27) and (0 <= b < 27) and (a // 9 == b // 9)

    def _chi_availability_planes(self, actor: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        left = torch.zeros(NUM_TILES, dtype=torch.float32)
        mid = torch.zeros(NUM_TILES, dtype=torch.float32)
        right = torch.zeros(NUM_TILES, dtype=torch.float32)

        if self.last_discard_tile is None:
            return left, mid, right

        t37 = self.last_discard_tile
        t = tile37_to_base34(t37)
        if not (0 <= t < 27):
            return left, mid, right

        counts34 = self.hand_counts_base34(actor)

        if t % 9 <= 6 and counts34[t + 1] >= 1 and counts34[t + 2] >= 1:
            if self._same_suit_base34(t, t + 1) and self._same_suit_base34(t, t + 2):
                left[t37] = 1.0

        if 1 <= (t % 9) <= 7 and counts34[t - 1] >= 1 and counts34[t + 1] >= 1:
            if self._same_suit_base34(t, t - 1) and self._same_suit_base34(t, t + 1):
                mid[t37] = 1.0

        if t % 9 >= 2 and counts34[t - 2] >= 1 and counts34[t - 1] >= 1:
            if self._same_suit_base34(t, t - 2) and self._same_suit_base34(t, t - 1):
                right[t37] = 1.0

        return left, mid, right

    # ----------------------------------------------------------
    # Decision masks
    # ----------------------------------------------------------
    def legal_dahai_reaction_mask(self, player_id: int) -> torch.Tensor:
        mask = torch.zeros(NUM_CALL_KINDS, dtype=torch.bool)

        if self.last_discard_actor is None or self.last_discard_tile is None:
            mask[CALL_KIND_TO_IDX["none"]] = True
            return mask

        discarder = self.last_discard_actor
        if player_id == discarder:
            mask[CALL_KIND_TO_IDX["none"]] = True
            return mask

        counts34 = self.hand_counts_base34(player_id)
        tile37 = self.last_discard_tile
        tile34 = tile37_to_base34(tile37)
        pai = idx_to_pai(tile37)

        mask[CALL_KIND_TO_IDX["hora"]] = self.check_ron(player_id, pai)
        mask[CALL_KIND_TO_IDX["pon"]] = counts34[tile34] >= 2
        mask[CALL_KIND_TO_IDX["kan"]] = counts34[tile34] >= 3

        if player_id == (discarder + 1) % 4 and 0 <= tile34 < 27:
            pos = tile34 % 9
            if pos <= 6 and counts34[tile34 + 1] >= 1 and counts34[tile34 + 2] >= 1:
                mask[CALL_KIND_TO_IDX["chi_low"]] = True
            if 1 <= pos <= 7 and counts34[tile34 - 1] >= 1 and counts34[tile34 + 1] >= 1:
                mask[CALL_KIND_TO_IDX["chi_mid"]] = True
            if pos >= 2 and counts34[tile34 - 2] >= 1 and counts34[tile34 - 1] >= 1:
                mask[CALL_KIND_TO_IDX["chi_high"]] = True

        mask[CALL_KIND_TO_IDX["none"]] = True
        return mask

    def legal_tsumo_action_masks(self, actor: int) -> Dict[str, torch.Tensor]:
        action_mask = torch.zeros(NUM_TSUMO_ACTIONS, dtype=torch.bool)
        tile_mask = torch.zeros(NUM_TILES, dtype=torch.bool)

        discard_mask = self.legal_discard_mask(actor)
        riichi_discards = self.find_riichi_discards(actor)
        kan_tiles = self.find_kan_tiles(actor)

        action_mask[TSUMO_ACTION_TO_IDX["none"]] = True
        if discard_mask.any():
            action_mask[TSUMO_ACTION_TO_IDX["dahai"]] = True
            tile_mask |= discard_mask

        if riichi_discards:
            action_mask[TSUMO_ACTION_TO_IDX["reach"]] = True

        if kan_tiles:
            action_mask[TSUMO_ACTION_TO_IDX["kan"]] = True

        if self.check_tsumo_agari(actor):
            action_mask[TSUMO_ACTION_TO_IDX["hora"]] = True

        return {
            "action_mask": action_mask,
            "discard_mask": discard_mask,
            "reach_mask": self._indices_to_mask(riichi_discards, NUM_TILES),
            "kan_mask": self._indices_to_mask(kan_tiles, NUM_TILES),
            "tile_mask": tile_mask,
        }

    @staticmethod
    def _indices_to_mask(indices: List[int], size: int) -> torch.Tensor:
        mask = torch.zeros(size, dtype=torch.bool)
        for i in indices:
            mask[i] = True
        return mask

    # ----------------------------------------------------------
    # Chi classification helper
    # ----------------------------------------------------------
    @staticmethod
    def classify_chi_from_event(event: dict) -> str:
        pai = event.get("pai")
        consumed = event.get("consumed", [])
        if pai is None or len(consumed) != 2:
            raise ValueError(f"Cannot classify chi from event: {event}")

        called = tile37_to_base34(pai_to_idx(pai))
        vals = sorted(tile37_to_base34(pai_to_idx(t)) for t in consumed + [pai])

        if vals != [called, called + 1, called + 2]:
            # middle tile
            if vals == [called - 1, called, called + 1]:
                return "chi_mid"
            if vals == [called - 2, called - 1, called]:
                return "chi_high"
            raise ValueError(f"Unexpected chi pattern for event: {event}")
        return "chi_low"

    # ----------------------------------------------------------
    # Event handlers
    # ----------------------------------------------------------
    def on_tsumo(self, event: dict):
        actor = event["actor"]
        pai = event["pai"]
        if actor == self.oya:
            self.junme += 1
        if pai != "?":
            self.hands[actor].append(pai)
            self.last_draw[actor] = pai

    def on_dahai(self, event: dict):
        actor = event["actor"]
        pai = event["pai"]
        tsumogiri = event.get("tsumogiri", False)

        tile_idx = pai_to_idx(pai)
        self.discards[actor][tile_idx] += 1
        self._remove_one_tile(self.hands[actor], pai)

        self.last_discard_tile = tile_idx
        self.last_discard_actor = actor

        if tsumogiri or self.last_draw[actor] == pai:
            self.last_draw[actor] = None

    def on_reach(self, event: dict):
        actor = event["actor"]
        step = event.get("step", 1)
        if step == 1:
            self.riichi[actor] = 1

    def on_reach_accepted(self, event: dict):
        actor = event.get("actor")
        if actor is not None:
            self.riichi[actor] = 1

        old_scores = self.scores[:]
        self._update_scores_from_event(event)
        self._update_round_counters_from_event(event)

        if old_scores == self.scores and actor is not None and self.scores[actor] >= 1000:
            self.scores[actor] -= 1000
            self.kyotaku += 1

    def on_dora(self, event: dict):
        self.dora_indicators.append(pai_to_idx(event["dora_marker"]))

    def _apply_meld(self, actor: int, consumed: List[str], called_pai: Optional[str] = None):
        for t in consumed:
            idx = pai_to_idx(t)
            self.melds[actor][idx] += 1
            self._remove_one_tile(self.hands[actor], t)
        if called_pai:
            self.melds[actor][pai_to_idx(called_pai)] += 1

    def on_chi(self, event: dict):
        actor = event["actor"]
        self.has_called[actor] = True
        self._apply_meld(actor, event.get("consumed", []), event.get("pai"))

    def on_pon(self, event: dict):
        actor = event["actor"]
        self.has_called[actor] = True
        self._apply_meld(actor, event.get("consumed", []), event.get("pai"))

    def on_daiminkan(self, event: dict):
        actor = event["actor"]
        self.has_called[actor] = True
        self._apply_meld(actor, event.get("consumed", []), event.get("pai"))

    def on_ankan(self, event: dict):
        actor = event["actor"]
        for t in event.get("consumed", []):
            idx = pai_to_idx(t)
            self.melds[actor][idx] += 1
            self._remove_one_tile(self.hands[actor], t)

    def on_kakan(self, event: dict):
        actor = event["actor"]
        pai = event.get("pai", "")
        if pai:
            idx = pai_to_idx(pai)
            self.melds[actor][idx] += 1
            self._remove_one_tile(self.hands[actor], pai)

    def on_hora(self, event: dict):
        self._update_scores_from_event(event)
        self._update_round_counters_from_event(event)

    def on_ryukyoku(self, event: dict):
        self._update_scores_from_event(event)
        self._update_round_counters_from_event(event)

    def on_end_kyoku(self, event: dict):
        self._update_scores_from_event(event)
        self._update_round_counters_from_event(event)

    def apply_event(self, event: dict):
        t = event["type"]
        if t == "start_kyoku":
            self.start_kyoku(event)
        elif t == "tsumo":
            self.on_tsumo(event)
        elif t == "dahai":
            self.on_dahai(event)
        elif t == "reach":
            self.on_reach(event)
        elif t == "reach_accepted":
            self.on_reach_accepted(event)
        elif t == "dora":
            self.on_dora(event)
        elif t == "chi":
            self.on_chi(event)
        elif t == "pon":
            self.on_pon(event)
        elif t == "daiminkan":
            self.on_daiminkan(event)
        elif t == "ankan":
            self.on_ankan(event)
        elif t == "kakan":
            self.on_kakan(event)
        elif t == "hora":
            self.on_hora(event)
        elif t == "ryukyoku":
            self.on_ryukyoku(event)
        elif t == "end_kyoku":
            self.on_end_kyoku(event)

        self.history.append(self._encode_history_event(event))

    # ----------------------------------------------------------
    # Feature builder: 31 channels x 37 tiles
    # ----------------------------------------------------------
    def to_feature(self, actor: int) -> torch.Tensor:
        x = torch.zeros(self.NUM_FEATURE_CHANNELS, NUM_TILES, dtype=torch.float32)

        opponents = [(actor + 1) % 4, (actor + 2) % 4, (actor + 3) % 4]

        hand_cnts = self.hand_counts37(actor)
        for i in range(NUM_TILES):
            x[0, i] = float(hand_cnts[i])

        for i in range(NUM_TILES):
            x[1, i] = float(self.discards[actor][i])

        for off, other in enumerate(opponents):
            for i in range(NUM_TILES):
                x[2 + off, i] = float(self.discards[other][i])

        for idx in self.dora_indicators:
            x[5, idx] = 1.0

        for off, other in enumerate(opponents):
            x[6 + off, :] = float(self.riichi[other])

        x[9, :] = float(self.riichi[actor])

        bakaze_val = (BAKAZE_MAP.get(self.bakaze, 0) + 1) / 4.0
        x[10, :] = bakaze_val

        jikaze = (actor - self.oya) % 4
        x[11, :] = (jikaze + 1) / 4.0

        x[12, :] = min(self.junme / 18.0, 1.0)

        for i in range(NUM_TILES):
            x[13, i] = float(self.melds[actor][i])

        for off, other in enumerate(opponents):
            for i in range(NUM_TILES):
                x[14 + off, i] = float(self.melds[other][i])

        x[17, :] = float(max(1, min(self.kyoku, 4))) / 4.0
        x[18, :] = min(float(self.honba) / 5.0, 1.0)

        seat_order = [actor, (actor + 1) % 4, (actor + 2) % 4, (actor + 3) % 4]
        for off, pid in enumerate(seat_order):
            x[19 + off, :] = min(max(float(self.scores[pid]) / 50000.0, 0.0), 1.5)

        if self.last_discard_tile is not None:
            x[23, self.last_discard_tile] = 1.0

        if self.last_discard_actor is not None:
            rel = (self.last_discard_actor - actor) % 4
            x[24 + rel, :] = 1.0

        chi_low, chi_mid, chi_high = self._chi_availability_planes(actor)
        x[28, :] = chi_low
        x[29, :] = chi_mid
        x[30, :] = chi_high

        return x

    def choose_discard_tile(self, actor: int, idx37: int) -> str:
        if self.last_draw[actor] is not None and pai_to_idx(self.last_draw[actor]) == idx37:
            return self.last_draw[actor]
        for pai in self.hands[actor]:
            if pai_to_idx(pai) == idx37:
                return pai
        return idx_to_pai(idx37)

    @staticmethod
    def _remove_one_tile(hand: List[str], pai: str):
        if pai in hand:
            hand.remove(pai)
            return

        target_idx = pai_to_idx(pai)
        for i, t in enumerate(hand):
            if pai_to_idx(t) == target_idx:
                del hand[i]
                return

        raise ValueError(
            f"Failed to remove tile {pai} from hand {hand}. "
            f"State desynced from log."
        )
