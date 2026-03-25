from typing import List, Optional, Tuple
import torch


# ============================================================
# Tile utilities
# ============================================================

HONOR_TO_IDX = {
    "E": 27, "S": 28, "W": 29, "N": 30,
    "P": 31, "F": 32, "C": 33,
}
IDX_TO_HONOR = {v: k for k, v in HONOR_TO_IDX.items()}
BAKAZE_MAP = {"E": 0, "S": 1, "W": 2, "N": 3}

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
    "chi": 1,
    "pon": 2,
    "hora": 3,
    "daiminkan": 4,
    "ankan": 5,
    "kakan": 6,
    "riichi": 7
}
NUM_CALL_KINDS = len(CALL_KIND_TO_IDX)

PAD_ACTOR = 4
PAD_TILE = 34


def pai_to_idx(pai: str) -> int:
    if pai in HONOR_TO_IDX:
        return HONOR_TO_IDX[pai]
    if len(pai) == 3 and pai[2] == "r":
        pai = pai[:2]
    if len(pai) != 2:
        raise ValueError(f"Invalid pai: {pai}")
    num, suit = pai[0], pai[1]
    if num == "0":
        num = "5"
    n = int(num)
    if suit == "m":
        return n - 1
    elif suit == "p":
        return 9 + (n - 1)
    elif suit == "s":
        return 18 + (n - 1)
    else:
        raise ValueError(f"Invalid pai: {pai}")


def idx_to_pai(idx: int) -> str:
    if 0 <= idx <= 8:
        return f"{idx + 1}m"
    if 9 <= idx <= 17:
        return f"{idx - 9 + 1}p"
    if 18 <= idx <= 26:
        return f"{idx - 18 + 1}s"
    if 27 <= idx <= 33:
        return IDX_TO_HONOR[idx]
    raise ValueError(f"Invalid idx: {idx}")


# ============================================================
# Win / tenpai detection
# ============================================================

def _is_winning_counts(counts: List[int]) -> bool:
    total = sum(counts)
    if total != 14:
        return False

    if sum(1 for c in counts if c == 2) == 7:
        return True

    terminals = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]
    if all(counts[t] >= 1 for t in terminals):
        has_pair = any(counts[t] >= 2 for t in terminals)
        if has_pair and total == 14:
            non_terminal_count = sum(counts[i] for i in range(34) if i not in terminals)
            if non_terminal_count == 0:
                return True

    return _check_standard_win(counts[:])


def _check_standard_win(counts: List[int]) -> bool:
    for pair_idx in range(34):
        if counts[pair_idx] < 2:
            continue
        counts[pair_idx] -= 2
        if _remove_mentsu(counts, 4):
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
        if suit_pos <= 6:
            if counts[i] >= 1 and counts[i + 1] >= 1 and counts[i + 2] >= 1:
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


def is_winning_hand(counts: List[int]) -> bool:
    return _is_winning_counts(counts[:])


def is_tenpai(counts: List[int]) -> Tuple[bool, List[int]]:
    assert sum(counts) == 13, f"Expected 13 tiles, got {sum(counts)}"

    waiting = []
    for tile in range(34):
        if counts[tile] >= 4:
            continue
        counts[tile] += 1
        if _is_winning_counts(counts):
            waiting.append(tile)
        counts[tile] -= 1

    return len(waiting) > 0, waiting


# ============================================================
# Enhanced game state tracker -- 31-channel features
# ============================================================

class RoundState:
    NUM_FEATURE_CHANNELS = 31

    def __init__(self):
        self.reset()

    def reset(self):
        self.hands: List[List[str]] = [[] for _ in range(4)]
        self.discards: List[List[int]] = [[0] * 34 for _ in range(4)]
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
        self.melds: List[List[int]] = [[0] * 34 for _ in range(4)]
        self.has_called: List[bool] = [False, False, False, False]

        # Track last discard explicitly
        self.last_discard_tile: Optional[int] = None       # 0..33 or None
        self.last_discard_actor: Optional[int] = None      # 0..3 or None

        # Each entry:
        # (type_id, actor_id, target_id, tile_id, red_flag, tsumogiri_flag, call_kind_id, riichi_flag)
        self.history: List[Tuple[int, int, int, int, int, int, int, int]] = []

    def start_kyoku(self, event: dict):
        self.discards = [[0] * 34 for _ in range(4)]
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
        self.melds = [[0] * 34 for _ in range(4)]
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
    def hand_counts(self, actor: int) -> List[int]:
        counts = [0] * 34
        for pai in self.hands[actor]:
            counts[pai_to_idx(pai)] += 1
        return counts

    def legal_discard_mask(self, actor: int) -> torch.Tensor:
        counts = self.hand_counts(actor)
        return torch.tensor([c > 0 for c in counts], dtype=torch.bool)

    def is_menzen(self, actor: int) -> bool:
        return not self.has_called[actor]

    def check_tsumo_agari(self, actor: int) -> bool:
        counts = self.hand_counts(actor)
        if sum(counts) != 14:
            return False
        return is_winning_hand(counts)

    def check_ron(self, actor: int, pai: str) -> bool:
        counts = self.hand_counts(actor)
        tile_idx = pai_to_idx(pai)

        meld_tile_count = sum(self.melds[actor])
        open_melds = meld_tile_count // 3
        expected_closed_tiles = 13 - 3 * open_melds

        if sum(counts) != expected_closed_tiles:
            return False

        counts[tile_idx] += 1
        total_after_ron = sum(counts) + meld_tile_count
        if total_after_ron != 14:
            return False

        return is_winning_hand(counts)

    def check_tenpai(self, actor: int) -> Tuple[bool, List[int]]:
        counts = self.hand_counts(actor)
        meld_tile_count = sum(self.melds[actor])
        open_melds = meld_tile_count // 3
        expected_closed_tiles = 13 - 3 * open_melds

        if sum(counts) != expected_closed_tiles:
            return False, []

        if open_melds > 0:
            waiting = []
            for tile in range(34):
                if counts[tile] >= 4:
                    continue
                counts[tile] += 1
                total_after_draw = sum(counts) + meld_tile_count
                if total_after_draw == 14 and is_winning_hand(counts):
                    waiting.append(tile)
                counts[tile] -= 1
            return len(waiting) > 0, waiting

        return is_tenpai(counts)

    def can_riichi(self, actor: int) -> bool:
        if self.riichi[actor]:
            return False
        if not self.is_menzen(actor):
            return False
        if self.scores[actor] < 1000:
            return False

        counts = self.hand_counts(actor)
        if sum(counts) == 14:
            return self._has_tenpai_discard(counts)
        elif sum(counts) == 13:
            tenpai, _ = is_tenpai(counts)
            return tenpai
        return False

    def _has_tenpai_discard(self, counts: List[int]) -> bool:
        for i in range(34):
            if counts[i] > 0:
                counts[i] -= 1
                tenpai, _ = is_tenpai(counts)
                counts[i] += 1
                if tenpai:
                    return True
        return False

    def find_riichi_discards(self, actor: int) -> List[int]:
        counts = self.hand_counts(actor)
        if sum(counts) != 14:
            return []

        valid = []
        for i in range(34):
            if counts[i] > 0:
                counts[i] -= 1
                tenpai, _ = is_tenpai(counts)
                counts[i] += 1
                if tenpai:
                    valid.append(i)
        return valid

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
    def _is_red_pai(pai: Optional[str]) -> int:
        return int(isinstance(pai, str) and len(pai) == 3 and pai[2] == "r")

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

    def _infer_call_kind(self, event_type: str) -> int:
        return CALL_KIND_TO_IDX.get(event_type, CALL_KIND_TO_IDX["none"])

    def _encode_history_event(self, event: dict) -> Tuple[int, int, int, int, int, int, int, int]:
        event_type = event["type"]
        actor = event.get("actor", None)
        pai = event.get("pai", None)

        type_id = EVENT_TYPE_TO_IDX.get(event_type, 0)
        actor_id = self._safe_actor(actor)
        target_id = self._infer_target(event)
        tile_id = self._safe_tile_from_pai(pai)
        red_flag = self._is_red_pai(pai)
        tsumogiri_flag = int(event.get("tsumogiri", False))
        call_kind_id = self._infer_call_kind(event_type)

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
        """
        Return padded history for the current decision point.

        Each history event is stored as 8 ints:
            [type_id, rel_actor, rel_target, tile_id, red_flag,
            tsumogiri_flag, call_kind_id, riichi_flag]
        """
        assert 0 <= observer < 4, f"Invalid observer: {observer}"

        hist = self.history[-max_len:]
        pad_len = max_len - len(hist)

        hist_events = torch.zeros(max_len, 8, dtype=torch.long)
        hist_pad_mask = torch.ones(max_len, dtype=torch.bool)

        start = pad_len
        for i, ev in enumerate(hist):
            type_id, actor_id, target_id, tile_id, red_flag, tsumogiri_flag, call_kind_id, riichi_flag = ev

            # 0=self, 1=shimocha, 2=toimen, 3=kamicha, 4=PAD
            if actor_id < 4:
                rel_actor = (actor_id - observer) % 4
            else:
                rel_actor = actor_id

            if type_id == EVENT_TYPE_TO_IDX["tsumo"] and actor_id != observer:
                tile_id = PAD_TILE
                red_flag = 0

            if target_id < 4:
                rel_target = (target_id - observer) % 4
            else:
                rel_target = target_id

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
    def _same_suit(a: int, b: int) -> bool:
        return (0 <= a < 27) and (0 <= b < 27) and (a // 9 == b // 9)

    def _chi_availability_planes(self, actor: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns 3 channels of shape [34]:
          - chi_left_available
          - chi_mid_available
          - chi_right_available

        Semantics:
          For discarded tile t:
            left  means using (t+1, t+2)
            mid   means using (t-1, t+1)
            right means using (t-2, t-1)

        Only one tile position can become 1.0 in these planes, namely the last discarded tile.
        Honors always remain 0.
        """
        left = torch.zeros(34, dtype=torch.float32)
        mid = torch.zeros(34, dtype=torch.float32)
        right = torch.zeros(34, dtype=torch.float32)

        if self.last_discard_tile is None:
            return left, mid, right

        t = self.last_discard_tile

        # honors cannot chi
        if not (0 <= t < 27):
            return left, mid, right

        counts = self.hand_counts(actor)

        # left: t, t+1, t+2
        if t % 9 <= 6 and counts[t + 1] >= 1 and counts[t + 2] >= 1:
            if self._same_suit(t, t + 1) and self._same_suit(t, t + 2):
                left[t] = 1.0

        # mid: t-1, t, t+1
        if 1 <= (t % 9) <= 7 and counts[t - 1] >= 1 and counts[t + 1] >= 1:
            if self._same_suit(t, t - 1) and self._same_suit(t, t + 1):
                mid[t] = 1.0

        # right: t-2, t-1, t
        if t % 9 >= 2 and counts[t - 2] >= 1 and counts[t - 1] >= 1:
            if self._same_suit(t, t - 2) and self._same_suit(t, t - 1):
                right[t] = 1.0

        return left, mid, right

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
    # Feature builder: 31 channels x 34 tiles
    # ----------------------------------------------------------
    def to_feature(self, actor: int) -> torch.Tensor:
        x = torch.zeros(self.NUM_FEATURE_CHANNELS, 34, dtype=torch.float32)

        opponents = [
            (actor + 1) % 4,
            (actor + 2) % 4,
            (actor + 3) % 4,
        ]

        # 0: self hand counts
        hand_cnts = self.hand_counts(actor)
        for i in range(34):
            x[0, i] = float(hand_cnts[i])

        # 1: self discards
        for i in range(34):
            x[1, i] = float(self.discards[actor][i])

        # 2..4: opponent discards
        for off, other in enumerate(opponents):
            for i in range(34):
                x[2 + off, i] = float(self.discards[other][i])

        # 5: dora
        for idx in self.dora_indicators:
            x[5, idx] = 1.0

        # 6..8: opponent riichi
        for off, other in enumerate(opponents):
            x[6 + off, :] = float(self.riichi[other])

        # 9: self riichi
        x[9, :] = float(self.riichi[actor])

        # 10: bakaze
        bakaze_val = (BAKAZE_MAP.get(self.bakaze, 0) + 1) / 4.0
        x[10, :] = bakaze_val

        # 11: jikaze
        jikaze = (actor - self.oya) % 4
        jikaze_val = (jikaze + 1) / 4.0
        x[11, :] = jikaze_val

        # 12: junme
        x[12, :] = min(self.junme / 18.0, 1.0)

        # 13: self melds
        for i in range(34):
            x[13, i] = float(self.melds[actor][i])

        # 14..16: opponent melds
        for off, other in enumerate(opponents):
            for i in range(34):
                x[14 + off, i] = float(self.melds[other][i])

        # 17: kyoku
        x[17, :] = float(max(1, min(self.kyoku, 4))) / 4.0

        # 18: honba
        x[18, :] = min(float(self.honba) / 5.0, 1.0)

        # 19..22: score planes in actor-relative order
        seat_order = [actor, (actor + 1) % 4, (actor + 2) % 4, (actor + 3) % 4]
        for off, pid in enumerate(seat_order):
            x[19 + off, :] = min(max(float(self.scores[pid]) / 50000.0, 0.0), 1.5)

        # 23: last_discard_tile one-hot
        if self.last_discard_tile is not None:
            x[23, self.last_discard_tile] = 1.0

        # 24..27: last_discard_actor in actor-relative one-hot planes
        if self.last_discard_actor is not None:
            rel = (self.last_discard_actor - actor) % 4
            x[24 + rel, :] = 1.0

        # 28..30: chi availability planes
        chi_left, chi_mid, chi_right = self._chi_availability_planes(actor)
        x[28, :] = chi_left
        x[29, :] = chi_mid
        x[30, :] = chi_right

        return x

    def choose_discard_tile(self, actor: int, idx: int) -> str:
        if self.last_draw[actor] is not None and pai_to_idx(self.last_draw[actor]) == idx:
            return self.last_draw[actor]
        for pai in self.hands[actor]:
            if pai_to_idx(pai) == idx:
                return pai
        return idx_to_pai(idx)

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
    
    def legal_call_mask_from_history(self, player_id: int) -> torch.Tensor:
        """
        Return call legality mask for the current decision point, inferred from
        the last event stored in self.history.

        Action order matches CALL_KIND_TO_IDX:
            0: none
            1: chi
            2: pon
            3: hora
            4: daiminkan
            5: ankan
            6: kakan
            7: riichi

        Supported decision points:
        - last event is tsumo by this player      -> self-draw decisions
        - last event is dahai by another player   -> reaction decisions

        In all other cases, only "none" is legal.
        """
        assert 0 <= player_id < 4, f"Invalid player_id: {player_id}"

        mask = torch.zeros(NUM_CALL_KINDS, dtype=torch.bool)
        mask[CALL_KIND_TO_IDX["none"]] = True  # pass / do nothing is always allowed

        if not self.history:
            return mask

        last = self.history[-1]
        type_id, actor_id, target_id, tile_id, red_flag, tsumogiri_flag, call_kind_id, riichi_flag = last

        # ----------------------------------------------------------
        # Case 1: self-draw decision after own tsumo
        # ----------------------------------------------------------
        if type_id == EVENT_TYPE_TO_IDX["tsumo"] and actor_id == player_id:
            counts = self.hand_counts(player_id)

            # hora (tsumo)
            if self.check_tsumo_agari(player_id):
                mask[CALL_KIND_TO_IDX["hora"]] = True

            # riichi
            if self.can_riichi(player_id):
                mask[CALL_KIND_TO_IDX["riichi"]] = True

            # ankan: any tile appears 4 times in closed hand
            if any(c >= 4 for c in counts):
                mask[CALL_KIND_TO_IDX["ankan"]] = True

            # kakan: have 1 tile in hand that can be added to an existing pon
            # Approximation based on meld tile counts:
            # if melds[player_id][t] >= 3 and hand has that tile, allow kakan.
            for t in range(34):
                if counts[t] >= 1 and self.melds[player_id][t] >= 3:
                    mask[CALL_KIND_TO_IDX["kakan"]] = True
                    break

            return mask

        # ----------------------------------------------------------
        # Case 2: reaction to another player's discard
        # ----------------------------------------------------------
        if type_id == EVENT_TYPE_TO_IDX["dahai"] and actor_id != player_id and actor_id < 4 and tile_id < 34:
            counts = self.hand_counts(player_id)
            pai = idx_to_pai(tile_id)

            # ron
            if self.check_ron(player_id, pai):
                mask[CALL_KIND_TO_IDX["hora"]] = True

            # pon
            if counts[tile_id] >= 2:
                mask[CALL_KIND_TO_IDX["pon"]] = True

            # daiminkan
            if counts[tile_id] >= 3:
                mask[CALL_KIND_TO_IDX["daiminkan"]] = True

            # chi: only from kamicha (player immediately before you)
            # actor_id must be (player_id - 1) mod 4
            if actor_id == (player_id - 1) % 4 and 0 <= tile_id < 27:
                suit_pos = tile_id % 9

                can_left = (
                    suit_pos <= 6
                    and counts[tile_id + 1] >= 1
                    and counts[tile_id + 2] >= 1
                )
                can_mid = (
                    1 <= suit_pos <= 7
                    and counts[tile_id - 1] >= 1
                    and counts[tile_id + 1] >= 1
                )
                can_right = (
                    suit_pos >= 2
                    and counts[tile_id - 2] >= 1
                    and counts[tile_id - 1] >= 1
                )

                if can_left or can_mid or can_right:
                    mask[CALL_KIND_TO_IDX["chi"]] = True

            return mask

        return mask
