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
    """
    Check if a 34-element count array forms a valid winning hand.
    Standard form: 4 mentsu (sets of 3) + 1 jantai (pair).
    Also checks for special forms: seven pairs and kokushi.
    """
    total = sum(counts)
    if total != 14:
        return False

    # --- seven pairs ---
    if sum(1 for c in counts if c == 2) == 7:
        return True

    # --- kokushi musou (thirteen orphans) ---
    terminals = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]
    if all(counts[t] >= 1 for t in terminals):
        has_pair = any(counts[t] >= 2 for t in terminals)
        if has_pair and total == 14:
            # check no non-terminal tiles
            non_terminal_count = sum(counts[i] for i in range(34) if i not in terminals)
            if non_terminal_count == 0:
                return True

    # --- standard form: 1 pair + 4 mentsu ---
    return _check_standard_win(counts[:])


def _check_standard_win(counts: List[int]) -> bool:
    """Try each possible pair, then check if remaining tiles form 4 mentsu."""
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
    """Recursively try to remove 'needed' mentsu (koutsu or shuntsu) from counts."""
    if needed == 0:
        return all(c == 0 for c in counts)

    # find first non-zero tile
    for i in range(34):
        if counts[i] > 0:
            break
    else:
        return False

    # try koutsu (triplet)
    if counts[i] >= 3:
        counts[i] -= 3
        if _remove_mentsu(counts, needed - 1):
            counts[i] += 3
            return True
        counts[i] += 3

    # try shuntsu (sequence) — only for number tiles, not honors
    if i < 27:
        suit_pos = i % 9
        if suit_pos <= 6:  # can form i, i+1, i+2
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
    """Public API: check if 14-tile hand is a winning hand."""
    return _is_winning_counts(counts[:])


def is_tenpai(counts: List[int]) -> Tuple[bool, List[int]]:
    """
    Check if a 13-tile hand is tenpai (one tile away from winning).
    Returns (is_tenpai, list_of_waiting_tiles).

    waiting_tiles: list of 34-indices that would complete the hand.
    """
    assert sum(counts) == 13, f"Expected 13 tiles, got {sum(counts)}"

    waiting = []
    for tile in range(34):
        if counts[tile] >= 4:  # can't draw a 5th copy
            continue
        counts[tile] += 1
        if _is_winning_counts(counts):
            waiting.append(tile)
        counts[tile] -= 1

    return len(waiting) > 0, waiting


# ============================================================
# Enhanced game state tracker — 16-channel features
# ============================================================

class ToyRoundState:
    """
    Enhanced state tracker with win/tenpai detection.

    16-channel feature layout:
        0     : own hand counts
        1     : own discards
        2-4   : opponents' discards (shimocha, toimen, kamicha)
        5     : dora indicators (one-hot)
        6-8   : opponents' riichi status (broadcast 0/1)
        9     : own riichi status (broadcast 0/1)
        10    : round wind / bakaze (broadcast, normalized)
        11    : seat wind / jikaze (broadcast, normalized)
        12    : turn number / junme (broadcast, normalized 0-1)
        13-15 : opponents' open melds (tiles exposed via chi/pon/kan)
    """

    NUM_FEATURE_CHANNELS = 16

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
        # track whether each player has called (not menzen)
        self.has_called: List[bool] = [False, False, False, False]

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

    def is_menzen(self, actor: int) -> bool:
        """Check if player is menzen (no open calls: chi/pon/daiminkan)."""
        return not self.has_called[actor]

    def check_tsumo_agari(self, actor: int) -> bool:
        """Check if current hand (14 tiles after tsumo) is a winning hand."""
        counts = self.hand_counts(actor)
        if sum(counts) != 14:
            return False
        return is_winning_hand(counts)

    def check_ron(self, actor: int, pai: str) -> bool:
        """Check if adding pai to hand (13 tiles) makes a winning hand."""
        counts = self.hand_counts(actor)
        if sum(counts) != 13:
            return False
        tile_idx = pai_to_idx(pai)
        counts[tile_idx] += 1
        return is_winning_hand(counts)

    def check_tenpai(self, actor: int) -> Tuple[bool, List[int]]:
        """Check if hand (13 tiles) is tenpai. Returns (bool, waiting_tiles)."""
        counts = self.hand_counts(actor)
        if sum(counts) != 13:
            return False, []
        return is_tenpai(counts)

    def can_riichi(self, actor: int) -> bool:
        """
        Check if player can declare riichi:
        - menzen (no open calls)
        - tenpai (13 tiles, waiting for at least 1 tile)
        - not already in riichi
        - has >= 1000 points
        """
        if self.riichi[actor]:
            return False
        if not self.is_menzen(actor):
            return False
        if self.scores[actor] < 1000:
            return False

        counts = self.hand_counts(actor)
        # after tsumo, hand has 14 tiles; need to check if discarding any tile leaves tenpai
        if sum(counts) == 14:
            return self._has_tenpai_discard(counts)
        elif sum(counts) == 13:
            tenpai, _ = is_tenpai(counts)
            return tenpai
        return False

    def _has_tenpai_discard(self, counts: List[int]) -> bool:
        """Check if there's any discard that leaves hand in tenpai."""
        for i in range(34):
            if counts[i] > 0:
                counts[i] -= 1
                tenpai, _ = is_tenpai(counts)
                counts[i] += 1
                if tenpai:
                    return True
        return False

    def find_riichi_discards(self, actor: int) -> List[int]:
        """Find all tiles that can be discarded to leave hand in tenpai."""
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
        self.discards[actor][pai_to_idx(pai)] += 1
        if self.hands[actor]:
            self._remove_one_tile(self.hands[actor], pai)
        if tsumogiri or self.last_draw[actor] == pai:
            self.last_draw[actor] = None

    def on_reach(self, event: dict):
        actor = event["actor"]
        step = event.get("step", 1)
        if step == 1:
            self.riichi[actor] = 1

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
        # ankan does NOT break menzen
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
            actor = event.get("actor")
            if actor is not None:
                self.riichi[actor] = 1
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
        elif t in {"hora", "ryukyoku", "end_kyoku"}:
            pass

    # ----------------------------------------------------------
    # Feature builder: 16 channels x 34 tiles
    # ----------------------------------------------------------
    def to_feature(self, actor: int) -> torch.Tensor:
        x = torch.zeros(self.NUM_FEATURE_CHANNELS, 34, dtype=torch.float32)

        opponents = [
            (actor + 1) % 4,
            (actor + 2) % 4,
            (actor + 3) % 4,
        ]

        hand_cnts = self.hand_counts(actor)
        for i in range(34):
            x[0, i] = float(hand_cnts[i])

        for i in range(34):
            x[1, i] = float(self.discards[actor][i])

        for off, other in enumerate(opponents):
            for i in range(34):
                x[2 + off, i] = float(self.discards[other][i])

        for idx in self.dora_indicators:
            x[5, idx] = 1.0

        for off, other in enumerate(opponents):
            x[6 + off, :] = float(self.riichi[other])

        x[9, :] = float(self.riichi[actor])

        bakaze_val = (BAKAZE_MAP.get(self.bakaze, 0) + 1) / 4.0
        x[10, :] = bakaze_val

        jikaze = (actor - self.oya) % 4
        jikaze_val = (jikaze + 1) / 4.0
        x[11, :] = jikaze_val

        x[12, :] = min(self.junme / 18.0, 1.0)

        for off, other in enumerate(opponents):
            for i in range(34):
                x[13 + off, i] = float(self.melds[other][i])

        return x

    # ----------------------------------------------------------
    # Bot helpers
    # ----------------------------------------------------------
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