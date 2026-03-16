from typing import List, Optional
import torch


# ============================================================
# Tile utilities
# ============================================================

HONOR_TO_IDX = {
    "E": 27,
    "S": 28,
    "W": 29,
    "N": 30,
    "P": 31,  # white
    "F": 32,  # green
    "C": 33,  # red
}

IDX_TO_HONOR = {v: k for k, v in HONOR_TO_IDX.items()}

BAKAZE_MAP = {"E": 0, "S": 1, "W": 2, "N": 3}


def pai_to_idx(pai: str) -> int:
    """
    mjai tile string -> 34 tile index

    1m..9m => 0..8
    1p..9p => 9..17
    1s..9s => 18..26
    E,S,W,N,P,F,C => 27..33

    Red fives (0m/0p/0s or 5mr/5pr/5sr) mapped to same index as 5m/5p/5s.
    """
    if pai in HONOR_TO_IDX:
        return HONOR_TO_IDX[pai]

    # handle red five notation like "5mr", "5pr", "5sr"
    if len(pai) == 3 and pai[2] == "r":
        pai = pai[:2]

    if len(pai) != 2:
        raise ValueError(f"Invalid pai: {pai}")

    num, suit = pai[0], pai[1]
    if num == "0":  # aka red five
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
    """34 tile index -> non-red mjai tile string."""
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
# Enhanced game state tracker -- 16-channel features
# ============================================================

class ToyRoundState:
    """
    Enhanced state tracker.

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

        # new tracked state
        self.junme: int = 0
        self.melds: List[List[int]] = [[0] * 34 for _ in range(4)]

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

    # ----------------------------------------------------------
    # Event handlers
    # ----------------------------------------------------------
    def on_tsumo(self, event: dict):
        actor = event["actor"]
        pai = event["pai"]

        # track junme: increment when oya draws
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
        """Common logic for chi / pon / daiminkan: move tiles to meld zone."""
        for t in consumed:
            idx = pai_to_idx(t)
            self.melds[actor][idx] += 1
            self._remove_one_tile(self.hands[actor], t)
        if called_pai:
            self.melds[actor][pai_to_idx(called_pai)] += 1

    def on_chi(self, event: dict):
        self._apply_meld(event["actor"], event.get("consumed", []), event.get("pai"))

    def on_pon(self, event: dict):
        self._apply_meld(event["actor"], event.get("consumed", []), event.get("pai"))

    def on_daiminkan(self, event: dict):
        self._apply_meld(event["actor"], event.get("consumed", []), event.get("pai"))

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

        # relative opponent seats
        opponents = [
            (actor + 1) % 4,  # shimocha (下家)
            (actor + 2) % 4,  # toimen   (対面)
            (actor + 3) % 4,  # kamicha  (上家)
        ]

        # plane 0: own hand counts
        hand_cnts = self.hand_counts(actor)
        for i in range(34):
            x[0, i] = float(hand_cnts[i])

        # plane 1: own discards
        for i in range(34):
            x[1, i] = float(self.discards[actor][i])

        # plane 2-4: opponents' discards
        for off, other in enumerate(opponents):
            for i in range(34):
                x[2 + off, i] = float(self.discards[other][i])

        # plane 5: dora indicators
        for idx in self.dora_indicators:
            x[5, idx] = 1.0

        # plane 6-8: opponents' riichi
        for off, other in enumerate(opponents):
            x[6 + off, :] = float(self.riichi[other])

        # plane 9: own riichi
        x[9, :] = float(self.riichi[actor])

        # plane 10: round wind (bakaze), normalized
        bakaze_val = (BAKAZE_MAP.get(self.bakaze, 0) + 1) / 4.0
        x[10, :] = bakaze_val

        # plane 11: seat wind (jikaze), normalized
        jikaze = (actor - self.oya) % 4
        jikaze_val = (jikaze + 1) / 4.0
        x[11, :] = jikaze_val

        # plane 12: turn number, normalized (max ~18 turns)
        x[12, :] = min(self.junme / 18.0, 1.0)

        # plane 13-15: opponents' open melds
        for off, other in enumerate(opponents):
            for i in range(34):
                x[13 + off, i] = float(self.melds[other][i])

        return x

    # ----------------------------------------------------------
    # Bot helper
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
