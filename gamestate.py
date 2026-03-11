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


def pai_to_idx(pai: str) -> int:
    """
    mjai tile string -> 34 tile index

    1m..9m => 0..8
    1p..9p => 9..17
    1s..9s => 18..26
    E,S,W,N,P,F,C => 27..33

    Red fives 0m/0p/0s are mapped to the same index as 5m/5p/5s.
    """
    if pai in HONOR_TO_IDX:
        return HONOR_TO_IDX[pai]

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
    """
    34 tile index -> non-red mjai tile string.
    """
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
# Tiny game state tracker
# ============================================================
class ToyRoundState:
    """
    Minimal state for a discard-only bot.

    This is intentionally simple:
    - tracks your own hand exactly
    - tracks public discards
    - tracks riichi flags
    - tracks dora indicator / round info
    - only produces discard decisions on your own tsumo

    If your training features use a richer state, replace `to_feature()`
    with your exact feature builder.
    """

    def __init__(self, player_id: int):
        self.player_id = player_id
        self.reset()

    def reset(self):
        self.hands: List[List[str]] = [[] for _ in range(4)]  # raw tile strings
        self.discards = [[0] * 34 for _ in range(4)]
        self.riichi = [0, 0, 0, 0]
        self.dora_indicators: List[int] = []
        self.bakaze = "E"
        self.kyoku = 1
        self.honba = 0
        self.kyotaku = 0
        self.oya = 0
        self.scores = [25000] * 4
        self.last_self_draw: Optional[str] = None

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
        self.last_self_draw = None

        self.hands = [[] for _ in range(4)]
        tehais = event["tehais"]
        my_tehais = tehais[self.player_id]
        self.hands[self.player_id] = [p for p in my_tehais if p != "?"]

    def hand_counts(self, actor: int) -> List[int]:
        counts = [0] * 34
        for pai in self.hands[actor]:
            counts[pai_to_idx(pai)] += 1
        return counts

    def on_tsumo(self, event: dict):
        actor = event["actor"]
        pai = event["pai"]

        if actor == self.player_id and pai != "?":
            self.hands[actor].append(pai)
            self.last_self_draw = pai

    def on_dahai(self, event: dict):
        actor = event["actor"]
        pai = event["pai"]
        tsumogiri = event.get("tsumogiri", False)

        self.discards[actor][pai_to_idx(pai)] += 1

        if actor == self.player_id:
            self._remove_one_tile(self.hands[actor], pai)
            if tsumogiri:
                self.last_self_draw = None
            elif self.last_self_draw == pai:
                self.last_self_draw = None

    def on_reach(self, event: dict):
        actor = event["actor"]
        step = event.get("step", 1)
        if step == 1:
            self.riichi[actor] = 1

    def on_dora(self, event: dict):
        self.dora_indicators.append(pai_to_idx(event["dora_marker"]))

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
        elif t == "dora":
            self.on_dora(event)
        elif t in {
            "chi", "pon", "daiminkan", "ankan", "kakan",
            "hora", "ryukyoku", "end_kyoku"
        }:
            # For a discard-only toy bot, we ignore deep hand-structure updates here.
            # If you later train call decisions, extend this section.
            pass

    def to_feature(self, actor: int) -> torch.Tensor:
        """
        Returns tensor shape [10, 34].

        This matches the kind of toy feature layout you described earlier.
        Replace this with your EXACT training feature builder if needed.
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

    def choose_discard_tile(self, idx: int) -> str:
        """
        Convert predicted 34-index into a concrete tile string from hand.

        Prefer tsumogiri if the drawn tile matches the chosen index.
        Otherwise discard any matching tile in hand.
        """
        if self.last_self_draw is not None and pai_to_idx(self.last_self_draw) == idx:
            return self.last_self_draw

        for pai in self.hands[self.player_id]:
            if pai_to_idx(pai) == idx:
                return pai

        # Should not happen if mask is correct, but keep a safe fallback.
        return idx_to_pai(idx)

    @staticmethod
    def _remove_one_tile(hand: List[str], pai: str):
        """
        Remove one concrete tile string if possible.
        For red fives, fall back to removing any tile with same 34-index.
        """
        if pai in hand:
            hand.remove(pai)
            return

        target_idx = pai_to_idx(pai)
        for i, t in enumerate(hand):
            if pai_to_idx(t) == target_idx:
                del hand[i]
                return

        print(f"Tried to remove {pai} but it was not found in hand: {hand}")