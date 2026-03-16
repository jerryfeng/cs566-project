import json
import sys
from typing import Optional

import torch
import pathlib

try:
    from .model import SmallMahjongResNet
    from .gamestate import ToyRoundState, pai_to_idx, idx_to_pai
except ImportError:
    from model import SmallMahjongResNet
    from gamestate import ToyRoundState, pai_to_idx, idx_to_pai


# ============================================================
# Bot with discard + call (pon/chi) support
# ============================================================

class Bot:
    def __init__(self, model_path: str = "./best.pt", device: Optional[str] = None):
        self.player_id: Optional[int] = None
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.round_state: Optional[ToyRoundState] = None

    def _load_model(self):
        if self.model is not None:
            return

        model = SmallMahjongResNet(in_channels=16)
        path = pathlib.Path(self.model_path)

        if not path.exists():
            path = pathlib.Path(__file__).parent / self.model_path
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        ckpt = torch.load(path, map_location=self.device, weights_only=True)

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model

    def _unload_model(self):
        self.model = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

    # ----------------------------------------------------------
    # Discard decision
    # ----------------------------------------------------------
    @torch.no_grad()
    def _predict_discard_idx(self) -> int:
        x = self.round_state.to_feature(self.player_id)   # [16, 34]
        x = x.unsqueeze(0).to(self.device)                 # [1, 16, 34]
        logits = self.model.forward_discard(x)[0]           # [34]

        mask = x[0, 0] > 0
        masked = logits.clone()
        masked[~mask] = -1e30

        return int(torch.argmax(masked).item())

    # ----------------------------------------------------------
    # Call decision (pon / chi)
    # ----------------------------------------------------------
    @torch.no_grad()
    def _predict_call(self, called_tile_idx: int) -> int:
        """
        Returns: 0=pass, 1=pon, 2=chi
        """
        feat = self.round_state.to_feature(self.player_id)  # [16, 34]
        called_plane = torch.zeros(1, 34, dtype=torch.float32)
        called_plane[0, called_tile_idx] = 1.0
        x = torch.cat([feat, called_plane], dim=0)           # [17, 34]
        x = x.unsqueeze(0).to(self.device)                   # [1, 17, 34]

        logits = self.model.forward_call(x)[0]                # [3]
        return int(torch.argmax(logits).item())

    def _can_pon(self, tile_idx: int) -> bool:
        hand_cnts = self.round_state.hand_counts(self.player_id)
        return hand_cnts[tile_idx] >= 2

    def _can_chi(self, tile_idx: int, discarder: int) -> bool:
        # only kamicha (previous seat) can chi
        if (discarder + 1) % 4 != self.player_id:
            return False
        if tile_idx >= 27:
            return False

        hand_cnts = self.round_state.hand_counts(self.player_id)
        suit_start = (tile_idx // 9) * 9
        pos = tile_idx - suit_start

        if pos >= 2 and hand_cnts[suit_start + pos - 2] > 0 and hand_cnts[suit_start + pos - 1] > 0:
            return True
        if pos >= 1 and pos <= 7 and hand_cnts[suit_start + pos - 1] > 0 and hand_cnts[suit_start + pos + 1] > 0:
            return True
        if pos <= 6 and hand_cnts[suit_start + pos + 1] > 0 and hand_cnts[suit_start + pos + 2] > 0:
            return True
        return False

    def _find_chi_consumed(self, tile_idx: int):
        """Find two tiles from hand that form a sequence with the called tile."""
        if tile_idx >= 27:
            return None

        hand_cnts = self.round_state.hand_counts(self.player_id)
        suit_start = (tile_idx // 9) * 9
        pos = tile_idx - suit_start

        # try each possible sequence and return first valid one
        sequences = []
        if pos >= 2:
            sequences.append((suit_start + pos - 2, suit_start + pos - 1))
        if pos >= 1 and pos <= 7:
            sequences.append((suit_start + pos - 1, suit_start + pos + 1))
        if pos <= 6:
            sequences.append((suit_start + pos + 1, suit_start + pos + 2))

        for a, b in sequences:
            if hand_cnts[a] > 0 and hand_cnts[b] > 0:
                return [idx_to_pai(a), idx_to_pai(b)]
        return None

    def _find_pon_consumed(self, tile_idx: int):
        """Find two tiles from hand matching the called tile."""
        target = idx_to_pai(tile_idx)
        found = []
        for pai in self.round_state.hands[self.player_id]:
            if pai_to_idx(pai) == tile_idx:
                found.append(pai)
                if len(found) == 2:
                    break
        return found if len(found) == 2 else None

    # ----------------------------------------------------------
    # React to events
    # ----------------------------------------------------------
    def _maybe_act(self, last_event: dict) -> Optional[dict]:
        etype = last_event["type"]

        # own tsumo OR own chi/pon: discard decision
        if etype in {"tsumo", "chi", "pon"} and last_event["actor"] == self.player_id:
            idx = self._predict_discard_idx()
            pai = self.round_state.choose_discard_tile(self.player_id, idx)
            return {
                "type": "dahai",
                "actor": self.player_id,
                "pai": pai,
                "tsumogiri": (etype == "tsumo" and self.round_state.last_draw[self.player_id] == pai),
            }

        # --- opponent discard: call decision ---
        if etype == "dahai" and last_event["actor"] != self.player_id:
            discarder = last_event["actor"]
            pai = last_event["pai"]
            tile_idx = pai_to_idx(pai)

            can_p = self._can_pon(tile_idx)
            can_c = self._can_chi(tile_idx, discarder)

            if not can_p and not can_c:
                return None

            decision = self._predict_call(tile_idx)

            # decision: 0=pass, 1=pon, 2=chi
            if decision == 1 and can_p:
                consumed = self._find_pon_consumed(tile_idx)
                if consumed:
                    return {
                        "type": "pon",
                        "actor": self.player_id,
                        "target": discarder,
                        "pai": pai,
                        "consumed": consumed,
                    }

            elif decision == 2 and can_c:
                consumed = self._find_chi_consumed(tile_idx)
                if consumed:
                    return {
                        "type": "chi",
                        "actor": self.player_id,
                        "target": discarder,
                        "pai": pai,
                        "consumed": consumed,
                    }

            # pass or invalid call
            return None

        return None

    def react(self, events: str) -> str:
        try:
            payload = json.loads(events)
            if isinstance(payload, dict):
                events = [payload]
            elif isinstance(payload, list):
                events = payload
            else:
                raise ValueError(f"Unexpected payload type: {type(payload)}")
        except json.JSONDecodeError as e:
            print(f"Failed to parse events: {events}, {e}", file=sys.stderr)
            return json.dumps({"type": "none"}, separators=(",", ":"))

        return_action = None

        for e in events:
            t = e["type"]

            if t == "start_game":
                self.player_id = e["id"]
                self.round_state = ToyRoundState()
                self._load_model()
                return_action = {"type": "none"}
                continue

            if t == "end_game":
                self.round_state = None
                self._unload_model()
                return_action = {"type": "none"}
                continue

            if self.player_id is None or self.round_state is None:
                continue

            if t == "dahai" and e["actor"] != self.player_id:
                maybe = self._maybe_act(e)
                if maybe is not None:
                    return_action = maybe

            self.round_state.apply_event(e)

            if t in {"tsumo", "chi", "pon"} and e.get("actor") == self.player_id:
                maybe = self._maybe_act(e)
                if maybe is not None:
                    return_action = maybe

        if return_action is None:
            return json.dumps({"type": "none"}, separators=(",", ":"))
        return json.dumps(return_action, separators=(",", ":"))


# ============================================================
# stdin/stdout server loop
# ============================================================

def main():
    model_path = sys.argv[1] if len(sys.argv) >= 2 else "checkpoints/best.pt"
    bot = Bot(model_path=model_path)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            res = bot.react(line)
        except Exception as e:
            print(f"Bot error: {e}", file=sys.stderr)
            res = json.dumps({"type": "none"}, separators=(",", ":"))

        print(res, flush=True)


if __name__ == "__main__":
    main()
