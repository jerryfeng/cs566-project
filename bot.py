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
        x = self.round_state.to_feature(self.player_id)
        x = x.unsqueeze(0).to(self.device)
        logits = self.model.forward_discard(x)[0]
        mask = x[0, 0] > 0
        masked = logits.clone()
        masked[~mask] = -1e30
        return int(torch.argmax(masked).item())

    # ----------------------------------------------------------
    # Call decision (pon / chi)
    # ----------------------------------------------------------
    @torch.no_grad()
    def _predict_call(self, called_tile_idx: int) -> int:
        feat = self.round_state.to_feature(self.player_id)
        called_plane = torch.zeros(1, 34, dtype=torch.float32)
        called_plane[0, called_tile_idx] = 1.0
        x = torch.cat([feat, called_plane], dim=0)
        x = x.unsqueeze(0).to(self.device)
        logits = self.model.forward_call(x)[0]
        return int(torch.argmax(logits).item())

    # ----------------------------------------------------------
    # Riichi decision
    # ----------------------------------------------------------
    @torch.no_grad()
    def _predict_riichi(self) -> bool:
        """Returns True if model recommends riichi."""
        x = self.round_state.to_feature(self.player_id)
        x = x.unsqueeze(0).to(self.device)
        logits = self.model.forward_riichi(x)[0]  # [2]
        return int(torch.argmax(logits).item()) == 1

    # ----------------------------------------------------------
    # Helper: can pon/chi
    # ----------------------------------------------------------
    def _can_pon(self, tile_idx: int) -> bool:
        return self.round_state.hand_counts(self.player_id)[tile_idx] >= 2

    def _can_chi(self, tile_idx: int, discarder: int) -> bool:
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
        if tile_idx >= 27:
            return None
        hand_cnts = self.round_state.hand_counts(self.player_id)
        suit_start = (tile_idx // 9) * 9
        pos = tile_idx - suit_start
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
        found = []
        for pai in self.round_state.hands[self.player_id]:
            if pai_to_idx(pai) == tile_idx:
                found.append(pai)
                if len(found) == 2:
                    break
        return found if len(found) == 2 else None

    # ----------------------------------------------------------
    # Main decision logic
    # ----------------------------------------------------------
    def _maybe_act(self, last_event: dict) -> Optional[dict]:
        etype = last_event["type"]
        rs = self.round_state

        # === OWN TSUMO: check win -> riichi -> discard ===
        if etype in {"tsumo", "chi", "pon"} and last_event["actor"] == self.player_id:

            # 1. Tsumo agari (self-draw win) — rule-based, always declare
            if etype == "tsumo" and rs.check_tsumo_agari(self.player_id):
                return {
                    "type": "hora",
                    "actor": self.player_id,
                    "target": self.player_id,
                    "pai": last_event["pai"],
                }

            # 2. Riichi — model-based decision
            if etype == "tsumo" and rs.can_riichi(self.player_id):
                if self._predict_riichi():
                    # find a valid riichi discard (must leave hand in tenpai)
                    riichi_discards = rs.find_riichi_discards(self.player_id)
                    if riichi_discards:
                        # use model to pick best discard among valid riichi discards
                        idx = self._predict_discard_idx()
                        if idx not in riichi_discards:
                            idx = riichi_discards[0]  # fallback to first valid
                        pai = rs.choose_discard_tile(self.player_id, idx)
                        return {
                            "type": "reach",
                            "actor": self.player_id,
                            "pai": pai,
                        }

            # 3. Normal discard
            idx = self._predict_discard_idx()
            pai = rs.choose_discard_tile(self.player_id, idx)
            return {
                "type": "dahai",
                "actor": self.player_id,
                "pai": pai,
                "tsumogiri": (etype == "tsumo" and rs.last_draw[self.player_id] == pai),
            }

        # === OPPONENT DISCARD: check ron -> pon/chi ===
        if etype == "dahai" and last_event["actor"] != self.player_id:
            discarder = last_event["actor"]
            pai = last_event["pai"]
            tile_idx = pai_to_idx(pai)

            # 1. Ron — rule-based, always declare
            if rs.check_ron(self.player_id, pai):
                return {
                    "type": "hora",
                    "actor": self.player_id,
                    "target": discarder,
                    "pai": pai,
                }

            # 2. Pon / Chi — model-based
            can_p = self._can_pon(tile_idx)
            can_c = self._can_chi(tile_idx, discarder)

            if not can_p and not can_c:
                return None

            decision = self._predict_call(tile_idx)

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

            # opponent discard: check ron / call BEFORE applying event
            if t == "dahai" and e["actor"] != self.player_id:
                maybe = self._maybe_act(e)
                if maybe is not None:
                    return_action = maybe

            self.round_state.apply_event(e)

            # own tsumo / chi / pon: check win / riichi / discard AFTER applying event
            if t in {"tsumo", "chi", "pon"} and e.get("actor") == self.player_id:
                maybe = self._maybe_act(e)
                if maybe is not None:
                    return_action = maybe

        if return_action is None:
            return json.dumps({"type": "none"}, separators=(",", ":"))
        return json.dumps(return_action, separators=(",", ":"))


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