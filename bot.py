import json
import sys
from typing import Optional

import torch
import torch.nn.functional as F
import pathlib

try:
    from .model import MahjongResNet
    from .gamestate import RoundState, pai_to_idx, idx_to_pai
except ImportError:
    from model import MahjongResNet
    from gamestate import RoundState, pai_to_idx, idx_to_pai


CALL_CLASS_NAMES = ["pass", "chi", "pon", "hora", "dmk", "ank", "kak", "rii"]

CALL_THRESHOLDS = {
    1: 0.75,  # chi
    2: 0.72,  # pon
    4: 0.96,  # daiminkan
    5: 0.92,  # ankan
    6: 0.94,  # kakan
}


class Bot:
    def __init__(self, device: Optional[str] = None):
        self.player_id: Optional[int] = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.round_state: Optional[RoundState] = None

    # ----------------------------------------------------------
    # Model loading
    # ----------------------------------------------------------
    def _load_model(self):
        if self.model is not None:
            return

        model = MahjongResNet().to(self.device)

        model.discard_model.load_state_dict(torch.load("./best_discard.pt", map_location=self.device))
        model.call_model.load_state_dict(torch.load("./best_call.pt", map_location=self.device))

        model.eval()
        self.model = model

    def _unload_model(self):
        self.model = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

    # ----------------------------------------------------------
    # Tensor helpers
    # ----------------------------------------------------------
    def _get_state_tensors(self):
        x = self.round_state.to_feature(self.player_id).unsqueeze(0).to(self.device)
        hist, hist_mask = self.round_state.get_history(self.player_id)
        hist = hist.unsqueeze(0).to(self.device)
        hist_mask = hist_mask.unsqueeze(0).to(self.device)
        return x, hist, hist_mask

    @staticmethod
    def _masked_prediction(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return logits.masked_fill(~mask, -1e9)

    def _masked_call_prediction(self, logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        masked_logits = logits.clone()
        masked_logits[~legal_mask] = -1e9

        probs = F.softmax(masked_logits, dim=-1)

        for action_idx, threshold in CALL_THRESHOLDS.items():
            p = probs[:, action_idx]
            reject = p < threshold
            if reject.any():
                probs[reject, action_idx] = 0.0

        return probs

    # ----------------------------------------------------------
    # Forward helpers
    # ----------------------------------------------------------
    @torch.no_grad()
    def _forward_discard(self, x, hist, hist_mask):
        if hasattr(self.model, "forward_discard"):
            out = self.model.forward_discard(x, hist, hist_mask)
        elif hasattr(self.model, "discard_model"):
            out = self.model.discard_model(x, hist, hist_mask)
        else:
            raise AttributeError("Model has neither forward_discard nor discard_model.")
        return out[0] if isinstance(out, tuple) else out

    @torch.no_grad()
    def _forward_call(self, x, hist, hist_mask):
        if hasattr(self.model, "forward_call"):
            out = self.model.forward_call(x, hist, hist_mask)
        elif hasattr(self.model, "call_model"):
            out = self.model.call_model(x, hist, hist_mask)
        else:
            raise AttributeError("Model has neither forward_call nor call_model.")
        return out[0] if isinstance(out, tuple) else out

    # ----------------------------------------------------------
    # Prediction helpers
    # ----------------------------------------------------------
    @torch.no_grad()
    def _predict_discard_idx(self) -> int:
        x, hist, hist_mask = self._get_state_tensors()
        logits = self._forward_discard(x, hist, hist_mask)[0]

        hand_mask = self.round_state.legal_discard_mask(self.player_id)
        masked = self._masked_prediction(logits, hand_mask)
        return int(torch.argmax(masked).item())

    @torch.no_grad()
    def _predict_call(self) -> int:
        """
        All call-like decisions come from forward_call:
        pass / chi / pon / hora / dmk / ank / kak / rii
        """
        x, hist, hist_mask = self._get_state_tensors()
        logits = self._forward_call(x, hist, hist_mask)

        legal_mask = self.round_state.legal_call_mask_from_history(self.player_id)
        probs = self._masked_call_prediction(logits, legal_mask)
        return int(torch.argmax(probs, dim=1).item())

    # ----------------------------------------------------------
    # Tile / meld helpers
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
        if 1 <= pos <= 7 and hand_cnts[suit_start + pos - 1] > 0 and hand_cnts[suit_start + pos + 1] > 0:
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
        if 1 <= pos <= 7:
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

    # Optional kan helpers if your RoundState exposes them.
    def _find_daiminkan_consumed(self, tile_idx: int):
        found = []
        for pai in self.round_state.hands[self.player_id]:
            if pai_to_idx(pai) == tile_idx:
                found.append(pai)
                if len(found) == 3:
                    break
        return found if len(found) == 3 else None

    def _find_ankan_consumed(self):
        hand = self.round_state.hands[self.player_id]
        counts = self.round_state.hand_counts(self.player_id)
        for tile_idx in range(34):
            if counts[tile_idx] >= 4:
                found = []
                for pai in hand:
                    if pai_to_idx(pai) == tile_idx:
                        found.append(pai)
                        if len(found) == 4:
                            return found
        return None

    def _find_kakan_pai(self):
        if not hasattr(self.round_state, "melds"):
            return None

        hand = self.round_state.hands[self.player_id]
        counts = self.round_state.hand_counts(self.player_id)
        player_melds = self.round_state.melds[self.player_id]

        for meld in player_melds:
            # Try to infer pon meld structure conservatively.
            meld_pais = meld.get("pais") or meld.get("consumed") or []
            if len(meld_pais) < 3:
                continue

            tile_idx = pai_to_idx(meld_pais[0])
            if any(pai_to_idx(p) != tile_idx for p in meld_pais[:3]):
                continue

            if counts[tile_idx] > 0:
                for pai in hand:
                    if pai_to_idx(pai) == tile_idx:
                        return pai
        return None

    # ----------------------------------------------------------
    # Action construction from call class
    # ----------------------------------------------------------
    def _build_action_from_call_decision(self, decision: int, trigger_event: dict) -> Optional[dict]:
        rs = self.round_state
        etype = trigger_event["type"]

        # 0 = pass
        if decision == 0:
            return None

        # ------------------------------------------------------
        # Opponent discard window
        # ------------------------------------------------------
        if etype == "dahai" and trigger_event["actor"] != self.player_id:
            discarder = trigger_event["actor"]
            pai = trigger_event["pai"]
            tile_idx = pai_to_idx(pai)

            if decision == 3:
                return {
                    "type": "hora",
                    "actor": self.player_id,
                    "target": discarder,
                    "pai": pai,
                }

            if decision == 2:
                consumed = self._find_pon_consumed(tile_idx)
                if consumed is not None:
                    return {
                        "type": "pon",
                        "actor": self.player_id,
                        "target": discarder,
                        "pai": pai,
                        "consumed": consumed,
                    }
                return None

            if decision == 1:
                consumed = self._find_chi_consumed(tile_idx)
                if consumed is not None:
                    return {
                        "type": "chi",
                        "actor": self.player_id,
                        "target": discarder,
                        "pai": pai,
                        "consumed": consumed,
                    }
                return None

            if decision == 4:
                consumed = self._find_daiminkan_consumed(tile_idx)
                if consumed is not None:
                    return {
                        "type": "daiminkan",
                        "actor": self.player_id,
                        "target": discarder,
                        "pai": pai,
                        "consumed": consumed,
                    }
                return None

            return None

        # ------------------------------------------------------
        # Own tsumo window
        # ------------------------------------------------------
        if etype == "tsumo" and trigger_event["actor"] == self.player_id:
            pai = trigger_event["pai"]

            if decision == 3:
                return {
                    "type": "hora",
                    "actor": self.player_id,
                    "target": self.player_id,
                    "pai": pai,
                }

            if decision == 7:
                riichi_discards = rs.find_riichi_discards(self.player_id)
                if not riichi_discards:
                    return None

                idx = self._predict_discard_idx()
                if idx not in riichi_discards:
                    idx = riichi_discards[0]

                discard_pai = rs.choose_discard_tile(self.player_id, idx)
                return {
                    "type": "reach",
                    "actor": self.player_id,
                    "pai": discard_pai,
                }

            if decision == 5:
                consumed = self._find_ankan_consumed()
                if consumed is not None:
                    return {
                        "type": "ankan",
                        "actor": self.player_id,
                        "consumed": consumed,
                    }
                return None

            if decision == 6:
                pai_to_add = self._find_kakan_pai()
                if pai_to_add is not None:
                    return {
                        "type": "kakan",
                        "actor": self.player_id,
                        "pai": pai_to_add,
                    }
                return None

            return None

        return None

    # ----------------------------------------------------------
    # Main decision logic
    # ----------------------------------------------------------
    def _maybe_act(self, last_event: dict) -> Optional[dict]:
        etype = last_event["type"]
        rs = self.round_state

        # ------------------------------------------------------
        # Opponent discard: ONLY use forward_call
        # ------------------------------------------------------
        if etype == "dahai" and last_event["actor"] != self.player_id:
            decision = self._predict_call()
            return self._build_action_from_call_decision(decision, last_event)

        # ------------------------------------------------------
        # Own tsumo: first ask forward_call for hora/riichi/kan/pass
        # then discard if call-head says pass
        # ------------------------------------------------------
        if etype == "tsumo" and last_event["actor"] == self.player_id:
            decision = self._predict_call()
            action = self._build_action_from_call_decision(decision, last_event)
            if action is not None:
                return action

            idx = self._predict_discard_idx()
            pai = rs.choose_discard_tile(self.player_id, idx)
            return {
                "type": "dahai",
                "actor": self.player_id,
                "pai": pai,
                "tsumogiri": (rs.last_draw[self.player_id] == pai),
            }

        # ------------------------------------------------------
        # After our chi / pon, just discard
        # No dedicated riichi/hora forward here.
        # ------------------------------------------------------
        if etype in {"chi", "pon"} and last_event["actor"] == self.player_id:
            idx = self._predict_discard_idx()
            pai = rs.choose_discard_tile(self.player_id, idx)
            return {
                "type": "dahai",
                "actor": self.player_id,
                "pai": pai,
                "tsumogiri": False,
            }

        return None

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------
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
                self.round_state = RoundState()
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

            # Opponent discard: react before applying event.
            if t == "dahai" and e["actor"] != self.player_id:
                maybe = self._maybe_act(e)
                if maybe is not None:
                    return_action = maybe

            self.round_state.apply_event(e)

            # Own tsumo / chi / pon: react after applying event.
            if t in {"tsumo", "chi", "pon"} and e.get("actor") == self.player_id:
                maybe = self._maybe_act(e)
                if maybe is not None:
                    return_action = maybe

        if return_action is None:
            return json.dumps({"type": "none"}, separators=(",", ":"))
        return json.dumps(return_action, separators=(",", ":"))


def main():
    bot = Bot()

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
    