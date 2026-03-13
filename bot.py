import json
import sys
from typing import Optional

import torch
from .model import SmallMahjongResNet
from .gamestate import ToyRoundState
from .logger import logger
import pathlib

# ============================================================
# Bot
# ============================================================

class Bot:
    def __init__(self, model_path: str = "./toy_mahjong_resnet.pt", device: Optional[str] = None):
        self.player_id: Optional[int] = None
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.round_state: Optional[ToyRoundState] = None

    def _load_model(self):
        if self.model is not None:
            return

        model = SmallMahjongResNet()
        path = pathlib.Path(__file__).parent / self.model_path

        ckpt = torch.load(path, map_location=self.device)

        # common checkpoint patterns
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

    @torch.no_grad()
    def _predict_discard_idx(self) -> int:
        x = self.round_state.to_feature(self.player_id).unsqueeze(0).to(self.device)  # [1, 6, 34]
        logits = self.model(x)  # expected [1, 34]
        logits = logits[0]

        # valid mask: only tiles we currently hold
        mask = x[0, 0] > 0  # plane 0 is hand counts
        masked_logits = logits.clone()
        masked_logits[~mask] = -1e30

        idx = int(torch.argmax(masked_logits).item())
        return idx

    def _maybe_act(self, last_event: dict) -> Optional[dict]:
        """
        For now: only act when it is our own tsumo.
        """
        if last_event["type"] != "tsumo":
            return None
        if last_event["actor"] != self.player_id:
            return None

        idx = self._predict_discard_idx()
        pai = self.round_state.choose_discard_tile(self.player_id, idx)

        return {
            "type": "dahai",
            "actor": self.player_id,
            "pai": pai,
            "tsumogiri": (self.round_state.last_self_draw == pai),
        }

    def react(self, events: str) -> str:
        try:
            events = json.loads(events)
        except json.JSONDecodeError as e:
            print(f"Failed to parse events: {events}, {e}")
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
                print("Received event before start_game; ignoring.")
                continue

            self.round_state.apply_event(e)
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
    """
    Typical mjai bot process:
    - read one line of JSON events from stdin
    - print one JSON action to stdout
    """
    model_path = sys.argv[1] if len(sys.argv) >= 2 else "./toy_mahjong_resnet.pt"
    bot = Bot(model_path=model_path)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            res = bot.react(line)
        except Exception as e:
            logger.error(f"Bot crashed while reacting to: {line}")
            res = json.dumps({"type": "none"}, separators=(",", ":"))

        logger.info(res, flush=True)


if __name__ == "__main__":
    main()