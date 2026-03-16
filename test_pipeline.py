"""
Smoke test for the full pipeline including win/tenpai/riichi.
Run: python test_pipeline.py
"""
from pathlib import Path
import torch
from gamestate import ToyRoundState, pai_to_idx, is_winning_hand, is_tenpai
from dataloader import (
    extract_discard_samples, extract_call_samples, extract_riichi_samples,
    MahjongDiscardDataset, MahjongCallDataset, MahjongRiichiDataset,
)
from model import SmallMahjongResNet
from torch.utils.data import DataLoader

SAMPLE_FILE = Path("sampleDataFile.jsonl")


def test_win_detection():
    print("=" * 50)
    print("1. Testing Win Detection")
    print("=" * 50)

    # winning hand: 1m2m3m 4m5m6m 7m8m9m 1p2p3p 5p5p
    counts = [0] * 34
    for i in range(9):  # 1m-9m
        counts[i] = 1
    counts[9] = 1   # 1p
    counts[10] = 1  # 2p
    counts[11] = 1  # 3p
    counts[13] = 2  # 5p pair
    assert is_winning_hand(counts), "Should be a winning hand (standard)"
    print("  Standard win: OK")

    # seven pairs: 1m1m 2m2m 3m3m 4m4m 5m5m 6m6m 7m7m
    counts = [0] * 34
    for i in range(7):
        counts[i] = 2
    assert is_winning_hand(counts), "Should be a winning hand (seven pairs)"
    print("  Seven pairs: OK")

    # not a winning hand: 1m 3m 5m 7m 9m 2p 4p 6p 8p 1s 3s 5s 7s 9s (all isolated)
    counts = [0] * 34
    for i in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]:
        counts[i] = 1
    assert not is_winning_hand(counts), "Should NOT be a winning hand"
    print("  Non-winning: OK")

    print("  PASSED\n")


def test_tenpai():
    print("=" * 50)
    print("2. Testing Tenpai Detection")
    print("=" * 50)

    # tenpai: 1m2m3m 4m5m6m 7m8m9m 1p2p3p 5p — waiting for 5p
    counts = [0] * 34
    for i in range(9):
        counts[i] = 1
    counts[9] = 1   # 1p
    counts[10] = 1  # 2p
    counts[11] = 1  # 3p
    counts[13] = 1  # 5p
    assert sum(counts) == 13

    tenpai, waits = is_tenpai(counts)
    assert tenpai, "Should be tenpai"
    assert 13 in waits, f"Should be waiting for 5p (idx 13), got {waits}"
    print(f"  Tenpai with waits: {waits}")
    print("  PASSED\n")


def test_gamestate_win():
    print("=" * 50)
    print("3. Testing GameState Win/Riichi Methods")
    print("=" * 50)

    state = ToyRoundState()
    event = {
        "type": "start_kyoku",
        "bakaze": "E", "dora_marker": "4s", "kyoku": 1,
        "honba": 0, "kyotaku": 0, "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "tehais": [
            # player 0: tenpai hand, waiting for 5p
            ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5p"],
            ["?"] * 13, ["?"] * 13, ["?"] * 13,
        ],
    }
    state.apply_event(event)

    # should be menzen
    assert state.is_menzen(0), "Player 0 should be menzen"
    print("  Menzen: OK")

    # should be tenpai
    tenpai, waits = state.check_tenpai(0)
    assert tenpai, "Player 0 should be tenpai"
    print(f"  Tenpai waits: {waits}")

    # ron check
    assert state.check_ron(0, "5p"), "Should be able to ron on 5p"
    assert not state.check_ron(0, "9s"), "Should NOT ron on 9s"
    print("  Ron check: OK")

    # tsumo agari
    state.apply_event({"type": "tsumo", "actor": 0, "pai": "5p"})
    assert state.check_tsumo_agari(0), "Should be tsumo agari"
    print("  Tsumo agari: OK")

    print("  PASSED\n")


def test_riichi_detection():
    print("=" * 50)
    print("4. Testing Riichi Detection")
    print("=" * 50)

    state = ToyRoundState()
    event = {
        "type": "start_kyoku",
        "bakaze": "E", "dora_marker": "4s", "kyoku": 1,
        "honba": 0, "kyotaku": 0, "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "tehais": [
            # after tsumo will have 14 tiles, need to discard to stay tenpai
            ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5p"],
            ["?"] * 13, ["?"] * 13, ["?"] * 13,
        ],
    }
    state.apply_event(event)

    # simulate tsumo
    state.apply_event({"type": "tsumo", "actor": 0, "pai": "9s"})

    # should be able to riichi (menzen + can discard to tenpai + enough points)
    assert state.can_riichi(0), "Player 0 should be able to riichi"

    riichi_discards = state.find_riichi_discards(0)
    assert len(riichi_discards) > 0, "Should have valid riichi discards"
    print(f"  Can riichi: True")
    print(f"  Valid riichi discards: {riichi_discards}")

    # player who has called cannot riichi
    state2 = ToyRoundState()
    state2.apply_event(event)
    state2.has_called[0] = True  # simulate having called
    state2.apply_event({"type": "tsumo", "actor": 0, "pai": "9s"})
    assert not state2.can_riichi(0), "Called player should NOT riichi"
    print("  Called player can't riichi: OK")

    print("  PASSED\n")


def test_dataloader():
    print("=" * 50)
    print("5. Testing Dataloader")
    print("=" * 50)

    if not SAMPLE_FILE.exists():
        print(f"  SKIPPED: {SAMPLE_FILE} not found")
        return

    d_samples = extract_discard_samples(SAMPLE_FILE, 1000)
    print(f"  Discard samples: {len(d_samples)}")
    assert len(d_samples) > 0
    assert d_samples[0][0].shape == (16, 34)

    c_samples = extract_call_samples(SAMPLE_FILE, 1000)
    print(f"  Call samples:    {len(c_samples)}")

    r_samples = extract_riichi_samples(SAMPLE_FILE, 1000)
    print(f"  Riichi samples:  {len(r_samples)}")
    if len(r_samples) > 0:
        assert r_samples[0][0].shape == (16, 34)
        assert r_samples[0][1] in (0, 1)

    # test dataloaders
    ds = MahjongDiscardDataset(d_samples)
    loader = DataLoader(ds, batch_size=4)
    bx, bm, by = next(iter(loader))
    assert bx.shape == (4, 16, 34)
    print("  Dataloader: OK")

    print("  PASSED\n")


def test_model():
    print("=" * 50)
    print("6. Testing Model (3 heads)")
    print("=" * 50)

    model = SmallMahjongResNet(in_channels=16)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total:,}")

    # discard
    out_d = model.forward_discard(torch.randn(4, 16, 34))
    assert out_d.shape == (4, 34)
    print(f"  Discard: {out_d.shape} OK")

    # call
    out_c = model.forward_call(torch.randn(4, 17, 34))
    assert out_c.shape == (4, 3)
    print(f"  Call:    {out_c.shape} OK")

    # riichi
    out_r = model.forward_riichi(torch.randn(4, 16, 34))
    assert out_r.shape == (4, 2)
    print(f"  Riichi:  {out_r.shape} OK")

    # backward
    loss = out_d.sum() + out_c.sum() + out_r.sum()
    loss.backward()
    print("  Backward: OK")

    print("  PASSED\n")


def test_bot():
    print("=" * 50)
    print("7. Testing Bot (full behavior)")
    print("=" * 50)

    import json

    model = SmallMahjongResNet(in_channels=16)
    save_path = "/tmp/test_bot_v3.pt"
    torch.save(model.state_dict(), save_path)

    from bot import Bot
    bot = Bot(model_path=save_path)

    # start game
    res = bot.react(json.dumps([{"type": "start_game", "id": 0, "names": ["bot", "p1", "p2", "p3"]}]))
    print(f"  start_game -> {res}")

    # start kyoku with tenpai hand + tsumo winning tile
    events = [
        {
            "type": "start_kyoku",
            "bakaze": "E", "dora_marker": "4s", "kyoku": 1,
            "honba": 0, "kyotaku": 0, "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "tehais": [
                ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5p"],
                ["?"] * 13, ["?"] * 13, ["?"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "5p"},
    ]
    res = bot.react(json.dumps(events))
    result = json.loads(res)
    print(f"  tsumo agari -> {res}")
    assert result["type"] == "hora", f"Expected hora (tsumo win), got {result['type']}"
    assert result["actor"] == 0
    assert result["target"] == 0  # self-draw
    print("  Tsumo agari: OK")

    # test ron: opponent discards winning tile
    bot2 = Bot(model_path=save_path)
    bot2.react(json.dumps([{"type": "start_game", "id": 0, "names": ["bot", "p1", "p2", "p3"]}]))
    events2 = [
        {
            "type": "start_kyoku",
            "bakaze": "E", "dora_marker": "4s", "kyoku": 1,
            "honba": 0, "kyotaku": 0, "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "tehais": [
                ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5p"],
                ["?"] * 13, ["?"] * 13, ["?"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 1, "pai": "?"},
        {"type": "dahai", "actor": 1, "pai": "5p", "tsumogiri": True},
    ]
    res2 = bot2.react(json.dumps(events2))
    result2 = json.loads(res2)
    print(f"  ron -> {res2}")
    assert result2["type"] == "hora", f"Expected hora (ron), got {result2['type']}"
    assert result2["target"] == 1  # ron from player 1
    print("  Ron: OK")

    # end game
    res = bot.react(json.dumps([{"type": "end_game"}]))
    print(f"  end_game -> {res}")

    print("  PASSED\n")


if __name__ == "__main__":
    print("\nMahjong AI Pipeline Smoke Test (v3: win + riichi)\n")

    test_win_detection()
    test_tenpai()
    test_gamestate_win()
    test_riichi_detection()
    test_dataloader()
    test_model()
    test_bot()

    print("=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)
    print("\nReady to train: python train.py")