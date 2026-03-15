"""
Quick smoke test: verify the full pipeline works before real training.
Run: python test_pipeline.py
"""
from pathlib import Path
import torch
from gamestate import ToyRoundState, pai_to_idx
from dataloader import extract_discard_samples, extract_call_samples, MahjongDiscardDataset, MahjongCallDataset
from model import SmallMahjongResNet
from torch.utils.data import DataLoader

SAMPLE_FILE = Path("sampleDataFile.jsonl")


def test_gamestate():
    print("=" * 50)
    print("1. Testing GameState")
    print("=" * 50)

    state = ToyRoundState()

    # simulate a minimal start_kyoku event
    event = {
        "type": "start_kyoku",
        "bakaze": "E",
        "dora_marker": "4s",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "tehais": [
            ["2m", "8m", "1p", "3p", "4p", "7p", "1s", "2s", "5sr", "6s", "S", "P", "C"],
            ["2m", "7m", "8m", "3p", "3p", "4p", "9p", "8s", "8s", "8s", "9s", "W", "F"],
            ["3m", "9m", "9m", "1p", "4p", "6p", "6p", "7p", "8p", "8p", "4s", "7s", "S"],
            ["1m", "2m", "9m", "1p", "2p", "5pr", "9p", "3s", "5s", "9s", "E", "E", "F"],
        ],
    }
    state.apply_event(event)

    feat = state.to_feature(0)
    print(f"  Feature shape: {feat.shape}")
    assert feat.shape == (16, 34), f"Expected (16, 34), got {feat.shape}"

    hand_cnts = state.hand_counts(0)
    assert sum(hand_cnts) == 13, f"Expected 13 tiles in hand, got {sum(hand_cnts)}"

    # test tsumo
    state.apply_event({"type": "tsumo", "actor": 0, "pai": "5p"})
    hand_cnts = state.hand_counts(0)
    assert sum(hand_cnts) == 14, f"Expected 14 tiles after tsumo, got {sum(hand_cnts)}"

    # test dahai
    state.apply_event({"type": "dahai", "actor": 0, "pai": "C", "tsumogiri": False})
    hand_cnts = state.hand_counts(0)
    assert sum(hand_cnts) == 13, f"Expected 13 tiles after dahai, got {sum(hand_cnts)}"

    print("  Hand counts OK")
    print("  Feature channels OK")
    print("  Tsumo/Dahai OK")
    print("  PASSED\n")


def test_dataloader():
    print("=" * 50)
    print("2. Testing Dataloader")
    print("=" * 50)

    if not SAMPLE_FILE.exists():
        print(f"  SKIPPED: {SAMPLE_FILE} not found")
        return 0, 0

    # discard samples
    d_samples = extract_discard_samples(SAMPLE_FILE, 1000)
    print(f"  Discard samples extracted: {len(d_samples)}")
    assert len(d_samples) > 0, "No discard samples extracted!"

    feat, mask, label = d_samples[0]
    print(f"  Discard feature shape: {feat.shape}")
    print(f"  Discard mask shape:    {mask.shape}")
    print(f"  Discard label:         {label} (tile: {label})")
    assert feat.shape == (16, 34)
    assert mask.shape == (34,)
    assert 0 <= label < 34

    # call samples
    c_samples = extract_call_samples(SAMPLE_FILE, 1000)
    print(f"  Call samples extracted:    {len(c_samples)}")
    if len(c_samples) > 0:
        feat_c, label_c = c_samples[0]
        print(f"  Call feature shape: {feat_c.shape}")
        assert feat_c.shape == (17, 34)
        assert label_c in (0, 1, 2)
    else:
        print("  (No call samples in this file - expected, it has no pon/chi events)")

    # test PyTorch dataset and dataloader
    ds = MahjongDiscardDataset(d_samples)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    batch_x, batch_mask, batch_y = next(iter(loader))
    print(f"  Batch shapes: x={batch_x.shape}, mask={batch_mask.shape}, y={batch_y.shape}")
    assert batch_x.shape == (4, 16, 34)

    print("  PASSED\n")
    return len(d_samples), len(c_samples)


def test_model():
    print("=" * 50)
    print("3. Testing Model")
    print("=" * 50)

    model = SmallMahjongResNet(in_channels=16)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # test discard forward
    x_discard = torch.randn(4, 16, 34)
    out_discard = model.forward_discard(x_discard)
    print(f"  Discard input:  {x_discard.shape}")
    print(f"  Discard output: {out_discard.shape}")
    assert out_discard.shape == (4, 34)

    # test call forward
    x_call = torch.randn(4, 17, 34)
    out_call = model.forward_call(x_call)
    print(f"  Call input:     {x_call.shape}")
    print(f"  Call output:    {out_call.shape}")
    assert out_call.shape == (4, 3)

    # test backward
    loss = out_discard.sum() + out_call.sum()
    loss.backward()
    print("  Backward pass OK")

    print("  PASSED\n")


def test_training_step():
    print("=" * 50)
    print("4. Testing Training Step")
    print("=" * 50)

    model = SmallMahjongResNet(in_channels=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    # fake discard batch
    x = torch.randn(8, 16, 34)
    mask = torch.ones(8, 34, dtype=torch.bool)
    y = torch.randint(0, 34, (8,))

    optimizer.zero_grad()
    logits = model.forward_discard(x)
    masked = logits.masked_fill(~mask, -1e9)
    loss_d = criterion(masked, y)
    loss_d.backward()
    optimizer.step()
    print(f"  Discard loss: {loss_d.item():.4f}")

    # fake call batch
    x_c = torch.randn(8, 17, 34)
    y_c = torch.randint(0, 3, (8,))

    optimizer.zero_grad()
    logits_c = model.forward_call(x_c)
    loss_c = criterion(logits_c, y_c)
    loss_c.backward()
    optimizer.step()
    print(f"  Call loss:    {loss_c.item():.4f}")

    print("  PASSED\n")


def test_save_load():
    print("=" * 50)
    print("5. Testing Save / Load")
    print("=" * 50)

    model = SmallMahjongResNet(in_channels=16)
    save_path = "/tmp/test_model.pt"

    # save
    torch.save(model.state_dict(), save_path)
    print(f"  Saved to {save_path}")

    # load
    model2 = SmallMahjongResNet(in_channels=16)
    model2.load_state_dict(torch.load(save_path, weights_only=True))
    model2.eval()

    # verify same output
    x = torch.randn(1, 16, 34)
    model.eval()
    with torch.no_grad():
        out1 = model.forward_discard(x)
        out2 = model2.forward_discard(x)
    assert torch.allclose(out1, out2), "Loaded model gives different output!"
    print("  Load and verify OK")

    print("  PASSED\n")


def test_bot_react():
    print("=" * 50)
    print("6. Testing Bot React")
    print("=" * 50)

    import json

    # save a dummy model first
    model = SmallMahjongResNet(in_channels=16)
    save_path = "/tmp/test_bot_model.pt"
    torch.save(model.state_dict(), save_path)

    from bot import Bot
    bot = Bot(model_path=save_path)

    # start game
    events = [{"type": "start_game", "id": 0, "names": ["bot", "p1", "p2", "p3"]}]
    res = bot.react(json.dumps(events))
    print(f"  start_game -> {res}")

    # start kyoku + tsumo (should trigger discard)
    events = [
        {
            "type": "start_kyoku",
            "bakaze": "E", "dora_marker": "4s", "kyoku": 1,
            "honba": 0, "kyotaku": 0, "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "tehais": [
                ["2m", "8m", "1p", "3p", "4p", "7p", "1s", "2s", "5sr", "6s", "S", "P", "C"],
                ["?"] * 13, ["?"] * 13, ["?"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "5p"},
    ]
    res = bot.react(json.dumps(events))
    result = json.loads(res)
    print(f"  tsumo -> {res}")
    assert result["type"] == "dahai", f"Expected dahai, got {result['type']}"
    assert result["actor"] == 0

    # opponent discard (may trigger pon/chi or pass)
    events = [
        {"type": "dahai", "actor": 0, "pai": result["pai"], "tsumogiri": False},
        {"type": "tsumo", "actor": 1, "pai": "?"},
        {"type": "dahai", "actor": 1, "pai": "8m", "tsumogiri": False},
    ]
    res = bot.react(json.dumps(events))
    result2 = json.loads(res)
    print(f"  opponent dahai -> {res}")
    assert result2["type"] in ("none", "pon", "chi"), f"Unexpected: {result2['type']}"

    # end game
    events = [{"type": "end_game"}]
    res = bot.react(json.dumps(events))
    print(f"  end_game -> {res}")

    print("  PASSED\n")


if __name__ == "__main__":
    print("\nMahjong AI Pipeline Smoke Test\n")

    test_gamestate()
    test_dataloader()
    test_model()
    test_training_step()
    test_save_load()
    test_bot_react()

    print("=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)
    print("\nYou're ready to train. Run: python train.py")