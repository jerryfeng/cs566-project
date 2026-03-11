from pathlib import Path
from dataloader import MahjongToyDataset, build_toy_dataset
import kagglehub
from model import SmallMahjongResNet
import torch
import random
from torch.utils.data import DataLoader

DATASET_HANDLE = "shokanekolouis/tenhou-to-mjai"
DATASET_PATH = Path(kagglehub.dataset_download(DATASET_HANDLE))
print("DATASET_PATH: ", DATASET_PATH)

YEARS = [str(y) for y in range(2024, 2027)]   # keep tiny for proof of concept
MAX_FILES = 1000                                # tiny subset
MAX_SAMPLES = 200000                            # tiny dataset cap
BATCH_SIZE = 128
EPOCHS = 15
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Train / eval
# ============================================================
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for x, mask, y in loader:
            x = x.to(DEVICE)
            mask = mask.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            masked_logits = logits.masked_fill(~mask, -1e9)
            loss = criterion(masked_logits, y)

            total_loss += loss.item() * x.size(0)
            pred = masked_logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)

def train():
    samples = build_toy_dataset(DATASET_PATH, YEARS, MAX_FILES, MAX_SAMPLES)

    if len(samples) < 100:
        print("Too few samples collected. Increase MAX_FILES or MAX_SAMPLES.")
        return

    random.shuffle(samples)
    split = int(0.8 * len(samples))
    train_samples = samples[:split]
    val_samples = samples[split:]

    train_ds = MahjongToyDataset(train_samples)
    val_ds = MahjongToyDataset(val_samples)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = SmallMahjongResNet().to(DEVICE)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total params:", total)
    print("Trainable params:", trainable)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")
    print(f"Device:        {DEVICE}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for x, mask, y in train_loader:
            x = x.to(DEVICE)   # [B, 9, 34]
            # a mask indicating current hands, which would be valid for discard
            mask = mask.to(DEVICE) # [B, 34]
            y = y.to(DEVICE)   # [B]

            optimizer.zero_grad()
            logits = model(x)

            # mask out invalid discards that are not currently in hands
            masked_logits = logits.masked_fill(~mask, -1e9)
            loss = criterion(masked_logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            pred = masked_logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    torch.save(model.state_dict(), "toy_mahjong_resnet.pt")
    print("Saved model to toy_mahjong_resnet.pt")


if __name__ == "__main__":
    train()