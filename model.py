import torch
import torch.nn as nn

class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        x = torch.relu(x)
        return x


class SmallMahjongResNet(nn.Module):
    def __init__(self, in_channels=10, hidden=128, num_blocks=10, num_classes=34):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock1D(hidden) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x