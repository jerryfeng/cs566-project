import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(p=dropout)

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
    """
    1D ResNet with three decision heads:

    - discard_head: 34 classes — which tile to discard
    - call_head:    3 classes  — pass(0) / pon(1) / chi(2)
    - riichi_head:  2 classes  — no_riichi(0) / riichi(1)

    For discard:  input [B, 16, 34]
    For call:     input [B, 17, 34]  (16 + called tile)
    For riichi:   input [B, 16, 34]  (same as discard)
    """

    def __init__(
        self,
        in_channels=16,
        hidden=128,
        num_blocks=6,
        num_discard_classes=34,
        num_call_classes=3,
        num_riichi_classes=2,
        dropout_block=0.1,
        dropout_head=0.3,
    ):
        super().__init__()
        self.in_channels = in_channels

        # --- discard stem: 16 ch -> hidden ---
        self.discard_stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )

        # --- call stem: 17 ch -> hidden ---
        self.call_stem = nn.Sequential(
            nn.Conv1d(in_channels + 1, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )

        # --- riichi stem: 16 ch -> hidden (shares structure with discard) ---
        self.riichi_stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )

        # --- shared backbone ---
        self.blocks = nn.Sequential(
            *[ResidualBlock1D(hidden, dropout=dropout_block) for _ in range(num_blocks)]
        )

        # --- discard head: -> 34 ---
        self.discard_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout_head),
            nn.Linear(hidden, num_discard_classes),
        )

        # --- call head: -> 3 ---
        self.call_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout_head),
            nn.Linear(hidden, num_call_classes),
        )

        # --- riichi head: -> 2 ---
        self.riichi_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout_head),
            nn.Linear(hidden, num_riichi_classes),
        )

    def forward_discard(self, x):
        """x: [B, 16, 34] -> [B, 34]"""
        x = self.discard_stem(x)
        x = self.blocks(x)
        return self.discard_head(x)

    def forward_call(self, x):
        """x: [B, 17, 34] -> [B, 3]"""
        x = self.call_stem(x)
        x = self.blocks(x)
        return self.call_head(x)

    def forward_riichi(self, x):
        """x: [B, 16, 34] -> [B, 2] (0=no_riichi, 1=riichi)"""
        x = self.riichi_stem(x)
        x = self.blocks(x)
        return self.riichi_head(x)

    def forward(self, x):
        """Default forward = discard (backward compatible)."""
        return self.forward_discard(x)