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
    1D ResNet with two decision heads:

    - discard_head: 34 classes — which tile to discard after tsumo
    - call_head:    3 classes  — pass(0) / pon(1) / chi(2) when opponent discards

    For discard:  input shape [B, 16, 34]
    For call:     input shape [B, 17, 34]  (16 + 1 extra channel for called tile)

    The backbone (blocks) is shared. Each task has its own stem to handle
    different input channel counts, then feeds into the shared blocks.
    """

    def __init__(
        self,
        in_channels=16,
        hidden=128,
        num_blocks=6,
        num_discard_classes=34,
        num_call_classes=3,
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

        # --- call head: -> 3 (pass / pon / chi) ---
        self.call_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout_head),
            nn.Linear(hidden, num_call_classes),
        )

    def forward_discard(self, x):
        """
        x: [B, 16, 34]
        Returns: [B, 34] discard logits
        """
        x = self.discard_stem(x)
        x = self.blocks(x)
        return self.discard_head(x)

    def forward_call(self, x):
        """
        x: [B, 17, 34]  (16 game state channels + 1 called tile channel)
        Returns: [B, 3] call logits — 0=pass, 1=pon, 2=chi
        """
        x = self.call_stem(x)
        x = self.blocks(x)
        return self.call_head(x)

    def forward(self, x):
        """Default forward = discard (backward compatible)."""
        return self.forward_discard(x)