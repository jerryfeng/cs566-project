import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    def __init__(self, channels, dropout=0.1, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.gn1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.gn2(x)

        x = x + residual
        x = torch.relu(x)
        return x


class SmallMahjongResNet(nn.Module):
    """
    1D ResNet with three decision heads:

    - discard_head: 34 logits -- which tile to discard
    - call_head:    3 logits  -- pass(0) / pon(1) / chi(2)
    - riichi_head:  2 logits  -- no_riichi(0) / riichi(1)

    Inputs:
    - discard: [B, 16, 34]
    - call:    [B, 17, 34]   (16 + called tile channel)
    - riichi:  [B, 16, 34]

    Notes:
    - Uses GroupNorm instead of BatchNorm to avoid train/eval mismatch
      from shared running statistics across different tasks.
    - Discard masking should still be applied in the training / eval script,
      not inside this model.
    """

    def __init__(
        self,
        in_channels=16,
        hidden=192,
        num_blocks=8,
        num_discard_classes=34,
        num_call_classes=3,
        num_riichi_classes=2,
        dropout_block=0.1,
        dropout_head=0.3,
        num_groups=8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_discard_classes = num_discard_classes
        self.num_call_classes = num_call_classes
        self.num_riichi_classes = num_riichi_classes

        # Separate stems per task
        self.discard_stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=hidden),
            nn.ReLU(),
        )

        self.call_stem = nn.Sequential(
            nn.Conv1d(in_channels + 1, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=hidden),
            nn.ReLU(),
        )

        self.riichi_stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=hidden),
            nn.ReLU(),
        )

        # Shared backbone
        self.blocks = nn.Sequential(
            *[
                ResidualBlock1D(
                    channels=hidden,
                    dropout=dropout_block,
                    num_groups=num_groups,
                )
                for _ in range(num_blocks)
            ]
        )

        # Discard head: produce one logit per tile position
        self.discard_head = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_head),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

        # Call head: pooled board representation -> 3 logits
        self.call_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout_head),
            nn.Linear(hidden, num_call_classes),
        )

        # Riichi head: pooled board representation -> 2 logits
        self.riichi_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout_head),
            nn.Linear(hidden, num_riichi_classes),
        )

    def forward_discard(self, x):
        """
        x: [B, 16, 34]
        returns: [B, 34] logits

        Mask illegal discards outside the model, for example:
            logits = model.forward_discard(x)
            logits = logits.masked_fill(~mask, -1e9)
        """
        x = self.discard_stem(x)
        x = self.blocks(x)
        x = self.discard_head(x).squeeze(1)  # [B, 34]
        return x

    def forward_call(self, x):
        """
        x: [B, 17, 34]
        returns: [B, 3] logits
        """
        x = self.call_stem(x)
        x = self.blocks(x)
        x = self.call_head(x)
        return x

    def forward_riichi(self, x):
        """
        x: [B, 16, 34]
        returns: [B, 2] logits
        """
        x = self.riichi_stem(x)
        x = self.blocks(x)
        x = self.riichi_head(x)
        return x

    def forward(self, x):
        """
        Default forward = discard branch for backward compatibility.
        """
        return self.forward_discard(x)
    