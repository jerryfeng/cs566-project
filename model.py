
import torch
import torch.nn as nn

from gamestate import NUM_TILES, NUM_CALL_KINDS, NUM_TSUMO_ACTIONS


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


class MahjongHistoryTransformer(nn.Module):
    """
    Encodes history events into:
      - hist_vec: [B, d_model]
      - hist_ch:  [B, out_channels, board_len]
    """

    def __init__(
        self,
        d_model=128,
        nhead=8,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_len=64,
        out_channels=16,
        board_len=NUM_TILES,
        use_cls_token=True,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.out_channels = out_channels
        self.board_len = board_len
        self.use_cls_token = use_cls_token

        self.type_emb = nn.Embedding(16, d_model)
        self.actor_emb = nn.Embedding(5, d_model)
        self.target_emb = nn.Embedding(5, d_model)
        self.tile_emb = nn.Embedding(NUM_TILES + 1, d_model)  # + PAD
        self.red_emb = nn.Embedding(2, d_model)
        self.tsumogiri_emb = nn.Embedding(2, d_model)
        self.call_emb = nn.Embedding(NUM_CALL_KINDS, d_model)
        self.riichi_emb = nn.Embedding(2, d_model)

        extra_pos = 1 if use_cls_token else 0
        self.pos_emb = nn.Embedding(max_len + extra_pos, d_model)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=enc_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_channels),
        )

    def _build_token_embedding(self, hist_events):
        type_id = hist_events[:, :, 0]
        actor_id = hist_events[:, :, 1]
        target_id = hist_events[:, :, 2]
        tile_id = hist_events[:, :, 3]
        red_flag = hist_events[:, :, 4]
        tsumogiri_flag = hist_events[:, :, 5]
        call_kind_id = hist_events[:, :, 6]
        riichi_flag = hist_events[:, :, 7]

        return (
            self.type_emb(type_id)
            + self.actor_emb(actor_id)
            + self.target_emb(target_id)
            + self.tile_emb(tile_id)
            + self.red_emb(red_flag)
            + self.tsumogiri_emb(tsumogiri_flag)
            + self.call_emb(call_kind_id)
            + self.riichi_emb(riichi_flag)
        )

    def forward(self, hist_events, padding_mask=None):
        if hist_events.dim() != 3 or hist_events.size(-1) != 8:
            raise ValueError(f"hist_events must have shape [B, T, 8], got {tuple(hist_events.shape)}")

        B, T, _ = hist_events.shape
        if T > self.max_len:
            raise ValueError(f"Sequence length T={T} exceeds max_len={self.max_len}")

        x = self._build_token_embedding(hist_events)

        if self.use_cls_token:
            cls = self.cls_token.expand(B, 1, -1)
            x = torch.cat([cls, x], dim=1)
            if padding_mask is not None:
                cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=padding_mask.device)
                padding_mask = torch.cat([cls_pad, padding_mask], dim=1)
            seq_len = T + 1
        else:
            seq_len = T

        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_emb(pos_ids)

        x = self.encoder(x, mask=None, src_key_padding_mask=padding_mask)

        if self.use_cls_token:
            hist_vec = x[:, 0]
        else:
            if padding_mask is None:
                hist_vec = x.mean(dim=1)
            else:
                valid = (~padding_mask).float().unsqueeze(-1)
                hist_vec = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

        hist_ch = self.out_proj(hist_vec).unsqueeze(-1).expand(-1, -1, self.board_len)
        return hist_vec, hist_ch


class DahaiReactionModel(nn.Module):
    def __init__(
        self,
        in_channels=31,
        hidden=192,
        num_blocks=8,
        num_classes=NUM_CALL_KINDS,
        hist_d_model=128,
        hist_out_channels=16,
        hist_max_len=64,
        hist_nhead=8,
        hist_num_layers=2,
        hist_ffn=256,
        dropout_block=0.1,
        dropout_head=0.3,
        hist_dropout=0.1,
        num_groups=8,
    ):
        super().__init__()
        self.history_transformer = MahjongHistoryTransformer(
            d_model=hist_d_model,
            nhead=hist_nhead,
            num_layers=hist_num_layers,
            dim_feedforward=hist_ffn,
            dropout=hist_dropout,
            max_len=hist_max_len,
            out_channels=hist_out_channels,
            board_len=NUM_TILES,
            use_cls_token=True,
        )

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=hidden),
            nn.ReLU(),
        )
        self.backbone = nn.Sequential(*[
            ResidualBlock1D(hidden, dropout=dropout_block, num_groups=num_groups)
            for _ in range(num_blocks)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + hist_d_model, hidden),
            nn.GroupNorm(num_groups=num_groups, num_channels=hidden),
            nn.ReLU(),
            nn.Dropout(dropout_head),

            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout_head * 0.5),

            nn.Linear(hidden // 2, num_classes),
        )

        bias = torch.tensor([
            -0.157,   # none
            -3.943,   # chi_low
            -3.820,   # chi_mid
            -3.917,   # chi_high
            -2.459,   # pon
            -7.847,   # kan
            -3.981,   # hora
        ])
        # initialize bias
        final_linear = self.head[-1]
        with torch.no_grad():
            final_linear.bias.copy_(bias)

    def forward(self, x, hist_event, hist_pad_mask=None):
        hist_vec, _ = self.history_transformer(hist_event, hist_pad_mask)
        x = self.stem(x)
        x = self.backbone(x)
        pooled = torch.cat([x.mean(dim=2), x.amax(dim=2), hist_vec], dim=1)
        logits = self.head(pooled)
        return logits, hist_vec


class TsumoDecisionModel(nn.Module):
    def __init__(
        self,
        in_channels=31,
        hidden=192,
        num_blocks=8,
        num_action_classes=NUM_TSUMO_ACTIONS,
        num_tile_classes=NUM_TILES,
        hist_d_model=128,
        hist_out_channels=16,
        hist_max_len=64,
        hist_nhead=8,
        hist_num_layers=2,
        hist_ffn=256,
        dropout_block=0.1,
        dropout_head=0.3,
        hist_dropout=0.1,
        num_groups=8,
    ):
        super().__init__()
        self.history_transformer = MahjongHistoryTransformer(
            d_model=hist_d_model,
            nhead=hist_nhead,
            num_layers=hist_num_layers,
            dim_feedforward=hist_ffn,
            dropout=hist_dropout,
            max_len=hist_max_len,
            out_channels=hist_out_channels,
            board_len=NUM_TILES,
            use_cls_token=True,
        )

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=hidden),
            nn.ReLU(),
        )
        self.backbone = nn.Sequential(*[
            ResidualBlock1D(hidden, dropout=dropout_block, num_groups=num_groups)
            for _ in range(num_blocks)
        ])

        self.action_head = nn.Sequential(
            nn.Linear(hidden * 2 + hist_d_model, hidden),
            nn.GroupNorm(num_groups=num_groups, num_channels=hidden),
            nn.ReLU(),
            nn.Dropout(dropout_head),
            nn.Linear(hidden, num_action_classes),
        )

        self.tile_head = nn.Sequential(
            nn.Conv1d(hidden + hist_d_model, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout_head),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

    def forward(self, x, hist_event, hist_pad_mask=None):
        hist_vec, _ = self.history_transformer(hist_event, hist_pad_mask)
        x = self.stem(x)
        x = self.backbone(x)

        pooled = torch.cat([x.mean(dim=2), x.amax(dim=2), hist_vec], dim=1)
        action_logits = self.action_head(pooled)

        hist_ch = hist_vec.unsqueeze(-1).expand(-1, -1, x.size(-1))
        tile_logits = self.tile_head(torch.cat([x, hist_ch], dim=1)).squeeze(1)
        return action_logits, tile_logits, hist_vec


class MahjongDecisionNet(nn.Module):
    """
    Wrapper with two separate branches:
      - dahai_model: reacts to another player's discard
      - tsumo_model: decides after own tsumo
    """

    def __init__(
        self,
        dahai_in_channels=31,
        tsumo_in_channels=31,
        hidden=192,
        num_blocks=8,
        num_dahai_classes=NUM_CALL_KINDS,
        num_tsumo_action_classes=NUM_TSUMO_ACTIONS,
        num_tile_classes=NUM_TILES,
        hist_d_model=128,
        hist_out_channels=16,
        hist_max_len=64,
        hist_nhead=8,
        hist_num_layers=2,
        hist_ffn=256,
        dropout_block=0.1,
        dropout_head=0.3,
        hist_dropout=0.1,
        num_groups=8,
    ):
        super().__init__()
        self.dahai_model = DahaiReactionModel(
            in_channels=dahai_in_channels,
            hidden=hidden,
            num_blocks=num_blocks,
            num_classes=num_dahai_classes,
            hist_d_model=hist_d_model,
            hist_out_channels=hist_out_channels,
            hist_max_len=hist_max_len,
            hist_nhead=hist_nhead,
            hist_num_layers=hist_num_layers,
            hist_ffn=hist_ffn,
            dropout_block=dropout_block,
            dropout_head=dropout_head,
            hist_dropout=hist_dropout,
            num_groups=num_groups,
        )
        self.tsumo_model = TsumoDecisionModel(
            in_channels=tsumo_in_channels,
            hidden=hidden,
            num_blocks=num_blocks,
            num_action_classes=num_tsumo_action_classes,
            num_tile_classes=num_tile_classes,
            hist_d_model=hist_d_model,
            hist_out_channels=hist_out_channels,
            hist_max_len=hist_max_len,
            hist_nhead=hist_nhead,
            hist_num_layers=hist_num_layers,
            hist_ffn=hist_ffn,
            dropout_block=dropout_block,
            dropout_head=dropout_head,
            hist_dropout=hist_dropout,
            num_groups=num_groups,
        )

    def forward_dahai(self, x, hist_event, hist_pad_mask=None):
        return self.dahai_model(x, hist_event, hist_pad_mask)

    def forward_tsumo(self, x, hist_event, hist_pad_mask=None):
        return self.tsumo_model(x, hist_event, hist_pad_mask)

    def forward(self, x, hist_event, hist_pad_mask=None, task="dahai"):
        if task == "dahai":
            return self.forward_dahai(x, hist_event, hist_pad_mask)
        if task == "tsumo":
            return self.forward_tsumo(x, hist_event, hist_pad_mask)
        raise ValueError(f"Unknown task: {task}")
