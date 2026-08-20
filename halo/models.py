from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.float().unsqueeze(-1)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-8)


class SequenceBackbone(nn.Module):
    def __init__(
        self,
        mode: Literal["gru", "lstm", "cnn", "transformer"],
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.1,
        n_heads: int = 4,
        cnn_kernel_size: int = 3,
    ):
        super().__init__()
        if mode not in {"gru", "lstm", "cnn", "transformer"}:
            raise ValueError(f"Unsupported backbone: {mode}")
        self.mode = mode
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = max(1, int(num_layers))
        self.dropout = nn.Dropout(float(dropout))

        if mode in {"gru", "lstm"}:
            rnn_cls = nn.GRU if mode == "gru" else nn.LSTM
            self.rnn = rnn_cls(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=float(dropout) if self.num_layers > 1 else 0.0,
            )
            self.out_dim = hidden_dim * 2
        elif mode == "cnn":
            layers = []
            in_channels = input_dim
            padding = int(cnn_kernel_size) // 2
            for _ in range(self.num_layers):
                layers.append(
                    nn.Sequential(
                        nn.Conv1d(in_channels, hidden_dim, kernel_size=cnn_kernel_size, padding=padding),
                        nn.ReLU(),
                        nn.Dropout(float(dropout)),
                    )
                )
                in_channels = hidden_dim
            self.cnn = nn.Sequential(*layers)
            self.out_dim = hidden_dim
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=float(dropout),
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
            self.out_dim = input_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.forward_layers(x, mask)[-1]

    def forward_layers(self, x: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
        x = self.dropout(x)
        if self.mode in {"gru", "lstm"}:
            lengths = mask.sum(dim=1).clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            if self.mode == "gru":
                _, h_n = self.rnn(packed)
            else:
                _, (h_n, _) = self.rnn(packed)
            h_n = h_n.view(self.num_layers, 2, -1, self.hidden_dim)
            return [torch.cat([h_n[l, 0], h_n[l, 1]], dim=1) for l in range(self.num_layers)]

        if self.mode == "cnn":
            y = (x * mask.unsqueeze(-1).float()).transpose(1, 2)
            feats = []
            for layer in self.cnn:
                y = layer(y)
                feats.append(masked_mean(y.transpose(1, 2), mask))
            return feats

        key_padding_mask = ~mask.bool()
        y = x
        feats = []
        for layer in self.encoder.layers:
            y = layer(y, src_key_padding_mask=key_padding_mask)
            feats.append(masked_mean(y, mask))
        return feats


class FSNetClassifier(nn.Module):
    def __init__(
        self,
        num_states: int,
        num_classes: int,
        d_model: int = 64,
        backbone: str = "gru",
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        max_pkt_len: float = 1500.0,
        transformer_heads: int = 4,
        cnn_kernel_size: int = 3,
    ):
        super().__init__()
        self.num_states = int(num_states)
        self.num_classes = int(num_classes)
        self.state_pad_id = int(num_states)
        self.max_pkt_len = float(max_pkt_len)
        self.state_emb = nn.Embedding(num_states + 1, d_model, padding_idx=self.state_pad_id)
        self.len_mlp = nn.Sequential(nn.Linear(1, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.backbone = SequenceBackbone(
            mode=backbone,
            input_dim=d_model,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            n_heads=transformer_heads,
            cnn_kernel_size=cnn_kernel_size,
        )
        self.fc = nn.Linear(self.backbone.out_dim, num_classes)

    def packet_embeddings(self, state_ids: torch.Tensor, lengths_signed: torch.Tensor) -> torch.Tensor:
        s_emb = self.state_emb(state_ids)
        norm_len = torch.clamp(lengths_signed / self.max_pkt_len, min=-1.0, max=1.0)
        return s_emb + self.len_mlp(norm_len.unsqueeze(-1))

    def extract_layer_features(
        self, state_ids: torch.Tensor, lengths_signed: torch.Tensor, mask: torch.Tensor
    ) -> list[torch.Tensor]:
        x = self.packet_embeddings(state_ids, lengths_signed)
        return self.backbone.forward_layers(x, mask)

    def forward_with_features(self, state_ids: torch.Tensor, lengths_signed: torch.Tensor, mask: torch.Tensor):
        feats = self.extract_layer_features(state_ids, lengths_signed, mask)
        return self.fc(feats[-1]), feats

    def forward(self, state_ids: torch.Tensor, lengths_signed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_features(state_ids, lengths_signed, mask)
        return logits
