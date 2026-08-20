from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNPadGenerator(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 1,
        max_pkt_len: float = 1500.0,
        only_server: bool = True,
        rnn_type: str = "gru",
    ):
        super().__init__()
        if rnn_type not in {"gru", "lstm"}:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn_type = rnn_type
        self.rnn = rnn_cls(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.max_pkt_len = float(max_pkt_len)
        self.only_server = bool(only_server)

    def forward(self, lengths, times, dirs, can_modify, mask):
        dt = torch.zeros_like(times)
        dt[:, 1:] = torch.clamp(times[:, 1:] - times[:, :-1], min=0.0)
        x = torch.stack(
            [
                torch.clamp(lengths / self.max_pkt_len, max=1.0),
                torch.log1p(dt),
                dirs,
                can_modify,
            ],
            dim=-1,
        )
        x = x * mask.unsqueeze(-1).float()
        h, _ = self.rnn(x)
        alpha = torch.sigmoid(self.fc(h).squeeze(-1))
        server_mask = (dirs > 0.5).float() if self.only_server else torch.ones_like(dirs)
        headroom = torch.clamp(self.max_pkt_len - lengths, min=0.0)
        can_mod_mask = can_modify * mask.float() * server_mask
        return alpha * headroom * can_mod_mask


class UniversalPadTrigger(nn.Module):
    def __init__(
        self,
        max_len: int,
        max_pkt_len: float = 1500.0,
        only_server: bool = True,
        uap_max_bytes: float = 256.0,
        uap_init_bytes: float = 0.0,
        uap_integer: bool = False,
    ):
        super().__init__()
        self.max_len = int(max_len)
        self.max_pkt_len = float(max_pkt_len)
        self.only_server = bool(only_server)
        self.uap_max_bytes = float(uap_max_bytes)
        self.uap_integer = bool(uap_integer)
        y = max(float(uap_init_bytes), 0.0)
        inv = float(np.log(np.exp(y) - 1.0 + 1e-8))
        self.delta_param = nn.Parameter(torch.full((self.max_len,), inv, dtype=torch.float32))

    @staticmethod
    def ste_round(x: torch.Tensor) -> torch.Tensor:
        return (x - x.detach()) + x.detach().round()

    def delta_bytes(self) -> torch.Tensor:
        delta = F.softplus(self.delta_param)
        if self.uap_max_bytes > 0:
            delta = torch.clamp(delta, max=self.uap_max_bytes)
        return delta

    def forward(self, lengths, times, dirs, can_modify, mask):
        batch_size, seq_len = lengths.shape
        delta = self.delta_bytes()[:seq_len].unsqueeze(0).expand(batch_size, seq_len)
        server_mask = (dirs > 0.5).float() if self.only_server else torch.ones_like(dirs)
        headroom = torch.clamp(self.max_pkt_len - lengths, min=0.0)
        can_mod_mask = can_modify * mask.float() * server_mask
        pad = torch.minimum(delta, headroom) * can_mod_mask
        if self.uap_integer:
            pad = self.ste_round(pad)
            pad = torch.minimum(torch.clamp(pad, min=0.0), headroom) * can_mod_mask
        return pad


def parse_insert_lens(raw: str) -> Tuple[float, ...]:
    values = [x.strip() for x in (raw or "").split(",") if x.strip()]
    return tuple(float(x) for x in values) if values else (400.0, 500.0, 600.0)


@dataclass
class BadNetsResult:
    lengths: torch.Tensor
    times: torch.Tensor
    dirs: torch.Tensor
    can_modify: torch.Tensor
    mask: torch.Tensor
    applied: torch.Tensor


def apply_badnets_insert_batch(
    lengths: torch.Tensor,
    times: torch.Tensor,
    dirs: torch.Tensor,
    can_modify: torch.Tensor,
    mask: torch.Tensor,
    apply_mask: torch.Tensor,
    insert_lens: Tuple[float, ...] = (400.0, 500.0, 600.0),
    after_kth_server: int = 2,
    dt_mode: str = "split",
    dt_const: float = 1e-3,
    max_pkt_len: float = 1500.0,
) -> BadNetsResult:
    batch_size, max_t = lengths.shape
    device = lengths.device
    n_insert = len(insert_lens)
    out_lengths = lengths.clone()
    out_times = times.clone()
    out_dirs = dirs.clone()
    out_can = can_modify.clone()
    out_mask = mask.clone()
    applied = torch.zeros(batch_size, dtype=torch.bool, device=device)

    if dt_mode not in {"split", "const"}:
        raise ValueError("dt_mode must be 'split' or 'const'")

    for i in range(batch_size):
        if not bool(apply_mask[i]):
            continue
        valid_len = int(mask[i].bool().sum().item())
        if valid_len <= 0 or valid_len + n_insert > max_t:
            continue
        server_pos = (dirs[i, :valid_len] > 0.5).nonzero(as_tuple=False).flatten()
        if server_pos.numel() < after_kth_server:
            continue
        insert_at = int(server_pos[after_kth_server - 1].item()) + 1

        new_lengths = torch.zeros((max_t,), device=device, dtype=lengths.dtype)
        new_times = torch.zeros((max_t,), device=device, dtype=times.dtype)
        new_dirs = torch.zeros((max_t,), device=device, dtype=dirs.dtype)
        new_can = torch.zeros((max_t,), device=device, dtype=can_modify.dtype)
        new_mask = torch.zeros((max_t,), device=device, dtype=mask.dtype)

        if insert_at > 0:
            new_lengths[:insert_at] = lengths[i, :insert_at]
            new_times[:insert_at] = times[i, :insert_at]
            new_dirs[:insert_at] = dirs[i, :insert_at]
            new_can[:insert_at] = can_modify[i, :insert_at]
            new_mask[:insert_at] = 1

        if insert_at < valid_len:
            dt_next = times[i, insert_at]
        else:
            dt_next = torch.tensor(float(dt_const), device=device, dtype=times.dtype)

        if dt_mode == "split":
            dt_each = dt_next / float(n_insert + 1)
            shifted_first_dt = dt_each
        else:
            dt_each = torch.tensor(float(dt_const), device=device, dtype=times.dtype)
            shifted_first_dt = dt_next

        for j, trigger_len in enumerate(insert_lens):
            pos = insert_at + j
            new_lengths[pos] = min(float(trigger_len), float(max_pkt_len))
            new_times[pos] = dt_each
            new_dirs[pos] = 1.0
            new_can[pos] = 1.0
            new_mask[pos] = 1

        tail_len = valid_len - insert_at
        if tail_len > 0:
            dst0 = insert_at + n_insert
            dst1 = dst0 + tail_len
            new_lengths[dst0:dst1] = lengths[i, insert_at:valid_len]
            new_dirs[dst0:dst1] = dirs[i, insert_at:valid_len]
            new_can[dst0:dst1] = can_modify[i, insert_at:valid_len]
            new_mask[dst0:dst1] = 1
            new_times[dst0] = shifted_first_dt
            if tail_len > 1:
                new_times[dst0 + 1:dst1] = times[i, insert_at + 1:valid_len]

        out_lengths[i] = new_lengths
        out_times[i] = new_times
        out_dirs[i] = new_dirs
        out_can[i] = new_can
        out_mask[i] = new_mask
        applied[i] = True

    return BadNetsResult(out_lengths, out_times, out_dirs, out_can, out_mask, applied)


def attack_mask(labels: torch.Tensor, target_class: int, poison_source_ids=None, poison_rate: float = 1.0):
    if poison_source_ids is None:
        src_mask = torch.ones_like(labels, dtype=torch.bool)
    else:
        src_mask = torch.zeros_like(labels, dtype=torch.bool)
        for class_id in poison_source_ids:
            src_mask |= labels == int(class_id)
    rand_mask = torch.rand(labels.size(0), device=labels.device) < float(poison_rate)
    return rand_mask & src_mask & (labels != int(target_class))
