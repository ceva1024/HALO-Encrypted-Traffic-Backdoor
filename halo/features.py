from __future__ import annotations

import torch

from .config import FeatureConfig


def compute_state_ids(
    times: torch.Tensor,
    dirs: torch.Tensor,
    mask: torch.Tensor,
    config: FeatureConfig,
    device: torch.device | None = None,
) -> torch.Tensor:
    device = device or times.device
    if config.use_dt_bucket:
        dt = torch.zeros_like(times, device=device)
        dt[:, 1:] = torch.clamp(times[:, 1:] - times[:, :-1], min=0.0)
        boundaries = torch.tensor(config.dt_thresholds, device=device, dtype=times.dtype)
        dt_bucket = torch.bucketize(dt, boundaries).clamp(max=config.num_dt_buckets - 1)
    else:
        dt_bucket = torch.zeros_like(times, dtype=torch.long, device=device)

    if config.use_dir:
        dir_id = dirs.long().clamp(min=0, max=1)
    else:
        dir_id = torch.zeros_like(dt_bucket)

    state_ids = dir_id * config.num_dt_buckets + dt_bucket
    return state_ids.masked_fill(~mask.bool(), config.state_pad_id)


def apply_dir_sign(lengths: torch.Tensor, dirs: torch.Tensor, config: FeatureConfig) -> torch.Tensor:
    if not config.use_dir_sign_for_len:
        return lengths
    sign = torch.where(dirs > 0.5, 1.0, -1.0)
    return lengths * sign


def classifier_inputs(batch, config: FeatureConfig, device: torch.device):
    state_ids = compute_state_ids(batch.times, batch.dirs, batch.mask, config, device=device)
    lengths_signed = apply_dir_sign(batch.lengths, batch.dirs, config)
    return state_ids, lengths_signed, batch.mask
