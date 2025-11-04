# SAR-W-MixMAE
# Copyright (c) 2025 Ali Caglayan, Nevrez Imamoglu, Toru Kouyama (AIST).
# License: MIT
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------

import torch
from typing import Literal

Mode = Literal["per_channel", "shared_avg"]

def _minmax_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # per-image, per-channel min-max: expects shape (C,H,W) or (B,C,H,W)
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)

def _per_channel(sar_db: torch.Tensor, exp_scale: float) -> torch.Tensor:
    # sar_db: (..., 2, H, W) in dB
    lin = torch.pow(10.0, sar_db / 10.0)
    lin_n = _minmax_norm(lin)                      # per-channel, per-image
    w = torch.exp(exp_scale * (1.0 - lin_n))       # same shape as input
    return w

def _shared_avg(sar_db: torch.Tensor, exp_scale: float) -> torch.Tensor:
    # generalized, vectorized dB→linear for all channels/batches  (no batch here)
    lin = torch.pow(10.0, sar_db / 10.0)
    # average over the channel axis generically (assumed to be the third from the end).
    avg = lin.mean(dim=-3, keepdim=True)           # average VH/VV → (..,1,H,W)
    avg_n = _minmax_norm(avg)
    # map so low backscatter ⇒ large weight (since exp(1−x) is bigger for small x)
    # a tunable exp_scale (default 1.0). gives a one-knob way to make the weighting gentler/stronger
    w = torch.exp(exp_scale * (1.0 - avg_n))
    return w.expand_as(sar_db)                      # broadcast to (..,2,H,W)

def compute_weights(
    sar_db: torch.Tensor,
    mode: Mode = "per_channel",
    exp_scale: float = 1.0,
) -> torch.Tensor:
    """
    Compute polarization-aware weights from SAR in dB.
    Args:
        sar_db: Tensor of shape (2,H,W) or (B,2,H,W), values in dB.
        mode: "per_channel" or "shared_avg"
        exp_scale: multiplier inside exp(); 1.0 matches manuscript.
    Returns:
        weights with same shape and device as sar_db.
    """
    if sar_db.dim() == 3:
        sar_db = sar_db.unsqueeze(0)  # (1,2,H,W) for unified handling
        squeeze_back = True
    else:
        squeeze_back = False

    if mode == "per_channel":
        w = _per_channel(sar_db, exp_scale)
    elif mode == "shared_avg":
        w = _shared_avg(sar_db, exp_scale)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return w.squeeze(0) if squeeze_back else w

