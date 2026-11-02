# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small image operations: Rec.709 luma, Gaussian blur, normalized (masked) blur and class-aware blur."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

LUMA_REC709 = (0.2126, 0.7152, 0.0722)


def luminance(rgb: torch.Tensor) -> torch.Tensor:
    """Rec.709 luma of display-encoded RGB ``[..., 3]`` -> ``[...]``."""
    w = torch.tensor(LUMA_REC709, dtype=rgb.dtype, device=rgb.device)
    return (rgb * w).sum(dim=-1)


def gaussian_blur2d(x: torch.Tensor, sigma_px: float) -> torch.Tensor:
    """Separable Gaussian blur of ``x [H, W]``: taps ``exp(-k^2 / 2 sigma^2)``, ``|k| <= ceil(3 sigma)``, normalized.

    Replicate padding, computed in the input dtype. ``sigma_px <= 0`` returns a clone.
    """
    if sigma_px <= 0:
        return x.clone()
    r = int(math.ceil(3.0 * sigma_px))
    k = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    taps = torch.exp(-(k * k) / (2.0 * sigma_px * sigma_px))
    taps = taps / taps.sum()
    y = F.pad(x[None, None], (r, r, r, r), mode="replicate")
    y = F.conv2d(y, taps.view(1, 1, 1, -1))
    y = F.conv2d(y, taps.view(1, 1, -1, 1))
    return y[0, 0]


def normalized_blur2d(x: torch.Tensor, mask: torch.Tensor, sigma_px: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``blur(where(mask, x, 0)) / blur(mask)`` where ``blur(mask) > 1e-12``, else 0.

    Masked-out values never contribute, even if non-finite. Returns ``(blurred, support = blur(mask))``; with
    ``sigma_px <= 0`` the result is ``(where(mask, x, 0), mask)``.
    """
    m = mask.to(x.dtype)
    zeros = torch.zeros_like(x)
    if sigma_px <= 0:
        return torch.where(mask, x, zeros), m
    num = gaussian_blur2d(torch.where(mask, x, zeros), sigma_px)
    sup = gaussian_blur2d(m, sigma_px)
    ok = sup > 1e-12
    out = torch.where(ok, num / torch.where(ok, sup, torch.ones_like(sup)), torch.zeros_like(num))
    return out, sup


def class_blur2d(
    x: torch.Tensor, classes: torch.Tensor, sigma_px: float, valid: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blur each pixel only over pixels of its own class that are also ``valid``. Returns ``(blurred, support)``."""
    if valid is None:
        valid = torch.ones_like(classes, dtype=torch.bool)
    out_a, sup_a = normalized_blur2d(x, classes & valid, sigma_px)
    out_b, sup_b = normalized_blur2d(x, ~classes & valid, sigma_px)
    return torch.where(classes, out_a, out_b), torch.where(classes, sup_a, sup_b)
