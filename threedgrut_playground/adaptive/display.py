# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Display model for perceptual weights: sRGB EOTF, CIELAB L* slope and the SDR display luminance model (D4, D9)."""

from __future__ import annotations

import torch

from threedgrut_playground.adaptive.image_ops import LUMA_REC709

# CIE definitions. The linear segment below CIELAB_EPS bounds the L* slope (and the weight) near black.
CIELAB_EPS = (6 / 29) ** 3
CIELAB_KAPPA = 24389 / 27

# D4: typical SDR desktop display with a 1000:1 black level (cd/m^2).
L_PEAK = 200.0
L_BLACK = 0.2
# TVI knee (cd/m^2): Weber (dL ~ L) at photopic levels, De Vries-Rose (dL ~ sqrt(L)) toward the mesopic range.
TVI_KNEE = 1.0

_SRGB_THRESHOLD = 0.04045


def srgb_eotf(v: torch.Tensor) -> torch.Tensor:
    """sRGB EOTF (encoded -> linear); input clamped to [0, 1]."""
    v = v.clamp(0.0, 1.0)
    return torch.where(v <= _SRGB_THRESHOLD, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def srgb_eotf_derivative(v: torch.Tensor) -> torch.Tensor:
    """d(linear)/d(encoded) of the sRGB EOTF; input clamped to [0, 1]."""
    v = v.clamp(0.0, 1.0)
    power = (2.4 / 1.055) * ((v + 0.055) / 1.055) ** 1.4
    return torch.where(v <= _SRGB_THRESHOLD, torch.full_like(v, 1.0 / 12.92), power)


def lstar_slope(y_prime: torch.Tensor) -> torch.Tensor:
    """``dL*/dY'``: CIELAB L* slope per unit encoded luma Y' (white Y_n = 1).

    ``dL*/dY`` is ``CIELAB_KAPPA`` on the linear segment (``Y <= CIELAB_EPS``), else ``(116/3) Y^(-2/3)``; multiplied by
    ``dY/dY' = srgb_eotf_derivative(Y')``.
    """
    y = srgb_eotf(y_prime)
    linear = y <= CIELAB_EPS
    safe = torch.where(linear, torch.ones_like(y), y)
    dl_dy = torch.where(linear, torch.full_like(y, CIELAB_KAPPA), (116.0 / 3.0) * safe ** (-2.0 / 3.0))
    return dl_dy * srgb_eotf_derivative(y_prime)


def display_luminance_cd(mean_rgb: torch.Tensor) -> torch.Tensor:
    """Displayed luminance in cd/m^2 of display-encoded ``mean_rgb [H, W, 3]`` -> ``[H, W]`` (D4)."""
    w = torch.tensor(LUMA_REC709, dtype=mean_rgb.dtype, device=mean_rgb.device)
    return L_PEAK * (srgb_eotf(mean_rgb) * w).sum(dim=-1) + L_BLACK


def tvi(l_cd: torch.Tensor) -> torch.Tensor:
    """Threshold-vs-intensity scale ``sqrt(L (L + TVI_KNEE))``: Weber for L >> knee, De Vries-Rose for L << knee."""
    return torch.sqrt(l_cd * (l_cd + TVI_KNEE))
