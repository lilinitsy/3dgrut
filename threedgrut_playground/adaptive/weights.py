# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Perceptual weight terms (D4, D9) and their product.

All terms are computed in float64 on sanitized inputs (NaN/inf in ``mean_rgb``/``luminance`` become 0, and
``Y' = clamp(ctx.luminance, 0, 1)``) and returned as float32 ``[H, W]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Sequence

import torch

from threedgrut_playground.adaptive.display import (
    L_PEAK,
    display_luminance_cd,
    lstar_slope,
    srgb_eotf_derivative,
    tvi,
)
from threedgrut_playground.adaptive.image_ops import class_blur2d, gaussian_blur2d
from threedgrut_playground.adaptive.protocols import WeightContext, WeightTerm


def _finite_f64(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x.to(torch.float64), nan=0.0, posinf=0.0, neginf=0.0)


def _y_prime(ctx: WeightContext) -> torch.Tensor:
    return _finite_f64(ctx.luminance).clamp(0.0, 1.0)


def _ref(value: float, like: torch.Tensor) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float64, device=like.device)


@dataclass(frozen=True)
class LuminanceAdaptationTerm:
    """D9 default: FLIP's colour-pipeline sensitivity, ``w = (lstar_slope(Y') / lstar_slope(y_ref))^2``.

    Pointwise (no pooling); treats the pixel as grey with luma ``Y'`` (R8). ``y_ref`` only normalizes (w = 1 at
    encoded mid-grey); the allocators are scale-invariant in w. The minimum 0.4886 holds for ``Y' <= 0.04045`` (both
    the EOTF and L* are linear there) and the maximum 1.7003 is at the L* knee, ``Y' = 0.09221``.
    """

    y_ref: float = 0.5
    name: ClassVar[str] = "luminance_adaptation"

    def __call__(self, ctx: WeightContext) -> torch.Tensor:
        yp = _y_prime(ctx)
        w = (lstar_slope(yp) / lstar_slope(_ref(self.y_ref, yp))) ** 2
        return w.to(torch.float32)


@dataclass(frozen=True)
class TviAdaptationTerm:
    """Selectable alternative (judged with CVVDP, not FLIP): local adaptation luminance with a TVI sensitivity.

    ``L_a`` is the Gaussian-pooled displayed luminance; ``g = L_PEAK * eotf'(Y') / tvi(L_a)`` is the weight per unit
    encoded error (Weber with a De Vries-Rose floor), normalized by the same expression for a uniform field at
    ``y_ref``; ``w = min((g / g_ref)^2, w_max)``.
    """

    adapt_sigma_px: float = 16.0
    y_ref: float = 0.5
    w_max: float = 100.0
    name: ClassVar[str] = "tvi_adaptation"

    def __call__(self, ctx: WeightContext) -> torch.Tensor:
        yp = _y_prime(ctx)
        l_a = gaussian_blur2d(display_luminance_cd(_finite_f64(ctx.mean_rgb)), self.adapt_sigma_px)
        g = L_PEAK * srgb_eotf_derivative(yp) / tvi(l_a)
        yr = _ref(self.y_ref, yp)
        l_ref = display_luminance_cd(yr.expand(3))  # uniform grey field at y_ref
        g_ref = L_PEAK * srgb_eotf_derivative(yr) / tvi(l_ref)
        return torch.clamp((g / g_ref) ** 2, max=self.w_max).to(torch.float32)


@dataclass(frozen=True)
class ContrastMaskingTerm:
    """Legge-Foley threshold elevation on local RMS Weber contrast in linear light, class-aware.

    ``mu`` and ``v`` are class-blurred mean and variance of the displayed luminance (``mesh_mask`` is the class);
    with ``noise_var`` the noise contribution ``(L_PEAK * eotf'(Y'))^2 * noise_var`` is subtracted from ``v``.
    ``C = sqrt(v) / mu``, ``e = (1 + (C / c0)^2)^(epsilon / 2)``, ``w = e^-2``.

    Known limitation (R8, self-masking): contrast is measured on the *noisy* current mean, so with few samples the
    noise itself raises the measured contrast and lowers the weight. The ``noise_var`` subtraction only removes this
    on average, because the variance estimate is itself noisy. This is documented, not corrected.
    """

    sigma_px: float = 2.0
    c0: float = 0.02
    epsilon: float = 0.7
    name: ClassVar[str] = "contrast_masking"

    def __call__(self, ctx: WeightContext) -> torch.Tensor:
        lum = display_luminance_cd(_finite_f64(ctx.mean_rgb))
        m = ctx.mesh_mask
        mu, _ = class_blur2d(lum, m, self.sigma_px)
        ex2, _ = class_blur2d(lum * lum, m, self.sigma_px)
        v = ex2 - mu * mu
        if ctx.noise_var is not None:
            gain = L_PEAK * srgb_eotf_derivative(_y_prime(ctx))
            nv, _ = class_blur2d(gain * gain * _finite_f64(ctx.noise_var), m, self.sigma_px)
            v = v - nv
        v = v.clamp_min(0.0)
        c = torch.sqrt(v) / mu
        e = (1.0 + (c / self.c0) ** 2) ** (self.epsilon / 2.0)
        return (e**-2).to(torch.float32)


@dataclass(frozen=True)
class DisplayClipTerm:
    """Fade to 0 once the mean exceeds display white (D4): such a pixel's error is invisible after ``clamp(.,0,1)``.

    ``m = min_c mean_rgb`` (a pixel is invisible only once all channels clip); ``delta = max(margin_min,
    noise_k * sqrt(v))`` with ``v = raw_var_of_mean`` when the context has it, else ``noise_var`` (``margin_min`` without
    either); ``t = clamp((m - 1) / delta, 0, 1)``; ``w = 1 - t^2 (3 - 2t)``.

    The margin must come from the same (raw) samples as ``m``: one outlier in a small pilot raises the raw mean far above
    white but cannot move the winsorized variance, so a winsorized margin would clip the pixel, and a clipped pixel gets
    no new rays to correct its mean. With the raw variance the outlier widens the margin as much as it raises the mean.
    """

    margin_min: float = 0.05
    noise_k: float = 2.0
    name: ClassVar[str] = "display_clip"

    def __call__(self, ctx: WeightContext) -> torch.Tensor:
        m = _finite_f64(ctx.mean_rgb).min(dim=-1).values
        var = ctx.raw_var_of_mean if ctx.raw_var_of_mean is not None else ctx.noise_var
        if var is None:
            delta = torch.full_like(m, self.margin_min)
        else:
            delta = (self.noise_k * _finite_f64(var).clamp_min(0.0).sqrt()).clamp_min(self.margin_min)
        t = ((m - 1.0) / delta).clamp(0.0, 1.0)
        return (1.0 - t * t * (3.0 - 2.0 * t)).to(torch.float32)


def combine_weights(terms: Sequence[WeightTerm], ctx: WeightContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Product of the term outputs and the per-term maps.

    Each output is shape-checked against ``ctx.luminance`` (ValueError on mismatch) and sanitized (NaN/inf -> 0,
    ``clamp_min(0)``). Empty ``terms`` give ones; a duplicate ``name`` raises ValueError.
    """
    shape = tuple(ctx.luminance.shape)
    w = torch.ones(shape, dtype=torch.float32, device=ctx.luminance.device)
    per_term: dict[str, torch.Tensor] = {}
    for term in terms:
        if term.name in per_term:
            raise ValueError(f"duplicate weight term name {term.name!r}")
        out = term(ctx)
        if tuple(out.shape) != shape:
            raise ValueError(f"weight term {term.name!r} returned shape {tuple(out.shape)}, expected {shape}")
        out = torch.nan_to_num(out.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        per_term[term.name] = out
        w = w * out
    return w, per_term
