# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Error estimators: the expected squared error of the current per-pixel mean (R2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from threedgrut_playground.adaptive.image_ops import class_blur2d
from threedgrut_playground.adaptive.protocols import WeightContext
from threedgrut_playground.adaptive.stats import (  # noqa: F401  (WINSOR_DELTA re-exported)
    WINSOR_DELTA,
    SampleStats,
)


@dataclass(frozen=True)
class VarianceOfMeanEstimator:
    """Error = sigma_hat^2 / max(n, 1), with sigma_hat the class-aware smoothed winsorized (display) std (R2, R3, D13).

    ``sigma = sqrt(display_variance)`` is the plain sample standard deviation (no c4 correction). It is smoothed, not
    sigma^2, with a Gaussian of ``sigma_px`` over valid (``n >= 2``) pixels of the same mesh class only. Pixels with no
    support fall back to the true median of the valid same-class sigma, then of all valid sigma, then 0.
    """

    sigma_px: float = 1.0

    def estimate(self, stats: SampleStats, ctx: Optional[WeightContext]) -> torch.Tensor:
        n = stats.counts()
        valid = n >= 2
        sigma = stats.display_variance().to(torch.float64).sqrt()
        classes = ctx.mesh_mask.to(torch.bool) if ctx is not None else torch.ones_like(valid)
        smoothed, support = class_blur2d(sigma, classes, self.sigma_px, valid)
        gap = support <= 1e-12
        if bool(gap.any()):
            all_valid = _median(sigma[valid]) if bool(valid.any()) else 0.0
            fallback = torch.zeros_like(sigma)
            for cls in (classes, ~classes):
                sel = valid & cls
                value = _median(sigma[sel]) if bool(sel.any()) else all_valid
                fallback = torch.where(cls, torch.full_like(sigma, value), fallback)
            smoothed = torch.where(gap, fallback, smoothed)
        return (smoothed * smoothed / n.clamp_min(1).to(torch.float64)).to(torch.float32)


def _median(x: torch.Tensor) -> float:
    """True median (mean of the two middle values for even counts)."""
    return float(torch.quantile(x, 0.5))
