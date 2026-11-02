# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neyman / power allocators (R4, D14): turn error, weight and cost maps into additional ray rates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


def neyman_fill(priority: torch.Tensor, counts: torch.Tensor, budget: float) -> torch.Tensor:
    """Water-fill ``budget`` extra samples over pixels so totals follow ``priority`` given existing ``counts``.

    Solves ``n_i = max(c_i, lam * a_i)`` with ``sum(x) = budget`` and ``x = n - c >= 0``. For priority
    ``a = sigma * sqrt(w / cost)`` (``gamma = 1``) this is exactly the Neyman solution of ``min sum w sigma^2 / n``.
    For any other priority (the D14 power allocation ``gamma != 1``) it is the same fill rule applied heuristically
    and makes no optimality claim. Pixels with ``priority <= 0`` get nothing.

    Args:
        priority: ``[P]`` float64 priorities ``a``.
        counts: ``[P]`` float64 existing sample counts ``c``.
        budget: number of additional samples to distribute.

    Returns:
        ``[P]`` float64 ``x >= 0`` with ``x.sum() == budget`` up to rounding (zeros if ``budget <= 0`` or no
        pixel has positive priority). The caller may renormalize.
    """
    x = torch.zeros_like(priority)
    idx = priority > 0
    if budget <= 0 or not idx.any():
        return x
    a, c = priority[idx], counts[idx]
    bp = c / a
    order = torch.argsort(bp)
    a_s, c_s, bp_s = a[order], c[order], bp[order]
    lam = (budget + torch.cumsum(c_s, 0)) / torch.cumsum(a_s, 0)
    consistent = torch.nonzero(bp_s < lam)
    k = int(consistent.max()) + 1 if consistent.numel() > 0 else 1  # largest consistent active set; k = 1 is valid
    x[idx] = torch.clamp(lam[k - 1] * a - c, min=0)
    return x


def _sanitize_error(error: torch.Tensor) -> torch.Tensor:
    e = error.to(torch.float64)
    finite = torch.isfinite(e)
    top = e[finite].max().clamp(min=0) if finite.any() else e.new_zeros(())
    e = torch.where(torch.isposinf(e), top, e)
    return torch.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0)


def _sanitize_weight(weight: torch.Tensor) -> torch.Tensor:
    w = weight.to(torch.float64)
    return torch.where(torch.isfinite(w), w, torch.zeros_like(w)).clamp(min=0)


@dataclass(frozen=True)
class _NeymanAllocator:
    """Shared rates: ``a = (b / sqrt(cost)) ** gamma`` on eligible (``weight > 0``) pixels, then ``neyman_fill``."""

    gamma: float = 1.0

    def __post_init__(self):
        if not math.isfinite(self.gamma) or self.gamma < 0:
            raise ValueError(f"gamma must be finite and >= 0, got {self.gamma}")

    def _base_priority(self, sigma: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def _exponent(self) -> float:
        return self.gamma

    def rates(
        self,
        error: torch.Tensor,
        weight: torch.Tensor,
        cost: Optional[torch.Tensor],
        counts: torch.Tensor,
        budget_rays: float,
    ) -> torch.Tensor:
        """Additional rays ``[H, W]`` float64, ``>= 0``, summing to ``budget_rays`` if any pixel is eligible (R4)."""
        err = _sanitize_error(error)
        w = _sanitize_weight(weight).to(err.device)
        if cost is None:
            cost64 = torch.ones_like(err)
        else:
            cost64 = cost.to(device=err.device, dtype=torch.float64)
            if not bool((torch.isfinite(cost64) & (cost64 > 0)).all()):
                raise ValueError("cost must be finite and > 0 everywhere")
        c = counts.to(device=err.device, dtype=torch.float64)

        eligible = w > 0
        if budget_rays <= 0 or not bool(eligible.any()):
            return torch.zeros_like(err)

        sigma = torch.sqrt(err * c.clamp(min=1.0))
        b = self._base_priority(sigma, w)
        a = torch.where(eligible, (b / cost64.sqrt()) ** self._exponent, torch.zeros_like(b))
        if float(a.sum()) == 0.0:
            a = eligible.to(torch.float64)  # uniform fallback

        x = neyman_fill(a.reshape(-1), c.reshape(-1), float(budget_rays))
        x = x * (float(budget_rays) / x.sum())
        return x.reshape(err.shape)


@dataclass(frozen=True)
class UniformAllocator(_NeymanAllocator):
    """Equalizes totals over eligible pixels. ``gamma`` is accepted for interface uniformity only (no effect)."""

    @property
    def _exponent(self) -> float:
        return 1.0

    def _base_priority(self, sigma, weight):
        return torch.ones_like(sigma)


@dataclass(frozen=True)
class VarianceAllocator(_NeymanAllocator):
    """Priority ``sigma`` (weight only decides eligibility). ``gamma = 1`` is Neyman allocation for ``w = 1``."""

    def _base_priority(self, sigma, weight):
        return sigma


@dataclass(frozen=True)
class PerceptualVarianceAllocator(_NeymanAllocator):
    """Priority ``sigma * sqrt(w)``. ``gamma = 1`` is exact Neyman allocation minimizing ``sum w sigma^2 / n``."""

    def _base_priority(self, sigma, weight):
        return sigma * weight.sqrt()
