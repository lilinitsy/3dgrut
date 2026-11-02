# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-pixel cap and total-budget enforcement on integer ray counts."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import torch


@dataclass(frozen=True)
class SimpleBudget:
    """Caps counts at ``max_per_pixel`` and trims the largest counts first until the total fits the budget.

    Algorithm: ``c = clamp(counts, 0, max_per_pixel)``. If ``sum(c) <= budget`` the clamped counts are returned.
    Otherwise the largest integer water level ``T`` with ``sum(min(c, T)) <= budget`` is found by binary search and
    ``min(c, T)`` is returned, plus +1 on up to ``budget - sum(min(c, T))`` pixels with ``c > T``. That leftover is
    fewer than ``#{c > T}`` and goes to the largest ``c`` first; ties are ordered by a fixed permutation seeded with
    ``seed`` (no top-of-image pattern). The result never exceeds ``budget_rays`` or ``max_per_pixel``.

    Note: trimming makes ``StochasticRounder`` output slightly biased low on the largest counts, since its unbiased
    draw is cut back where it is largest. Trimmed rays are not lost: they are returned to the renderer's remaining
    budget and flow to the next iteration or the final top-up.
    """

    max_per_pixel: int
    seed: int = 0

    def __post_init__(self):
        m = self.max_per_pixel
        if isinstance(m, bool) or not isinstance(m, Integral) or m < 1:
            raise ValueError(f"max_per_pixel must be an int >= 1, got {m!r}")

    def enforce(self, counts: torch.Tensor, budget_rays: int) -> torch.Tensor:
        """Return int64 counts with the shape and device of ``counts``, ``<= max_per_pixel`` and total ``<= budget``."""
        c = counts.to(torch.int64).clamp(0, int(self.max_per_pixel))
        budget = int(budget_rays)
        if budget <= 0:
            return torch.zeros_like(c)
        if int(c.sum()) <= budget:
            return c
        flat = c.reshape(-1)
        lo, hi = 0, int(flat.max())  # sum(min(c, 0)) = 0 <= budget; sum(min(c, hi)) > budget
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if int(torch.clamp(flat, max=mid).sum()) <= budget:
                lo = mid
            else:
                hi = mid
        t = lo
        out = torch.clamp(flat, max=t)
        leftover = budget - int(out.sum())
        if leftover > 0:
            perm = torch.randperm(flat.numel(), generator=torch.Generator().manual_seed(int(self.seed))).to(flat.device)
            _, order = torch.sort(flat[perm], descending=True, stable=True)
            chosen = perm[order[:leftover]]  # leftover < #{c > T}, so every chosen pixel has c > T
            out[chosen] += 1
        return out.reshape(c.shape)
