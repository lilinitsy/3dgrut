# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pluggable stage interfaces of the adaptive-sampling loop: estimate -> weight -> allocate -> round -> enforce."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

import torch


@dataclass
class WeightContext:
    """Per-pixel inputs for weight terms (all tensors share the image grid ``[H, W]``).

    Attributes:
        mean_rgb: current display-encoded HDR mean ``[H, W, 3]``.
        luminance: Rec.709 luma Y' of ``mean_rgb`` ``[H, W]``.
        mesh_mask: bool ``[H, W]``, visible-mesh union mask.
        view_dirs: unit view directions ``[H, W, 3]``.
        camera: the camera the frame is rendered with.
        gaze_dir: optional gaze direction (reserved).
        noise_var: optional winsorized variance of the mean ``[H, W]``.
        raw_var_of_mean: optional raw (not winsorized) luma variance of the mean ``[H, W]``; when given, the
            display-clip margin uses it, so the margin reflects outliers that inflate the raw ``mean_rgb`` it gates.
    """

    mean_rgb: torch.Tensor
    luminance: torch.Tensor
    mesh_mask: torch.Tensor
    view_dirs: torch.Tensor
    camera: Any
    gaze_dir: Optional[torch.Tensor] = None
    noise_var: Optional[torch.Tensor] = None
    raw_var_of_mean: Optional[torch.Tensor] = None


@runtime_checkable
class ErrorEstimator(Protocol):
    """Estimates per-pixel error from sample statistics."""

    def estimate(self, stats: Any, ctx: WeightContext) -> torch.Tensor:
        """Return the expected squared error of the current mean at ``stats.counts()``, modelled as sigma^2 / n (R2).

        sigma^2 is the display (winsorized) variance. Allocators recover ``sigma = sqrt(error * max(counts, 1))``.
        """
        ...


@runtime_checkable
class WeightTerm(Protocol):
    """A multiplicative perceptual weight factor."""

    name: str

    def __call__(self, ctx: WeightContext) -> torch.Tensor:
        """Return a non-negative weight ``[H, W]`` for this term."""
        ...


@runtime_checkable
class Allocator(Protocol):
    """Turns error, weight and cost maps into additional ray rates."""

    def rates(
        self,
        error: torch.Tensor,
        weight: torch.Tensor,
        cost: Optional[torch.Tensor],
        counts: torch.Tensor,
        budget_rays: float,
    ) -> torch.Tensor:
        """Return the **additional** rays for this iteration (R4).

        float64, >= 0, summing exactly to ``budget_rays`` whenever any pixel is eligible. Eligibility is
        ``weight > 0`` for every allocator; ``cost=None`` means ones.
        """
        ...


@runtime_checkable
class Rounder(Protocol):
    """Rounds fractional rates to integer ray counts."""

    def round(self, rates: torch.Tensor, step: int) -> torch.Tensor:
        """Return integer counts. ``step`` is the iteration index (R5); tie-breaks use seeded permutations."""
        ...


@runtime_checkable
class BudgetEnforcer(Protocol):
    """Enforces per-pixel caps and the total ray budget on integer counts."""

    def enforce(self, counts: torch.Tensor, budget_rays: int) -> torch.Tensor:
        """Return counts whose total does not exceed ``budget_rays`` (R5).

        Per-pixel caps (e.g. ``max_per_pixel``) apply to the new rays of one launch; trimming ties use a seeded
        permutation, not image order.
        """
        ...
