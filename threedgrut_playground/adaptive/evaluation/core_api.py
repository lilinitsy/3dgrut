# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single import point for the adaptive core (plan B). Every other evaluation module imports core names from here,
so a rename on the core side is reconciled in this file only."""

# ruff: noqa: F401

from __future__ import annotations

import dataclasses
from typing import List

from ..allocate import PerceptualVarianceAllocator, UniformAllocator, VarianceAllocator
from ..budget import SimpleBudget
from ..display import L_BLACK, L_PEAK
from ..dump import MapDumper
from ..estimators import VarianceOfMeanEstimator
from ..image_ops import luminance
from ..protocols import WeightContext
from ..rays import RayBatch, center_view_dirs, rays_for_counts
from ..renderer import MESH_VISIBLE_T, AdaptiveRenderer, AdaptiveResult
from ..rounding import DeterministicRounder, StochasticRounder
from ..stats import SampleStats
from ..telemetry import CostTelemetry
from ..weights import (
    ContrastMaskingTerm,
    DisplayClipTerm,
    LuminanceAdaptationTerm,
    TviAdaptationTerm,
    combine_weights,
)

# The core's top-up round count is private (``renderer._TOPUP_ROUNDS``); the evaluation keeps its own copy.
TOPUP_ROUNDS = 2

CORE_NAMES = (
    "RayBatch",
    "rays_for_counts",
    "center_view_dirs",
    "SampleStats",
    "WeightContext",
    "combine_weights",
    "LuminanceAdaptationTerm",
    "TviAdaptationTerm",
    "ContrastMaskingTerm",
    "DisplayClipTerm",
    "VarianceOfMeanEstimator",
    "UniformAllocator",
    "VarianceAllocator",
    "PerceptualVarianceAllocator",
    "DeterministicRounder",
    "StochasticRounder",
    "SimpleBudget",
    "CostTelemetry",
    "MapDumper",
    "AdaptiveRenderer",
    "AdaptiveResult",
    "MESH_VISIBLE_T",
    "luminance",
    "L_PEAK",
    "L_BLACK",
)


def telemetry_total_ms(telemetry, start: int = 0) -> float:
    """Sum of resolved launch times recorded from ``start``."""
    return float(telemetry.total_ms(start))


def telemetry_cost_fits(telemetry) -> dict[str, dict]:
    """``ms_per_ray()`` as plain dicts (``{}`` if the telemetry has no such fit)."""
    fits = getattr(telemetry, "ms_per_ray", None)
    if fits is None:
        return {}
    return {kind: dataclasses.asdict(fit) for kind, fit in fits().items()}


def make_weight_terms(model: str) -> List:
    """Weight terms for a perceptual model name: ``"cielab"``, ``"cielab_nomask"`` (cielab without contrast masking, the
    masking ablation) or ``"tvi"``."""
    if model == "cielab":
        return [LuminanceAdaptationTerm(), ContrastMaskingTerm(), DisplayClipTerm()]
    if model == "cielab_nomask":
        return [LuminanceAdaptationTerm(), DisplayClipTerm()]
    if model == "tvi":
        return [TviAdaptationTerm(), ContrastMaskingTerm(), DisplayClipTerm()]
    raise ValueError(f"unknown weight model {model!r}; expected 'cielab', 'cielab_nomask' or 'tvi'")
