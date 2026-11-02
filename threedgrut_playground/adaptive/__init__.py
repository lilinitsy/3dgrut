# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adaptive sampling instrumentation for the 3DGRUT playground."""

# --- adaptive core (plan B) ---
from .allocate import PerceptualVarianceAllocator, UniformAllocator, VarianceAllocator
from .budget import SimpleBudget
from .dump import MapDumper
from .estimators import VarianceOfMeanEstimator
from .protocols import (
    Allocator,
    BudgetEnforcer,
    ErrorEstimator,
    Rounder,
    WeightContext,
    WeightTerm,
)
from .rays import (
    RayBatch,
    center_view_dirs,
    drop_padding,
    generate_rays,
    launch_padding,
    pixel_jitter,
    rays_for_counts,
)
from .renderer import MESH_VISIBLE_T, AdaptiveRenderer, AdaptiveResult
from .rounding import DeterministicRounder, StochasticRounder
from .stats import WINSOR_DELTA, SampleStats
from .telemetry import CostFit, CostTelemetry, LaunchRecord
from .weights import (
    ContrastMaskingTerm,
    DisplayClipTerm,
    LuminanceAdaptationTerm,
    TviAdaptationTerm,
    combine_weights,
)

__all__ = [
    "MESH_VISIBLE_T",
    "WINSOR_DELTA",
    "AdaptiveRenderer",
    "AdaptiveResult",
    "Allocator",
    "BudgetEnforcer",
    "ContrastMaskingTerm",
    "CostFit",
    "CostTelemetry",
    "DeterministicRounder",
    "DisplayClipTerm",
    "ErrorEstimator",
    "LaunchRecord",
    "LuminanceAdaptationTerm",
    "MapDumper",
    "PerceptualVarianceAllocator",
    "RayBatch",
    "Rounder",
    "SampleStats",
    "SimpleBudget",
    "StochasticRounder",
    "TviAdaptationTerm",
    "UniformAllocator",
    "VarianceAllocator",
    "VarianceOfMeanEstimator",
    "WeightContext",
    "WeightTerm",
    "center_view_dirs",
    "combine_weights",
    "drop_padding",
    "generate_rays",
    "launch_padding",
    "pixel_jitter",
    "rays_for_counts",
]
