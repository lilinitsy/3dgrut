# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public API of the adaptive core (plan B) and the guarantee that importing it never loads ``engine.py``."""

import subprocess
import sys

from threedgrut_playground.adaptive.tests.kernel_helpers import REPO_ROOT

EXPORTS = [
    "RayBatch",
    "drop_padding",
    "launch_padding",
    "generate_rays",
    "rays_for_counts",
    "pixel_jitter",
    "center_view_dirs",
    "SampleStats",
    "WINSOR_DELTA",
    "WeightContext",
    "ErrorEstimator",
    "WeightTerm",
    "Allocator",
    "Rounder",
    "BudgetEnforcer",
    "VarianceOfMeanEstimator",
    "LuminanceAdaptationTerm",
    "TviAdaptationTerm",
    "ContrastMaskingTerm",
    "DisplayClipTerm",
    "combine_weights",
    "UniformAllocator",
    "VarianceAllocator",
    "PerceptualVarianceAllocator",
    "DeterministicRounder",
    "StochasticRounder",
    "SimpleBudget",
    "CostTelemetry",
    "CostFit",
    "LaunchRecord",
    "MapDumper",
    "AdaptiveRenderer",
    "AdaptiveResult",
    "MESH_VISIBLE_T",
]


def test_public_api_is_complete():
    import threedgrut_playground.adaptive as A

    assert all(n in A.__all__ and hasattr(A, n) for n in EXPORTS)


def test_package_import_does_not_load_engine():
    code = (
        "import sys, threedgrut_playground.adaptive, threedgrut_playground.adaptive.renderer; "
        "assert 'threedgrut_playground.engine' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO_ROOT)
