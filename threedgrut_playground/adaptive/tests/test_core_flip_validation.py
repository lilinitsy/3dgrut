# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""D9 validation: the default LuminanceAdaptationTerm tracks FLIP's response to grey noise across grey levels."""

import math

import numpy as np
import pytest
import torch

from threedgrut_playground.adaptive.image_ops import luminance
from threedgrut_playground.adaptive.protocols import WeightContext
from threedgrut_playground.adaptive.weights import LuminanceAdaptationTerm

flip = pytest.importorskip("flip_evaluator")

LEVELS = (0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9)
PPD = 512 / 60  # D14: ppd = H / vertical fov_deg (512 px at 60 deg ~ 8.53)


def _grey(v, shape):
    return torch.full((*shape, 3), float(v), dtype=torch.float32)


def _ctx(rgb):
    h, w = rgb.shape[:2]
    return WeightContext(
        mean_rgb=rgb,
        luminance=luminance(rgb),
        mesh_mask=torch.ones(h, w, dtype=torch.bool),
        view_dirs=torch.zeros(h, w, 3),
        camera=None,
    )


def _flip_mean(level, sigma):
    rng = np.random.default_rng(1234)
    ref = np.full((256, 256, 3), level, np.float32)
    test = np.clip(ref + rng.normal(0.0, sigma, (256, 256, 1)).astype(np.float32), 0.0, 1.0)  # grey noise
    # flip-evaluator 1.7 returns mean_error = -1 with computeMeanError=False, so take the mean of the error map.
    err_map, _, _ = flip.evaluate(
        ref, test, "LDR", inputsRGB=True, applyMagma=False, computeMeanError=False, parameters={"ppd": PPD}
    )
    return float(np.asarray(err_map, dtype=np.float64).mean())


_E1 = {}
_E2 = {}


def _errors():
    """FLIP mean errors for noise sigma 0.005 and 0.010 at every level (cached across the two tests)."""
    if not _E1:
        _E1.update({y: _flip_mean(y, 0.005) for y in LEVELS})
        _E2.update({y: _flip_mean(y, 0.010) for y in LEVELS})
    return _E1, _E2


def _exponents():
    e1, e2 = _errors()
    return {y: math.log(e2[y] / e1[y]) / math.log(2.0) for y in LEVELS}  # FLIP's response exponent in noise amplitude


def test_flip_response_exponent_in_range():
    q = _exponents()
    print("q", q)
    assert all(0.3 < q[y] < 1.2 for y in LEVELS), q  # ~0.7 expected (FLIP's colour exponent)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "D9 open: L* weight is outside the factor-1.5 tolerance [0.667, 1.5] of FLIP's implied weight. Measured "
        "w_flip/w_model at Y'=0.05,0.1,0.2,0.35,0.5,0.7,0.9 = 0.571, 0.543, 0.653, 0.818, 1.0, 1.278, 1.606 "
        "(q median 0.768). FLIP is flatter than the L* slope: darks over-weighted, brights under-weighted. "
        "Changing D9 is the user's decision."
    ),
)
def test_default_luminance_weight_tracks_flip():
    """D9 check, currently a known failure (xfail strict, raises=AssertionError).

    Strict only reports an unexpected pass, i.e. when every ratio has moved inside the tolerance; other changes in
    the numbers are not noticed, and unrelated exceptions fail loudly.

    Measured (PPD 512/60, grey noise 0.005 vs 0.010): w_flip / w_model at Y' = 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9 is
    0.571, 0.543, 0.653, 0.818, 1.0, 1.278, 1.606 against a tolerance of [0.667, 1.5]; median q = 0.768.

    Reviewer finding: the implied weight depends on the noise amplitude (iso-error bisection: sigma_ref = 0.005 gives
    ratios 0.60, 0.61, 0.70, 0.85, 1.22, 1.46 and sigma_ref = 0.02 gives 0.52, 0.46, 0.60, 0.79, 1.33, 1.71 at
    Y' = 0.05, 0.1, 0.2, 0.35, 0.7, 0.9), so FLIP is not separable as w * sigma^2, and FLIP is flatter than L* at
    every amplitude.
    """
    _, e2 = _errors()
    q = _exponents()
    qm = float(np.median(list(q.values())))
    w_flip = {y: (e2[y] / e2[0.5]) ** (2.0 / qm) for y in LEVELS}  # FLIP-implied weight per unit noise variance
    w_model = {y: LuminanceAdaptationTerm()(_ctx(_grey(y, (1, 1))))[0, 0].item() for y in LEVELS}
    ratio = {y: w_flip[y] / w_model[y] for y in LEVELS}
    print("qm", qm)
    print(f"{'Y':>6} {'w_flip':>9} {'w_model':>9} {'ratio':>8}")
    for y in LEVELS:
        print(f"{y:6.2f} {w_flip[y]:9.4f} {w_model[y]:9.4f} {ratio[y]:8.3f}")
    assert all(1 / 1.5 <= r <= 1.5 for r in ratio.values()), ratio  # stated tolerance: within a factor 1.5
