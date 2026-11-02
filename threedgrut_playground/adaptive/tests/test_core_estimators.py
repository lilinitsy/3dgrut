# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for threedgrut_playground.adaptive.estimators."""

import pytest
import torch

from threedgrut_playground.adaptive.estimators import VarianceOfMeanEstimator
from threedgrut_playground.adaptive.image_ops import class_blur2d
from threedgrut_playground.adaptive.oracle import smooth_sigma_like_b
from threedgrut_playground.adaptive.protocols import ErrorEstimator, WeightContext
from threedgrut_playground.adaptive.stats import SampleStats

PAT = [0.2, 0.8, 0.2, 0.8]  # unbiased variance 0.12, n = 4 -> error 0.03 (inside the display range)


def _stats(h, w, per_pixel):
    """SampleStats with grey samples [v, v, v] (luma = v) per pixel; sample_index = arange per pixel."""
    st = SampleStats(h, w, "cpu")
    pid, rgb, idx = [], [], []
    for p, vals in per_pixel.items():
        for i, v in enumerate(vals):
            pid.append(p)
            rgb.append([v, v, v])
            idx.append(i)
    st.add(
        torch.tensor(pid, dtype=torch.int64),
        torch.tensor(rgb, dtype=torch.float32),
        torch.tensor(idx, dtype=torch.int64),
    )
    return st


def _ctx(mask):
    h, w = mask.shape
    return WeightContext(
        mean_rgb=torch.zeros(h, w, 3),
        luminance=torch.zeros(h, w),
        mesh_mask=mask,
        view_dirs=torch.zeros(h, w, 3),
        camera=None,
    )


def test_constant_variance_gives_sigma2_over_n():
    e = VarianceOfMeanEstimator().estimate(_stats(4, 4, dict.fromkeys(range(16), PAT)), None)
    assert e.dtype == torch.float32
    torch.testing.assert_close(e, torch.full((4, 4), 0.03))


def test_allocation_variance_is_winsorized():
    e = VarianceOfMeanEstimator().estimate(_stats(2, 2, {p: [0.0, 3.0, 0.0, 3.0] for p in range(4)}), None)
    torch.testing.assert_close(e, torch.full((2, 2), 4 * 0.525**2 / 3 / 4))  # not 3.0 / 4


def test_count_below_two_borrows_neighbour_sigma():
    s = dict.fromkeys(range(16), PAT)
    s[5] = [0.5]
    assert VarianceOfMeanEstimator().estimate(_stats(4, 4, s), None)[1, 1].item() == pytest.approx(0.12, rel=1e-5)


def test_isolated_pixel_without_support_uses_median():
    s = {y * 9 + x: PAT for y in range(9) for x in range(3)} | {y * 9 + 8: [0.5] for y in range(9)}
    e = VarianceOfMeanEstimator(sigma_px=0.5).estimate(_stats(9, 9, s), None)
    torch.testing.assert_close(e[:, 8], torch.full((9,), 0.12))


def test_smoothing_does_not_leak_across_mesh_boundary():
    mask = torch.zeros(8, 8, dtype=torch.bool)
    mask[:, :4] = True
    s = {y * 8 + x: (PAT if x < 4 else [0.5] * 4) for y in range(8) for x in range(8)}
    e = VarianceOfMeanEstimator(sigma_px=2.0).estimate(_stats(8, 8, s), _ctx(mask))
    torch.testing.assert_close(e[:, :4], torch.full((8, 4), 0.03))
    assert e[:, 4:].abs().max() < 1e-12


def test_zero_variance_and_no_valid_pixels_give_zero():
    for s in ({p: [0.5] * 3 for p in range(9)}, {p: [0.7] for p in range(9)}):
        e = VarianceOfMeanEstimator().estimate(_stats(3, 3, s), None)
        assert torch.isfinite(e).all() and e.abs().max() < 1e-12


def test_conforms_to_protocol():
    assert isinstance(VarianceOfMeanEstimator(), ErrorEstimator)


def test_smoothed_sigma_matches_oracle_including_even_count_median_fallback():
    # 12x12, mesh = columns 0-5. A 5x6 block of n = 1 pixels inside the mesh leaves its centre rows without support at
    # sigma_px = 0.5 (radius 2), so the median fallback runs over an even number (42) of valid mesh pixels.
    h, w = 12, 12
    mask = torch.zeros(h, w, dtype=torch.bool)
    mask[:, :6] = True
    gen = torch.Generator().manual_seed(3)
    s = {}
    for y in range(h):
        for x in range(w):
            if x < 6 and 3 <= y <= 7:
                s[y * w + x] = [0.4]
            else:
                base = 0.1 + 0.3 * torch.rand(1, generator=gen).item()
                s[y * w + x] = [base, base + 0.2, base + 0.1, base + 0.5][: 4 if (x + y) % 2 else 3]
    st = _stats(h, w, s)
    n = st.counts()
    sigma = st.display_variance().double().sqrt()
    _, sup = class_blur2d(sigma, mask, 0.5, n >= 2)
    assert (sup <= 1e-12).any() and int(((n >= 2) & mask).sum()) % 2 == 0  # fallback with an even-sized median
    e = VarianceOfMeanEstimator(sigma_px=0.5).estimate(st, _ctx(mask))
    expected = smooth_sigma_like_b(sigma, n, mask, 0.5)
    expected = (expected**2 / n.clamp_min(1).double()).float()
    torch.testing.assert_close(e, expected)
