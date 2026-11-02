# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for threedgrut_playground.adaptive.image_ops and protocols."""

import pytest
import torch

from threedgrut_playground.adaptive.image_ops import (
    class_blur2d,
    gaussian_blur2d,
    luminance,
    normalized_blur2d,
)


def test_luminance_rec709():
    rgb = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [2.0, 2.0, 2.0]])
    torch.testing.assert_close(luminance(rgb), torch.tensor([0.2126, 0.7152, 0.0722, 2.0]))


def test_gaussian_blur_constant_identity_and_mass():
    x = torch.full((7, 5), 3.0, dtype=torch.float64)
    torch.testing.assert_close(gaussian_blur2d(x, 1.5), x)
    y = torch.rand(7, 5)
    torch.testing.assert_close(gaussian_blur2d(y, 0.0), y)
    z = torch.zeros(21, 21, dtype=torch.float64)
    z[10, 10] = 1.0
    b = gaussian_blur2d(z, 1.0)
    assert b[10, 13] > 0 and b[10, 14] == 0 and abs(b.sum().item() - 1.0) < 1e-9


def test_normalized_blur_fills_holes_and_reports_zero_support():
    x = torch.full((6, 6), 2.0, dtype=torch.float64)
    m = torch.ones(6, 6, dtype=torch.bool)
    m[2, 3] = False
    x[2, 3] = 999.0
    out, sup = normalized_blur2d(x, m, 1.0)
    torch.testing.assert_close(out, torch.full_like(x, 2.0))
    assert (sup > 0).all()
    out, sup = normalized_blur2d(torch.ones(4, 4), torch.zeros(4, 4, dtype=torch.bool), 1.0)
    assert torch.equal(out, torch.zeros(4, 4)) and torch.equal(sup, torch.zeros(4, 4))


def test_class_blur_does_not_leak_across_classes():
    cls = torch.zeros(8, 8, dtype=torch.bool)
    cls[:, :4] = True
    x = torch.where(cls, 5.0, 1.0).double()
    out, _ = class_blur2d(x, cls, 2.0)
    torch.testing.assert_close(out, x)  # a plain blur would mix 5 and 1 at columns 3/4


def test_masked_values_do_not_leak_even_if_nonfinite():
    x = torch.full((6, 6), 2.0, dtype=torch.float64)
    m = torch.ones(6, 6, dtype=torch.bool)
    m[2, 3] = False
    x[2, 3] = float("nan")
    out, _ = normalized_blur2d(x, m, 1.0)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.full_like(x, 2.0))


def test_normalized_blur_nonpositive_sigma_returns_masked_values_and_mask_support():
    x = torch.arange(16.0).reshape(4, 4)
    m = x % 2 == 0
    out, sup = normalized_blur2d(x, m, 0.0)
    assert torch.equal(out, torch.where(m, x, torch.zeros_like(x)))
    assert torch.equal(sup, m.to(x.dtype))


def test_class_blur_respects_valid_mask():
    cls = torch.zeros(8, 8, dtype=torch.bool)
    cls[:, :4] = True
    valid = torch.ones(8, 8, dtype=torch.bool)
    valid[3, 1] = False
    x = torch.where(cls, 5.0, 1.0).double()
    x[3, 1] = 1e6
    out, sup = class_blur2d(x, cls, 1.5, valid=valid)
    torch.testing.assert_close(out, torch.where(cls, 5.0, 1.0).double())
    assert (sup > 0).all()


def test_blur_preserves_dtype():
    assert gaussian_blur2d(torch.rand(5, 5), 1.0).dtype == torch.float32
    assert gaussian_blur2d(torch.rand(5, 5, dtype=torch.float64), 1.0).dtype == torch.float64


def test_protocols_are_runtime_checkable_and_context_defaults():
    from threedgrut_playground.adaptive.protocols import (
        Allocator,
        BudgetEnforcer,
        ErrorEstimator,
        Rounder,
        WeightContext,
        WeightTerm,
    )

    ctx = WeightContext(mean_rgb=None, luminance=None, mesh_mask=None, view_dirs=None, camera=None)
    assert ctx.gaze_dir is None and ctx.noise_var is None

    class _Est:
        def estimate(self, stats, ctx):
            return None

    class _Term:
        name = "t"

        def __call__(self, ctx):
            return None

    class _Alloc:
        def rates(self, error, weight, cost, counts, budget_rays):
            return None

    class _Round:
        def round(self, rates, step):
            return None

    class _Enf:
        def enforce(self, counts, budget_rays):
            return None

    assert isinstance(_Est(), ErrorEstimator)
    assert isinstance(_Term(), WeightTerm)
    assert isinstance(_Alloc(), Allocator)
    assert isinstance(_Round(), Rounder)
    assert isinstance(_Enf(), BudgetEnforcer)
    assert not isinstance(object(), Allocator)


@pytest.mark.parametrize("sigma", [0.0, 0.7, 1.0, 2.5])
def test_image_ops_match_oracle_local_copies(sigma):
    from threedgrut_playground.adaptive import oracle

    g = torch.Generator().manual_seed(1234)
    for dtype in (torch.float32, torch.float64):
        rgb = torch.rand(9, 11, 3, generator=g, dtype=dtype) * 2
        x = torch.rand(13, 17, generator=g, dtype=dtype) * 4
        mask = torch.rand(13, 17, generator=g) > 0.4
        cls = torch.rand(13, 17, generator=g) > 0.5
        valid = torch.rand(13, 17, generator=g) > 0.2

        assert torch.equal(luminance(rgb), oracle.luminance(rgb))
        assert torch.equal(gaussian_blur2d(x, sigma), oracle.gaussian_blur2d(x, sigma))
        for mine, ref in zip(normalized_blur2d(x, mask, sigma), oracle.normalized_blur2d(x, mask, sigma)):
            assert torch.equal(mine, ref)
        for mine, ref in zip(class_blur2d(x, cls, sigma), oracle.class_blur2d(x, cls, sigma)):
            assert torch.equal(mine, ref)
        for mine, ref in zip(class_blur2d(x, cls, sigma, valid), oracle.class_blur2d(x, cls, sigma, valid)):
            assert torch.equal(mine, ref)
