# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for threedgrut_playground.adaptive.allocate (Neyman / power allocators, D14)."""

import pytest
import torch

from threedgrut_playground.adaptive.allocate import (
    PerceptualVarianceAllocator,
    UniformAllocator,
    VarianceAllocator,
)
from threedgrut_playground.adaptive.protocols import Allocator

ALL = [UniformAllocator(), VarianceAllocator(), PerceptualVarianceAllocator()]
g = torch.Generator().manual_seed(0)


def J(s, w, n):
    return (w * s**2 / n).sum()


def _synthetic():
    s = torch.linspace(0.05, 2.0, 64, dtype=torch.float64)
    w = torch.logspace(-2, 1, 64, dtype=torch.float64)
    w = w[torch.randperm(64, generator=g)]
    s, w = s.reshape(8, 8), w.reshape(8, 8)
    c = torch.full((8, 8), 2)
    return s, w, c, s**2 / 2


@pytest.mark.parametrize("alloc", ALL)
def test_rates_sum_to_budget_and_are_nonnegative(alloc):
    r = alloc.rates(
        torch.rand(8, 8, generator=g), torch.rand(8, 8, generator=g) + 0.1, None, torch.full((8, 8), 2), 1000
    )
    assert r.dtype == torch.float64 and r.min() >= 0 and abs(r.sum().item() - 1000) < 1e-9


def test_neyman_optimality_beats_uniform_and_sigma_only():
    s, w, c, err = _synthetic()
    n = {type(a).__name__: c + a.rates(err.float(), w.float(), None, c, 2000) for a in ALL}
    assert J(s, w, n["PerceptualVarianceAllocator"]) < J(s, w, n["UniformAllocator"])
    assert J(s, w, n["PerceptualVarianceAllocator"]) <= J(s, w, n["VarianceAllocator"]) * (1 + 1e-6)


def test_kkt_totals_proportional_to_sigma_sqrt_w():
    s, w, c, err = _synthetic()
    x = PerceptualVarianceAllocator().rates(err.float(), w.float(), None, c, 2000)
    act = x > 0
    ratio = (c + x)[act] / (s * w.sqrt())[act]
    assert (ratio.max() - ratio.min()) / ratio.mean() < 1e-4


def test_oversampled_pixel_gets_zero_not_negative():
    c = torch.full((4, 4), 2)
    c[0, 0] = 10_000
    r = VarianceAllocator().rates(torch.ones(4, 4) / 2, torch.ones(4, 4), None, c, 100)
    assert r[0, 0] == 0 and abs(r.sum().item() - 100) < 1e-9


@pytest.mark.parametrize("alloc", ALL[1:])
def test_zero_variance_everywhere_falls_back_to_uniform(alloc):
    r = alloc.rates(torch.zeros(4, 4), torch.ones(4, 4), None, torch.full((4, 4), 2), 32)
    torch.testing.assert_close(r, torch.full((4, 4), 2.0, dtype=torch.float64))


@pytest.mark.parametrize("alloc", ALL)
def test_weight_zero_pixels_are_ineligible(alloc):
    w = torch.ones(4, 4)
    w[:, :2] = 0
    r = alloc.rates(torch.rand(4, 4, generator=g), w, None, torch.full((4, 4), 2), 50)
    assert (r[:, :2] == 0).all() and abs(r.sum().item() - 50) < 1e-9


def test_nan_and_inf_inputs_are_sanitized():
    err = torch.ones(3, 3)
    err[0, 0] = float("nan")
    err[1, 1] = float("inf")
    w = torch.ones(3, 3)
    w[2, 2] = float("nan")
    r = VarianceAllocator().rates(err, w, None, torch.full((3, 3), 2), 90)
    assert torch.isfinite(r).all() and abs(r.sum().item() - 90) < 1e-9 and r[0, 0] == 0 and r[2, 2] == 0


def test_nonpositive_budget_or_no_eligible_pixels_returns_zeros():
    for budget, w in ((0, torch.ones(2, 2)), (-5, torch.ones(2, 2)), (10, torch.zeros(2, 2))):
        assert torch.equal(
            UniformAllocator().rates(torch.ones(2, 2), w, None, torch.ones(2, 2), budget),
            torch.zeros(2, 2, dtype=torch.float64),
        )


def test_uniform_equalizes_unequal_counts():
    r = UniformAllocator().rates(torch.ones(1, 2), torch.ones(1, 2), None, torch.tensor([[1, 3]]), 2)
    torch.testing.assert_close(r, torch.tensor([[2.0, 0.0]], dtype=torch.float64))


def test_cost_divides_priority_by_sqrt_cost():
    cost = torch.tensor([[1.0, 4.0]])
    r = VarianceAllocator().rates(torch.ones(1, 2), torch.ones(1, 2), cost, torch.zeros(1, 2, dtype=torch.long), 30)
    torch.testing.assert_close(r, torch.tensor([[20.0, 10.0]], dtype=torch.float64))
    with pytest.raises(ValueError, match="cost"):
        VarianceAllocator().rates(torch.ones(1, 2), torch.ones(1, 2), torch.zeros(1, 2), torch.ones(1, 2), 30)


def test_allocators_conform_to_protocol():
    assert all(isinstance(a, Allocator) for a in ALL)


def test_gamma_one_is_neyman_and_half_compresses_spread():
    s = torch.linspace(0.1, 1.6, 64, dtype=torch.float64).reshape(8, 8)  # sigma spread 16x
    c0 = torch.zeros(8, 8, dtype=torch.long)
    err = s**2  # error*max(c,1) = s^2
    n1 = PerceptualVarianceAllocator(gamma=1.0).rates(err, torch.ones(8, 8), None, c0, 6400)
    torch.testing.assert_close(n1, PerceptualVarianceAllocator().rates(err, torch.ones(8, 8), None, c0, 6400))
    torch.testing.assert_close(n1 / n1.sum(), s / s.sum())  # n ~ sigma (Neyman)
    nh = PerceptualVarianceAllocator(gamma=0.5).rates(err, torch.ones(8, 8), None, c0, 6400)
    torch.testing.assert_close(nh / nh.sum(), s.sqrt() / s.sqrt().sum())  # n ~ sigma^0.5
    assert (nh.max() / nh.min()).item() == pytest.approx(4.0) and (n1.max() / n1.min()).item() == pytest.approx(16.0)
    nv = VarianceAllocator(gamma=0.5).rates(err, torch.full((8, 8), 7.0), None, c0, 6400)
    torch.testing.assert_close(nv, nh)  # weight ignored by Variance


def test_gamma_zero_is_uniform_and_negative_gamma_rejected():
    r = PerceptualVarianceAllocator(gamma=0.0).rates(
        torch.rand(4, 4, generator=g),
        torch.rand(4, 4, generator=g) + 0.1,
        None,
        torch.zeros(4, 4, dtype=torch.long),
        32,
    )
    torch.testing.assert_close(r, torch.full((4, 4), 2.0, dtype=torch.float64))
    with pytest.raises(ValueError, match="gamma"):
        VarianceAllocator(gamma=-1.0)
