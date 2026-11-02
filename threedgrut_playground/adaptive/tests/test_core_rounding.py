# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for threedgrut_playground.adaptive.rounding."""

import pytest
import torch

from threedgrut_playground.adaptive.protocols import Rounder
from threedgrut_playground.adaptive.rounding import (
    DeterministicRounder,
    StochasticRounder,
)

g = torch.Generator().manual_seed(0)


def test_deterministic_preserves_rounded_total():
    r = torch.rand(8, 8, generator=g, dtype=torch.float64)
    r *= 100.0 / r.sum()
    n = DeterministicRounder().round(r, 0)
    assert n.dtype == torch.int64 and n.sum().item() == 100 and ((n - r).abs() < 1).all()


def test_deterministic_small_budget_goes_to_largest_fractions():
    r = torch.full((2, 5), 0.3, dtype=torch.float64)
    r[1, 4] = 0.35
    n = DeterministicRounder().round(r, 0)
    assert n.sum().item() == 3 and n[1, 4] == 1


def test_deterministic_ties_have_no_spatial_bias_and_vary_by_step():
    r = torch.full((32, 32), 0.5, dtype=torch.float64)
    n0, n1 = DeterministicRounder().round(r, 0), DeterministicRounder().round(r, 1)
    rows = torch.nonzero(n0)[:, 0].double()
    assert n0.sum() == 512 and abs(rows.mean().item() - 15.5) < 2.0 and not torch.equal(n0, n1)
    assert torch.equal(n0, DeterministicRounder().round(r, 0))


def test_stochastic_rounder_is_unbiased():
    r = torch.linspace(0, 3, 64, dtype=torch.float64).reshape(8, 8)
    acc = torch.zeros_like(r)
    for step in range(4000):
        n = StochasticRounder(seed=1234).round(r, step)
        assert ((n == r.floor()) | (n == r.floor() + 1)).all()
        acc += n
    assert (acc / 4000 - r).abs().max() < 0.04 and abs(
        acc.sum().item() / 4000 - r.sum().item()
    ) < 0.005 * r.sum().item()


def test_stochastic_reproducible_per_seed_and_step():
    r = torch.rand(16, 16, generator=g) * 2
    assert torch.equal(StochasticRounder(3).round(r, 5), StochasticRounder(3).round(r, 5))
    assert not torch.equal(StochasticRounder(3).round(r, 5), StochasticRounder(3).round(r, 6))
    assert not torch.equal(StochasticRounder(3).round(r, 5), StochasticRounder(4).round(r, 5))


@pytest.mark.parametrize("rounder", [DeterministicRounder(), StochasticRounder()])
def test_rounders_sanitize_negative_and_nan(rounder):
    n = rounder.round(torch.tensor([[-2.0, float("nan"), float("inf"), 1.0]]), 0)
    assert n.tolist() == [[0, 0, 0, 1]]


def test_rounders_conform_to_protocol():
    assert isinstance(DeterministicRounder(), Rounder) and isinstance(StochasticRounder(), Rounder)
