# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for threedgrut_playground.adaptive.budget."""

import pytest
import torch

from threedgrut_playground.adaptive.budget import SimpleBudget
from threedgrut_playground.adaptive.protocols import BudgetEnforcer


def test_never_exceeds_budget_or_cap():
    g = torch.Generator().manual_seed(0)
    for _ in range(200):
        counts = torch.randint(-3, 40, (6, 7), generator=g)
        budget = int(torch.randint(0, 600, (1,), generator=g))
        cap = int(torch.randint(1, 30, (1,), generator=g))
        out = SimpleBudget(cap).enforce(counts, budget)
        clamped = counts.clamp(0, cap)
        assert out.dtype == torch.int64 and out.max() <= cap and (out <= clamped).all() and (out >= 0).all()
        assert out.sum().item() == min(budget, clamped.sum().item())


def test_trims_largest_first():
    assert SimpleBudget(100).enforce(torch.tensor([[10, 1, 1]]), 6).tolist() == [[4, 1, 1]]
    out = SimpleBudget(100).enforce(torch.tensor([[5, 5, 1]]), 6)
    assert out.sum() == 6 and out[0, 2] == 1 and sorted(out[0, :2].tolist()) == [2, 3]


def test_tie_breaking_is_seeded_not_flat_order():
    counts = torch.full((32, 32), 2)
    out = SimpleBudget(100).enforce(counts, 32 * 32 + 512)  # T = 1, 512 ties get +1
    rows = torch.nonzero(out == 2)[:, 0].double()
    assert abs(rows.mean().item() - 15.5) < 2.0


def test_nonpositive_budget_and_invalid_cap():
    assert SimpleBudget(4).enforce(torch.tensor([[3, 2]]), 0).tolist() == [[0, 0]]
    with pytest.raises(ValueError):
        SimpleBudget(0)


def test_conforms_to_protocol():
    assert isinstance(SimpleBudget(8), BudgetEnforcer)
