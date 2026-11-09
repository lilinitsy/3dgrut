# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the reflection-rate headroom test."""

import torch

from threedgrut_playground.adaptive.evaluation import reflection_rate as R


def test_stratified_keep_has_one_pixel_per_cell():
    keep = R.stratified_keep(64, 64, 8, seed=3)
    assert int(keep.sum()) == 64
    assert (keep.reshape(8, 8, 8, 8).permute(0, 2, 1, 3).sum(dim=(2, 3)) == 1).all()
    assert R.stratified_keep(16, 16, 1, seed=0).all()


def test_reconstruct_preserves_constant_and_never_crosses_objects():
    height = width = 64
    values = torch.zeros(height, width, 3)
    values[:, :32] = 1.0
    values[:, 32:] = 3.0
    ids = torch.zeros(height, width, dtype=torch.int64)
    ids[:, 32:] = 1
    ids[0, 0] = -1
    values[0, 0] = 7.0
    out = R.reconstruct(values, R.stratified_keep(height, width, 8, seed=1), ids, 8)
    assert torch.allclose(out[:, 1:32], torch.ones_like(out[:, 1:32]), atol=1e-5)
    assert torch.allclose(out[:, 32:], torch.full_like(out[:, 32:], 3.0), atol=1e-5)
    assert (out[0, 0] == 7.0).all()  # off every object: untouched


def test_reconstruct_is_identity_at_full_rate():
    values = torch.rand(16, 16, 3, generator=torch.Generator().manual_seed(0))
    ids = torch.zeros(16, 16, dtype=torch.int64)
    assert torch.equal(R.reconstruct(values, R.stratified_keep(16, 16, 1, seed=0), ids, 1), values)


def test_safe_rate_stops_at_the_first_failure():
    yardstick = torch.tensor([0.1, 0.1, 0.1])
    flip = {1: torch.tensor([0.0, 0.0, 0.0]), 2: torch.tensor([0.05, 0.2, 0.05]), 4: torch.tensor([0.09, 0.05, 0.2])}
    assert R.safe_rate(flip, yardstick).tolist() == [4.0, 1.0, 2.0]


def test_decide_needs_both_spread_and_a_predictor():
    varied = torch.tensor([1.0] * 5 + [8.0] * 5)
    assert R.decide(varied, {"roughness": 0.8})["worth_building"]
    assert not R.decide(varied, {"roughness": 0.1})["worth_building"]
    assert not R.decide(torch.full((10,), 4.0), {"roughness": 0.9})["worth_building"]
    assert not R.decide(varied, {"roughness": float("nan")})["worth_building"]
