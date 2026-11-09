# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The equal-time cone vs Monte Carlo comparison: its time matching, its detail measure, and one small end-to-end run."""

import argparse
import csv
import json

import pytest
import torch

from threedgrut_playground.adaptive.evaluation.cone_equal_time import (
    BLURRED,
    NOISY,
    ConeSetting,
    detail_ratio,
    equal_time_spp,
    object_boxes,
    parse_cone_setting,
    verdict,
)
from threedgrut_playground.adaptive.tests.kernel_helpers import requires_gpu_ckpt


def test_equal_time_spp_rounds_to_nearest_frame_count():
    assert equal_time_spp(23.2, 16.57) == 1  # the measured room numbers: a cone frame buys one MC frame
    assert equal_time_spp(33.0, 16.0) == 2
    assert equal_time_spp(250.0, 10.0) == 25


def test_equal_time_spp_subtracts_overhead_and_never_drops_below_one():
    assert equal_time_spp(50.0, 10.0, overhead_ms=5.0) == 5  # (50 - 5) / 10 = 4.5 rounds up
    assert equal_time_spp(5.0, 16.0) == 1
    assert equal_time_spp(10.0, 16.0, overhead_ms=20.0) == 1
    with pytest.raises(ValueError):
        equal_time_spp(10.0, 0.0)


def test_parse_cone_setting():
    assert parse_cone_setting("0.005:0.2") == ConeSetting(0.005, 0.2)
    assert ConeSetting(0.05, 1.0).name == "cone_sigma0.05_rmax1"


def _textured(size=64, seed=0):
    generator = torch.Generator().manual_seed(seed)
    base = torch.nn.functional.avg_pool2d(torch.rand(1, 3, size, size, generator=generator), 5, 1, 2)[0]
    return base.permute(1, 2, 0).contiguous(), torch.ones(size, size, dtype=torch.bool)


def test_detail_ratio_is_one_for_the_reference_itself():
    image, mask = _textured()
    assert detail_ratio(image, image, mask) == pytest.approx(1.0)


def test_detail_ratio_flags_noise_and_blur():
    reference, mask = _textured()
    noisy = reference + 0.2 * torch.randn(reference.shape, generator=torch.Generator().manual_seed(1))
    blurred = torch.nn.functional.avg_pool2d(reference.permute(2, 0, 1)[None], 7, 1, 3)[0].permute(1, 2, 0)
    assert detail_ratio(noisy, reference, mask) > NOISY
    assert detail_ratio(blurred, reference, mask) < BLURRED
    assert verdict("mc", detail_ratio(noisy, reference, mask)) == "noisy"
    assert verdict("cone", detail_ratio(noisy, reference, mask)) == "too sharp"
    assert verdict("cone", detail_ratio(blurred, reference, mask)) == "blurred"


def test_detail_ratio_ignores_the_silhouette():
    """A constant disc on black has no interior detail; the edge must not count."""
    yy, xx = torch.meshgrid(torch.arange(64.0), torch.arange(64.0), indexing="ij")
    mask = (yy - 32) ** 2 + (xx - 32) ** 2 < 20**2
    disc = mask[..., None].float().expand(64, 64, 3) * 0.5
    textured, _ = _textured()
    reference = torch.where(mask[..., None], textured, torch.zeros_like(textured))
    assert detail_ratio(disc, reference, mask) < 0.05


def test_object_boxes_are_square_and_cover_each_sphere():
    ids = torch.full((100, 120), -1)
    ids[10:30, 5:25] = 1  # ladder sphere
    ids[60:95, 90:118] = 2  # ladder sphere near the corner
    ids[40:50, 40:50] = 0  # the floor: no box
    boxes = object_boxes(ids, ["floor", "sphere_1", "sphere_2"])
    assert set(boxes) == {"sphere_1", "sphere_2"}
    for index, name in ((1, "sphere_1"), (2, "sphere_2")):
        y0, y1, x0, x1 = boxes[name]
        assert y1 - y0 == x1 - x0
        assert 0 <= y0 and y1 <= 100 and 0 <= x0 and x1 <= 120
        ys, xs = torch.nonzero(ids == index, as_tuple=True)
        assert y0 <= int(ys.min()) and int(ys.max()) < y1 and x0 <= int(xs.min()) and int(xs.max()) < x1


@requires_gpu_ckpt
def test_run_matches_frame_time_and_writes_the_figures(playground_engine, tmp_path):
    """Small end-to-end run: every equal-time column costs within half an MC frame of one cone frame."""
    from threedgrut_playground.adaptive.evaluation.cone_equal_time import run

    args = argparse.Namespace(ckpt="session", out_dir=str(tmp_path), resolution=128, ref_spp=16, cones="0.005:0.2")
    summary = run(args, engine=playground_engine)
    setting = summary["settings"]["cone_sigma0.005_rmax0.2"]
    mc_ms, denoise_ms, cone_ms = summary["mc_ms"], summary["denoise_ms"], setting["cone_ms"]
    assert setting["mc_equal_time_spp"] == equal_time_spp(cone_ms, mc_ms)
    assert setting["mc_denoised_spp"] == equal_time_spp(cone_ms, mc_ms, denoise_ms)
    if setting["mc_equal_time_spp"] > 1:
        assert abs(setting["mc_equal_time_spp"] * mc_ms - cone_ms) <= 0.5 * mc_ms
    if setting["mc_denoised_spp"] > 1:
        assert abs(setting["mc_denoised_spp"] * mc_ms + denoise_ms - cone_ms) <= 0.5 * mc_ms

    with open(tmp_path / "results.csv") as handle:
        rows = list(csv.DictReader(handle))
    columns = {row["column"] for row in rows}
    assert "MC 1 spp" in columns and "Cone, cap 0.005" in columns
    assert {row["object"] for row in rows} >= {"floor", "sphere_1", "sphere_2", "sphere_3", "sphere_4"}
    for name in ("crops_cone_sigma0.005_rmax0.2.png", "full_cone_sigma0.005_rmax0.2.png", "reference.png"):
        assert (tmp_path / name).stat().st_size > 0
    assert json.loads((tmp_path / "summary.json").read_text())["ref_spp"] == 16
