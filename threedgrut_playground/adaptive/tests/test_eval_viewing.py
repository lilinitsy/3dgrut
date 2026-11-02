# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from kaolin.render.camera import Camera

from threedgrut_playground.adaptive.evaluation.core_api import center_view_dirs
from threedgrut_playground.adaptive.evaluation.viewing import (
    band_masks,
    band_names,
    eccentricity_deg,
    image_center_dir,
    ppd_from_camera,
)


def _cam(w, h, fov_deg=60.0):
    return Camera.from_args(
        eye=torch.tensor([0.0, 0.0, -5.0]),
        at=torch.zeros(3),
        up=torch.tensor([0.0, -1.0, 0.0]),
        fov=math.radians(fov_deg),
        width=w,
        height=h,
    )


def test_ppd_uses_vertical_fov():
    assert ppd_from_camera(_cam(512, 512)) == pytest.approx(512 / 60.0, rel=1e-5)  # 8.53 ppd
    assert ppd_from_camera(_cam(1024, 512)) == pytest.approx(512 / 60.0, rel=1e-5)  # width does not matter
    assert isinstance(ppd_from_camera(_cam(512, 512)), float)


def test_eccentricity_center_edge_and_corner():
    cam = _cam(101, 101)  # row-0 pixel center is 50 of 50.5 px above the axis (checked: 29.7538 deg)
    ecc = eccentricity_deg(center_view_dirs(cam), image_center_dir(cam))
    assert ecc[50, 50].item() < 1e-3
    assert ecc[0, 50].item() == pytest.approx(
        math.degrees(math.atan(math.tan(math.radians(30.0)) * 50.0 / 50.5)), abs=1e-3
    )
    corner = math.degrees(math.atan(math.tan(math.radians(30.0)) * math.sqrt(2.0) * 50.0 / 50.5))
    assert ecc[0, 0].item() == pytest.approx(corner, abs=1e-3)  # ~39 deg


def test_band_masks_partition_nonsquare_image():
    cam = _cam(64, 36)
    masks = band_masks(eccentricity_deg(center_view_dirs(cam), image_center_dir(cam)))
    stack = torch.stack(list(masks.values()))
    assert list(masks) == band_names() and bool((stack.sum(0) == 1).all())
    assert all(m.any() for m in masks.values())


def test_gaze_dir_shifts_zero_point():
    dirs = center_view_dirs(_cam(101, 101))
    ecc = eccentricity_deg(dirs, dirs[10, 20])
    assert ecc[10, 20].item() < 1e-3 and ecc[50, 50].item() > 5.0
