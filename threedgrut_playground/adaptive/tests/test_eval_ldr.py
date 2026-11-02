# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path

import pytest
import torch

from threedgrut_playground.adaptive.evaluation.ldr import (
    count_nonfinite,
    ensure_outside_repo,
    save_png,
    to_display_ldr,
)

REPO = Path(__file__).resolve().parents[3]


def test_display_ldr_is_clamp_with_finite_output():
    hdr = torch.tensor([[[-0.5, 0.25, 2.0], [math.nan, math.inf, -math.inf]]])
    torch.testing.assert_close(to_display_ldr(hdr), torch.tensor([[[0.0, 0.25, 1.0], [0.0, 1.0, 0.0]]]))
    assert count_nonfinite(hdr) == 1


def test_save_png_roundtrip(tmp_path):
    from PIL import Image

    save_png(torch.full((4, 5, 3), 0.5), tmp_path / "a.png")
    img = Image.open(tmp_path / "a.png")
    assert img.size == (5, 4) and img.getpixel((0, 0)) == (128, 128, 128)


def test_ensure_outside_repo(tmp_path):
    assert ensure_outside_repo(tmp_path / "x") == (tmp_path / "x").resolve()
    for inside in (REPO, REPO / "runs" / "sweep"):
        with pytest.raises(ValueError, match="inside the repository"):
            ensure_outside_repo(inside)
