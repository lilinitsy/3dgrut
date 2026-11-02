# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Viewing model (D14), shared by every metric: the viewer sits at the camera's center of projection, so the display
subtends the camera's vertical FOV. FLIP and CVVDP get ``ppd = H / fov_deg`` (the mean pixels per degree); the
eccentricity bands use the true visual angle from the gaze direction."""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Sequence

import torch
from kaolin.render.camera import generate_pinhole_rays

ECC_BAND_EDGES_DEG = (0.0, 5.0, 10.0, 20.0, 30.0, math.inf)


def ppd_from_camera(camera) -> float:
    """Mean pixels per degree: ``camera.height / vertical_fov_deg`` as a Python float."""
    fov_deg = float(torch.as_tensor(camera.fov(in_degrees=True)).reshape(-1)[0])
    return float(camera.height) / fov_deg


def _edge_label(value: float) -> str:
    return "inf" if math.isinf(value) else f"{value:g}"


def band_names(edges: Sequence[float] = ECC_BAND_EDGES_DEG) -> List[str]:
    """``["ecc_0_5", "ecc_5_10", ...]``, one per consecutive pair of ``edges``."""
    return [f"ecc_{_edge_label(lo)}_{_edge_label(hi)}" for lo, hi in itertools.pairwise(edges)]


def image_center_dir(camera) -> torch.Tensor:
    """Unit view direction ``[3]`` through the continuous image center (``H/2``, ``W/2`` in pixel coordinates)."""
    h, w = int(camera.height), int(camera.width)
    device = camera.device
    _, d = generate_pinhole_rays(
        camera,
        coords_grid=[torch.full((1, 1), 0.5 * h, device=device), torch.full((1, 1), 0.5 * w, device=device)],
    )
    return torch.nn.functional.normalize(d.reshape(-1, 3)[0].float(), dim=0)


def eccentricity_deg(view_dirs: torch.Tensor, gaze_dir: torch.Tensor) -> torch.Tensor:
    """Visual angle in degrees ``[H, W]`` between each unit view direction ``[H, W, 3]`` and ``gaze_dir[3]``:
    ``atan2(|v x g|, v . g)``."""
    v = view_dirs.float()
    g = gaze_dir.to(device=v.device, dtype=v.dtype).reshape(3)
    cross = torch.linalg.cross(v, g.expand_as(v), dim=-1)
    return torch.rad2deg(torch.atan2(cross.norm(dim=-1), (v * g).sum(-1)))


def band_masks(ecc_deg: torch.Tensor, edges: Sequence[float] = ECC_BAND_EDGES_DEG) -> Dict[str, torch.Tensor]:
    """Boolean masks ``[H, W]`` keyed by ``band_names(edges)``, with ``lo <= ecc < hi``."""
    return {
        name: (ecc_deg >= lo) & (ecc_deg < hi) for name, (lo, hi) in zip(band_names(edges), itertools.pairwise(edges))
    }
