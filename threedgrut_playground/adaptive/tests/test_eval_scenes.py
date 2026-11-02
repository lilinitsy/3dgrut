# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests of the glossy scene preset layout (evaluation/scenes.py)."""

import dataclasses

import torch

from threedgrut_playground.adaptive.evaluation.scenes import (
    PRESETS,
    SceneFrame,
    _place_sphere,
    assign_hits_to_objects,
    floor_quad_vertices,
    glossy_layout,
    scene_frame,
)
from threedgrut_playground.utils.kaolin_future.transform import ObjectTransform

EXPECTED = {"brushed_copper", "rose_gold", "polished_marble", "blue_plastic", "oak_wood"}
GARDEN = SceneFrame(center=torch.tensor([0.197, 2.144, 1.478]), spread=2.339)


def test_scene_frame_matches_probe_definition():
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 4.0, 9.0], [3.0, 6.0, 12.0], [4.0, 8.0, 15.0]])
    f = scene_frame(pos)
    torch.testing.assert_close(f.center, torch.tensor([2.0, 4.0, 9.0]))
    assert f.spread == 6.0


def test_floor_winding_normal_points_along_normal():
    n = torch.tensor([0.0, -1.0, 0.0])
    v = floor_quad_vertices(torch.tensor([1.0, 2.0, 3.0]), n, torch.tensor([1.0, 0.0, 0.0]), 2.0)
    g = torch.linalg.cross(v[1] - v[0], v[2] - v[0])
    torch.testing.assert_close(g / g.norm(), n)


def test_garden_layout_spheres_rest_on_floor_inside_bounds():
    preset = PRESETS["garden_glossy"]
    lay = glossy_layout(GARDEN, preset)
    n, c0 = lay.floor_normal, lay.floor_vertices.mean(0)
    assert {m for m, _, _ in lay.spheres} | {lay.floor_material} == EXPECTED
    for _, c, r in lay.spheres:
        assert abs(float((c - c0) @ n) - r) < 1e-4
        inplane = (c - c0) - ((c - c0) @ n) * n
        assert inplane.abs().max() + r <= preset.floor_half_extent * GARDEN.spread + 1e-4


def test_cameras_ignore_floor_height_and_mesh_shift():
    preset = PRESETS["garden_glossy"]
    frame = SceneFrame(center=torch.zeros(3), spread=2.0)
    lay = glossy_layout(frame, preset)
    ov = lay.cameras["overview"]
    torch.testing.assert_close(torch.tensor(ov.eye) - torch.tensor(ov.at), torch.tensor([0.0, -2.0, -6.0]))
    moved = glossy_layout(frame, dataclasses.replace(preset, floor_height=-3.0, mesh_shift=(0.0, 1.0, 2.0)))
    assert moved.cameras == lay.cameras
    # floor moves by (Δfloor_height·n + mesh_shift)·spread = ((-3.05)·(0,-1,0) + (0,1,2))·2 = (0, 8.1, 4)
    torch.testing.assert_close(
        moved.floor_vertices - lay.floor_vertices, torch.tensor([[0.0, 8.1, 4.0]] * 4), atol=1e-4, rtol=0
    )
    shift = moved.spheres[0][1] - lay.spheres[0][1] - (moved.floor_vertices.mean(0) - lay.floor_vertices.mean(0))
    torch.testing.assert_close(shift, torch.zeros(3), atol=1e-5, rtol=0)


def test_assign_hits_to_objects():
    lay = glossy_layout(GARDEN, PRESETS["garden_glossy"])
    _, c1, r1 = lay.spheres[0]
    pts = torch.stack([c1 + torch.tensor([r1, 0.0, 0.0]), lay.floor_vertices.mean(0), torch.tensor([99.0, 99.0, 99.0])])
    assert assign_hits_to_objects(pts, lay).tolist() == [1, 0, -1]


def test_sphere_translate_accepts_float64_cpu_center():
    tr = ObjectTransform(device="cpu")
    _place_sphere(tr, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64), 0.25)
    torch.testing.assert_close(tr.model_matrix()[:3, 3], torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(tr.model_matrix()[0, 0], torch.tensor(0.5))
