# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cone test scene: a row of silver spheres of increasing roughness placed inside the Gaussians."""

import pytest
import torch

from threedgrut_playground.adaptive.tests.kernel_helpers import SMOOTH, grid_ids, requires_gpu_ckpt, trace_playground

CONE_TRACING = 8  # OptixPlaygroundRenderOptions.CONE_TRACING
SIZE = 256
MESH_VISIBLE_T = 0.5


_CHOSEN = {}


@pytest.fixture
def ramp(playground_engine):
    from threedgrut_playground.adaptive.evaluation.cone_scene import (
        RAMP_ROUGHNESS,
        apply_plan,
        choose_cone_ramp,
    )

    engine = playground_engine
    if "plan" not in _CHOSEN:  # the ranking renders hundreds of views; it is the same for every test
        _CHOSEN["plan"] = choose_cone_ramp(engine)[0]
    plan = _CHOSEN["plan"]
    assert len(plan.centers) == len(RAMP_ROUGHNESS)
    apply_plan(engine, plan)
    return engine, plan


def _opaque(engine):
    from threedgrut_playground.adaptive.evaluation.cone_scene import OPACITY_MIN

    mog = engine.scene_mog
    return mog.positions.detach().float()[mog.get_density().detach().reshape(-1) > OPACITY_MIN]


def _rays(plan):
    from threedgrut_playground.adaptive.oracle import oracle_rays

    camera = plan.camera.to_camera(SIZE, SIZE)
    pixel_id = torch.arange(SIZE * SIZE, device="cuda", dtype=torch.int32)
    o, d = oracle_rays(camera, pixel_id, torch.zeros_like(pixel_id), jitter=False)
    return o.reshape(1, SIZE, SIZE, 3).contiguous(), d.reshape(1, SIZE, SIZE, 3).contiguous()


def _sphere_pixels(plan, rays_o, rays_d):
    """``[n, H, W]`` bool: the ray hits sphere ``k`` first (analytic, among the spheres)."""
    from threedgrut_playground.adaptive.evaluation.cone_scene import sphere_ownership

    return sphere_ownership(rays_o[0], rays_d[0], plan.centers, plan.radius)


def test_ramp_spheres_not_buried(ramp):
    from threedgrut_playground.adaptive.evaluation.cone_scene import BURIED_COUNT, BURY_RADIUS

    engine, plan = ramp
    opaque = _opaque(engine)
    for center in plan.centers:
        near = ((opaque - center.to(opaque)).square().sum(dim=1) <= (BURY_RADIUS * plan.radius) ** 2).sum()
        assert int(near) < BURIED_COUNT


@requires_gpu_ckpt
def test_ramp_spheres_visible(ramp):
    from threedgrut_playground.engine import OptixPrimitiveTypes

    engine, plan = ramp
    rays_o, rays_d = _rays(plan)
    out = trace_playground(engine, rays_o, rays_d, opts=SMOOTH)
    visible = (out["aov_first_prim_type"][0, :, :, 0] == int(OptixPrimitiveTypes.PBR)) & (
        out["aov_first_hit_transmittance"][0, :, :, 0] >= MESH_VISIBLE_T
    )
    covered = _sphere_pixels(plan, rays_o, rays_d) & visible
    fractions = covered.flatten(1).float().mean(dim=1)
    assert (fractions >= 0.01).all(), fractions.tolist()


@requires_gpu_ckpt
def test_ramp_scores_above_median_location(ramp):
    from threedgrut_playground.adaptive.evaluation.cone_scene import (
        BURIED_COUNT,
        BURY_RADIUS,
        SPACING,
        _row_centers,
        row_score,
    )
    from threedgrut_playground.adaptive.evaluation.scenes import _floor_frame
    from threedgrut_playground.adaptive.evaluation import cone_scene

    engine, plan = ramp
    opaque = _opaque(engine)
    median = engine.scene_mog.positions.detach().float().median(dim=0).values
    _, e1, _ = _floor_frame(torch.tensor(cone_scene.UP))
    spheres = _row_centers(median, e1.to(median), len(plan.centers), SPACING * plan.radius)
    buried = max(int(((opaque - c).square().sum(dim=1) <= (BURY_RADIUS * plan.radius) ** 2).sum()) for c in spheres)
    median_score = row_score(opaque, median, plan.radius)
    print(f"chosen score {plan.score}, median-location score {median_score}, median buried count {buried}")
    assert plan.score > 0
    if buried < BURIED_COUNT:  # a buried median is not a legal placement, so it is not a baseline
        assert plan.score >= median_score


@requires_gpu_ckpt
def test_ramp_selection_maximises_score_times_non_env(ramp):
    from threedgrut_playground.adaptive.evaluation.cone_scene import env_fraction, rank_cone_ramp_candidates

    engine, plan = ramp
    ranked = rank_cone_ramp_candidates(engine)  # all legal pairs, unpruned
    assert ranked
    chosen = [
        c for c in ranked if torch.equal(c.center.cpu(), plan.row_center) and tuple(c.eye.tolist()) == plan.camera.eye
    ]
    assert len(chosen) == 1
    chosen = chosen[0]
    value = chosen.score * (1.0 - chosen.env_fraction)
    assert all(value >= c.score * (1.0 - c.env_fraction) - 1e-6 for c in ranked)
    envs = torch.tensor([c.env_fraction for c in ranked])
    print(f"chosen env fraction {chosen.env_fraction:.3f}, median over {len(ranked)} pairs {float(envs.median()):.3f}")
    assert chosen.env_fraction <= float(envs.median())
    # The same ratio at the test resolution, for the log (the absolute bound is checked on the room checkpoint).
    from threedgrut_playground.adaptive.evaluation.cone_scene import apply_plan

    apply_plan(engine, plan)
    engine.tracer.build_gs_acc(gaussians=engine.scene_mog, rebuild=True, inflate=engine.cone_max_sigma)
    try:
        print(f"chosen env fraction at {SIZE}x{SIZE}: {env_fraction(engine, plan, SIZE):.3f}")
    finally:
        engine.tracer.build_gs_acc(gaussians=engine.scene_mog, rebuild=True, inflate=engine._gs_bvh_inflate)


@requires_gpu_ckpt
def test_place_cone_ramp_returns_chosen_camera(ramp):
    import time

    from threedgrut_playground.adaptive.evaluation.cone_scene import place_cone_ramp

    engine, plan = ramp
    start = time.perf_counter()
    camera = place_cone_ramp(engine)
    print(f"place_cone_ramp wall time {time.perf_counter() - start:.1f} s")
    assert camera == plan.camera
    assert len(engine.primitives.objects) == len(plan.centers)
