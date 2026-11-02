# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU tests of the adaptive core on the real engine: parity with ``_render_playground_hybrid`` and a smoke render.

Every test uses the function-scoped ``playground_engine`` fixture (scene, settings and RNG restored afterwards) and
``monkeypatch`` only, so nothing leaks into the golden tests that run after this file.
"""

import math

import torch
from kaolin.render.camera import Camera

import threedgrut_playground.engine as engine_mod
from threedgrut_playground.adaptive import (
    AdaptiveRenderer,
    ContrastMaskingTerm,
    CostTelemetry,
    DeterministicRounder,
    DisplayClipTerm,
    LuminanceAdaptationTerm,
    MapDumper,
    PerceptualVarianceAllocator,
    SimpleBudget,
    StochasticRounder,
    UniformAllocator,
    VarianceOfMeanEstimator,
    rays_for_counts,
)
from threedgrut_playground.engine import OptixPrimitiveTypes


class _NoClampTorch:
    """Proxy for the ``torch`` module whose ``clamp`` is the identity: exposes the engine's pre-clamp output."""

    def __getattr__(self, name):
        if name == "clamp":
            return lambda x, *args, **kwargs: x
        return getattr(torch, name)


def _single(engine, prim_type):
    """Scene with exactly one sphere of ``prim_type``."""
    prims = engine.primitives
    prims.objects.clear()
    prims.add_primitive("Sphere", prim_type, device=engine.device)
    prims.rebuild_bvh_if_needed(force=True)


def _cam(engine):
    """64x48 pinhole camera (fov 45 deg) three radii in front of the scene's meshes, looking at their center."""
    verts = engine.primitives.stacked_fields.vertices
    c = 0.5 * (verts.min(dim=0).values + verts.max(dim=0).values)
    r = (verts - c).norm(dim=-1).max()
    engine.camera_type = "Pinhole"
    return Camera.from_args(
        eye=c + torch.tensor([0.0, 0.0, 3.0], device=c.device) * r,
        at=c,
        up=torch.tensor([0.0, 1.0, 0.0], device=c.device),
        fov=math.radians(45),
        width=64,
        height=48,
        dtype=torch.float32,
        device=engine.device,
    )


def test_engine_parity_unjittered_1spp_including_hdr(playground_engine, monkeypatch):
    e = playground_engine
    _single(e, OptixPrimitiveTypes.MIRROR)  # mirror: deterministic
    cam = _cam(e)
    P = cam.width * cam.height
    bg = e.scene_mog.background  # bright background -> HDR > 1
    monkeypatch.setattr(bg, "background_color_type", "white")
    monkeypatch.setattr(bg, "color", torch.full((3,), 3.0, device="cuda"))
    grid = e.raygen(cam, use_spp=False)
    b = rays_for_counts(
        cam,
        torch.ones(P, dtype=torch.long, device="cuda"),
        torch.zeros(P, dtype=torch.long, device="cuda"),
        jitter=False,
    )
    torch.testing.assert_close(b.origins, grid.rays_ori.reshape(-1, 3), atol=1e-6, rtol=0)
    torch.testing.assert_close(b.directions, grid.rays_dir.reshape(-1, 3), atol=1e-6, rtol=0)
    with monkeypatch.context() as mp:
        mp.setattr(engine_mod, "torch", _NoClampTorch())  # the engine's own clamp becomes the identity
        ref = e._render_playground_hybrid(grid.rays_ori.clone(), grid.rays_dir.clone())
    out = AdaptiveRenderer(
        e, VarianceOfMeanEstimator(), [], UniformAllocator(), DeterministicRounder(), SimpleBudget(64)
    ).trace(b)
    ref_rgb = ref["pred_features"].reshape(-1, 3)
    assert (ref_rgb > 1).any()  # the comparison covers HDR values
    torch.testing.assert_close(out["rgb"], ref_rgb, atol=1e-5, rtol=0)  # identical before any clamp
    assert torch.equal(out["aov_first_prim_type"].reshape(-1), ref["aov_first_prim_type"].reshape(-1))
    assert (out["aov_first_prim_type"] == int(OptixPrimitiveTypes.MIRROR)).any()


def test_gpu_adaptive_render_smoke(playground_engine, tmp_path):
    e = playground_engine
    _single(e, OptixPrimitiveTypes.PBR)
    cam = _cam(e)
    P = 64 * 48
    tel = CostTelemetry()
    r = AdaptiveRenderer(
        e,
        VarianceOfMeanEstimator(),
        [LuminanceAdaptationTerm(), ContrastMaskingTerm(), DisplayClipTerm()],
        PerceptualVarianceAllocator(),
        StochasticRounder(seed=0),
        SimpleBudget(64),
        telemetry=tel,
        dumper=MapDumper(tmp_path),
    )
    res = r.render(cam, budget_rays=8 * P, region="mesh")
    m = res.mesh_mask
    t = res.timings
    assert res.mean_hdr.shape == (48, 64, 3) and torch.isfinite(res.mean_hdr).all() and m.any() and (~m).any()
    assert res.counts.sum().item() + t["rays_padding"] == t["rays_launched"] <= 8 * P
    assert t["rays_launched"] + t["rays_unspent"] == 8 * P and t["rays_unspent"] <= 16
    assert (res.counts[~m] == 4).all() and res.counts[m].float().mean() > 4
    grid = e.raygen(cam, use_spp=False)
    ref = e._render_playground_hybrid(grid.rays_ori.clone(), grid.rays_dir.clone())["pred_features"][0]
    assert abs(res.mean_hdr[~m].clamp(0, 1).mean().item() - ref[~m].mean().item()) < 0.02  # AA changes little
    assert t["wall_ms"] >= t["gpu_ms"] > 0 and "class:mesh" in tel.ms_per_ray()
    assert (tmp_path / "iter_002" / "rates.png").is_file()


def test_gpu_decoupled_halves_agree_on_average(playground_engine):
    e = playground_engine
    _single(e, OptixPrimitiveTypes.PBR)
    cam = _cam(e)
    P = 64 * 48
    r = AdaptiveRenderer(
        e,
        VarianceOfMeanEstimator(),
        [LuminanceAdaptationTerm()],
        PerceptualVarianceAllocator(gamma=0.5),
        StochasticRounder(seed=1),
        SimpleBudget(64),
    )
    res = r.render(cam, budget_rays=16 * P, decouple_halves=True, sample_offset=2**24)
    m = res.mesh_mask
    assert torch.isfinite(res.mean_hdr_half_b).all()
    assert abs(res.mean_hdr_half_b[m].mean().item() - res.mean_hdr[m].mean().item()) < 0.05
