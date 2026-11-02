# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``AdaptiveRenderer.trace`` against the CPU fake engine of ``core_fakes``.

The fake derives each ray's scene value from its *direction* (projected to continuous pixel coordinates), and its
randomness only from the ids, as the kernel does: ``u = hash01(pid, sidx)``, a 32-bit avalanche hash computed in int64
with ``M = 0xFFFFFFFF`` and masking before each shift::

    h = (pid * 0x9E3779B1 + sidx * 0x85EBCA77) & M
    h ^= h >> 15; h = (h * 0x2C1B3C6D) & M
    h ^= h >> 12; h = (h * 0x297A2D39) & M
    h ^= h >> 15; u = h / 2**32

Ray/pixel mapping errors therefore fail the fake's identity-contract assertions.
"""

import pytest
import torch

from threedgrut_playground.adaptive.rays import RayBatch, rays_for_counts
from threedgrut_playground.adaptive.renderer import AdaptiveRenderer
from threedgrut_playground.adaptive.telemetry import CostTelemetry
from threedgrut_playground.adaptive.tests.core_fakes import (
    _renderer,
    make_camera,
    make_fake_engine,
)


@pytest.fixture
def camera():
    return make_camera(16)


def _all_pixels(cam, k=1, base=0):
    return rays_for_counts(cam, torch.full((256,), k, dtype=torch.int64), torch.full((256,), base, dtype=torch.int64))


def test_fake_catches_wrong_ray_pixel_mapping(camera):
    b = _all_pixels(camera)
    bad = RayBatch(b.origins, b.directions, b.pixel_id.roll(1), b.sample_index)
    with pytest.raises(AssertionError):
        _renderer(make_fake_engine(camera)).trace(bad)
    _renderer(make_fake_engine(camera)).trace(b)  # correct mapping passes


def test_trace_does_not_alias_ray_tensors(camera, monkeypatch):
    monkeypatch.setattr(AdaptiveRenderer, "ROW_WIDTH", 16)  # exact rows: a view-based launch would alias
    b = _all_pixels(camera, k=2)
    before = b.directions.clone()
    _renderer(make_fake_engine(camera)).trace(b)
    assert torch.equal(b.directions, before)


def test_trace_blends_background_without_clamp(camera):
    out = _renderer(make_fake_engine(camera, mesh_level=1.5)).trace(_all_pixels(camera, k=4))
    vis = out["aov_first_hit_transmittance"][:, 0] >= 0.5
    assert out["rgb"][vis].max() > 1.0  # HDR kept
    torch.testing.assert_close(out["rgb"][~vis], torch.full_like(out["rgb"][~vis], 0.4))
    out2 = _renderer(make_fake_engine(camera, ignore_envmap=True)).trace(_all_pixels(camera))
    nonmesh = out2["aov_first_prim_type"][:, 0] == 0
    torch.testing.assert_close(out2["rgb"][nonmesh], torch.full_like(out2["rgb"][nonmesh], 0.3))


def test_trace_outputs_are_flat_and_typed(camera):
    out = _renderer(make_fake_engine(camera)).trace(_all_pixels(camera, k=3))
    assert out["rgb"].shape == (768, 3) and out["opacity"].shape == (768, 1)
    assert out["aov_first_prim_type"].dtype == torch.int32 and out["aov_first_normal"].shape == (768, 3)
    assert {"aov_first_hit_t", "aov_first_hit_transmittance", "aov_num_bounces", "aov_gauss_depth"} <= set(out)


def test_trace_respects_engine_state(camera):
    eng = make_fake_engine(camera)
    r = _renderer(eng)
    r.trace(_all_pixels(camera))
    r.trace(_all_pixels(camera, base=1))
    assert [c["is_sync_materials"] for c in eng.tracer.calls] == [True, False]
    assert all(c["frame_id"] == 7 for c in eng.tracer.calls) and eng.frame_id == 7 and eng.is_materials_dirty is False


def test_trace_chunks_with_minimal_padding(camera, monkeypatch):
    monkeypatch.setattr(AdaptiveRenderer, "MAX_RAYS_PER_LAUNCH", 100)
    eng = make_fake_engine(camera)
    tel = CostTelemetry(use_cuda_events=False)
    _renderer(eng, telemetry=tel).trace(_all_pixels(camera), kind="pilot")
    assert [r.n_rays for r in tel.records()] == [100, 100, 56] and all(r.kind == "pilot" for r in tel.records())
    assert all(
        c["launch_shape"][1] * c["launch_shape"][2] - n < c["launch_shape"][1]
        for c, n in zip(eng.tracer.calls, (100, 100, 56))
    )


@pytest.mark.parametrize("kw", [dict(visible=False), dict(enabled=False)])
def test_trace_requires_enabled_visible_primitives(camera, kw):
    with pytest.raises(ValueError, match="visible mesh primitive"):
        _renderer(make_fake_engine(camera, **kw)).trace(_all_pixels(camera))
