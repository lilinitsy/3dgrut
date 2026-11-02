# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sampling rounds, the plain uniform renderer and the plumbing statistics of ``evaluation/sampling.py``.

The plain-renderer tests use the CPU fake engine of ``core_fakes`` (values keyed by ray direction, randomness only by
the ray ids); the last test repeats the pilot-equivalence check on the real engine.
"""

import math

import pytest
import torch

from threedgrut_playground.adaptive.evaluation import core_api as C
from threedgrut_playground.adaptive.evaluation import sampling as S
from threedgrut_playground.adaptive.evaluation.core_api import MESH_VISIBLE_T
from threedgrut_playground.adaptive.tests import core_fakes


def test_visible_rule_uses_core_threshold():
    out = {
        "aov_first_prim_type": torch.tensor([[0], [4], [4], [1], [4]], dtype=torch.int32),
        "aov_first_hit_transmittance": torch.tensor([[1.0], [0.5], [0.49], [0.9], [0.0]]),
    }
    assert MESH_VISIBLE_T == 0.5 and S.visible_mesh_hits(out).tolist() == [False, True, False, True, False]


def test_union_by_pixel():
    assert S.union_by_pixel(
        torch.tensor([0, 0, 2, 2, 3]), torch.tensor([False, True, False, False, True]), 5
    ).tolist() == [True, False, False, True, False]


@pytest.mark.parametrize("counts,max_rays,n_rounds", [([0, 5, 1, 9, 3], 4, None), ([1] * 10, 3, 4)])
def test_count_rounds_invariants(counts, max_rays, n_rounds):
    c = torch.tensor(counts, dtype=torch.int64)
    rounds = S.count_rounds(c, max_rays)
    assert torch.equal(torch.stack(rounds).sum(0), c) and all(0 < int(r.sum()) <= max_rays for r in rounds)
    assert n_rounds is None or len(rounds) == n_rounds


def test_traced_rounds_advances_base_and_issues_disjoint_ids():
    from kaolin.render.camera import Camera

    cam = Camera.from_args(
        eye=torch.tensor([0.0, 0.0, 5.0]),
        at=torch.zeros(3),
        up=torch.tensor([0.0, 1.0, 0.0]),
        fov=math.radians(45),
        width=4,
        height=2,
        device="cpu",
    )

    class FakeRenderer:
        def trace(self, batch, kind="external", iteration=-1):
            return {"rgb": torch.zeros(batch.num_rays, 3)}

    counts = torch.tensor([3, 0, 1, 2, 0, 0, 5, 1], dtype=torch.int64)
    base = torch.full((8,), 2**30, dtype=torch.int64)
    seen = [
        (b.pixel_id.long(), b.sample_index.long())
        for b, _ in S.traced_rounds(FakeRenderer(), cam, counts, base, kind="reference", max_rays=4)
    ]
    pid = torch.cat([p for p, _ in seen])
    sid = torch.cat([s for _, s in seen])
    assert len(set(zip(pid.tolist(), sid.tolist()))) == int(counts.sum()) and torch.equal(base, 2**30 + counts)


def test_z_statistics_on_gaussian_noise():
    g = torch.Generator().manual_seed(0)
    v = torch.full((20000,), 0.01, dtype=torch.float64)
    d2 = (torch.randn(20000, generator=g, dtype=torch.float64) * 0.1) ** 2
    st = S.z_statistics(d2, v)
    assert st["z_mean"] == pytest.approx(1.0, abs=0.05) and st["z_median"] == pytest.approx(1.0, abs=0.05)


def test_check_plumbing_warns_and_fails():
    ok = {"z_median": 1.0, "z_mean": 1.0, "slope": -1.0, "exact_mismatch": 0.0}
    assert S.check_plumbing(ok) == []
    lo, hi = S.PLUMBING_BOUNDS["z_mean_soft"]
    assert S.check_plumbing({**ok, "z_mean": hi * 1.01})  # soft violation -> warning text
    with pytest.raises(S.PlumbingError, match="z_median"):
        S.check_plumbing({**ok, "z_median": 10.0})
    with pytest.raises(S.PlumbingError, match="exact_mismatch"):
        S.check_plumbing({**ok, "exact_mismatch": 1e-3})


def test_check_plumbing_accepts_slope_only():
    """Controller amendment: only the keys present are checked (Task 8 passes the slope alone)."""
    assert S.check_plumbing({"slope": -1.0}) == []
    soft_lo, _ = S.PLUMBING_BOUNDS["slope_soft"]
    hard_lo, _ = S.PLUMBING_BOUNDS["slope_hard"]
    warnings = S.check_plumbing({"slope": (soft_lo + hard_lo) / 2})  # between the soft and the hard bound
    assert len(warnings) == 1 and "slope" in warnings[0]
    with pytest.raises(S.PlumbingError, match="slope"):
        S.check_plumbing({"slope": hard_lo - 0.1})
    with pytest.raises(S.PlumbingError, match="slope"):
        S.check_plumbing({"slope": float("nan")})


def test_firefly_simulation_stays_inside_soft_bounds():
    sims = S.simulate_firefly_plumbing(
        20, 0, p=1e-3, F=50.0, sigma=0.05, P=2048, n_test=256, n_ref=512, ns=(8, 32, 128)
    )
    for z in sims["z_median"]:
        assert S.PLUMBING_BOUNDS["z_median_soft"][0] <= z <= S.PLUMBING_BOUNDS["z_median_soft"][1]
    for s in sims["slope"]:
        assert S.PLUMBING_BOUNDS["slope_hard"][0] <= s <= S.PLUMBING_BOUNDS["slope_hard"][1]


# ---------------------------------------------------------------------------------------------------------------------
# plain uniform renderer (CPU fake engine)
# ---------------------------------------------------------------------------------------------------------------------


def _capture_pilot(renderer):
    """Wrap ``renderer.trace`` so that every ``pilot*`` launch's ``(pixel_id, sample_index, out)`` is recorded."""
    seen = []
    orig = renderer.trace

    def trace(batch, kind="external", iteration=-1):
        out = orig(batch, kind, iteration)
        if kind.startswith("pilot"):
            seen.append((batch.pixel_id.long(), batch.sample_index.long(), out))
        return out

    renderer.trace = trace
    return seen


def _pipeline(engine, telemetry):
    return C.AdaptiveRenderer(
        engine,
        C.VarianceOfMeanEstimator(),
        C.make_weight_terms("cielab"),
        C.UniformAllocator(),
        C.DeterministicRounder(),
        C.SimpleBudget(64),
        telemetry=telemetry,
    )


def _stats_of(seen, height, width, device):
    st = C.SampleStats(height, width, device, track_halves=False)
    for pid, sidx, out in seen:
        st.add(pid, out["rgb"], sidx)
    return st


def test_plain_uniform_matches_b_pilot_and_counts_rays():
    cam = core_fakes.make_camera(16)
    engine = core_fakes.make_fake_engine(cam)
    tel = C.CostTelemetry(use_cuda_events=False)
    plain = S.plain_uniform_render(S.trace_only_renderer(engine, telemetry=tel), cam, spp=4, sample_offset=2**24)
    assert plain.mean_hdr.shape == (16, 16, 3) and plain.var_mean.shape == (16, 16)
    assert plain.rays_launched == 4 * 256 and plain.n_nonfinite == 0
    assert plain.wall_ms >= plain.gpu_ms > 0 and [r.kind for r in tel.records()] == ["plain_uniform"]

    pipe = _pipeline(core_fakes.make_fake_engine(cam), C.CostTelemetry(use_cuda_events=False))
    seen = _capture_pilot(pipe)
    pipe.render(cam, budget_rays=8 * 256, region="mesh", sample_offset=2**24)
    pilot = _stats_of(seen, 16, 16, "cpu")
    assert torch.equal(pilot.counts(), torch.full((16, 16), 4))
    torch.testing.assert_close(plain.mean_hdr, pilot.mean(), atol=0, rtol=0)
    torch.testing.assert_close(plain.var_mean, pilot.variance_of_mean(), atol=0, rtol=0)


def test_plain_uniform_rounds_do_not_change_the_result(monkeypatch):
    cam = core_fakes.make_camera(16)
    engine = core_fakes.make_fake_engine(cam, nan_pixel=5, nan_sample=103)
    one = S.plain_uniform_render(
        S.trace_only_renderer(engine, telemetry=C.CostTelemetry(use_cuda_events=False)), cam, 8, 100
    )
    monkeypatch.setattr(S, "MAX_RAYS_PER_ROUND", 300)  # 256 pixels -> 1 ray per pixel per round
    tel = C.CostTelemetry(use_cuda_events=False)
    many = S.plain_uniform_render(S.trace_only_renderer(engine, telemetry=tel), cam, 8, 100)
    assert tel.num_records == 8 and many.rays_launched == one.rays_launched == 8 * 256
    assert one.n_nonfinite == many.n_nonfinite == 1
    torch.testing.assert_close(many.mean_hdr, one.mean_hdr, atol=1e-7, rtol=0)
    torch.testing.assert_close(many.var_mean, one.var_mean, atol=1e-9, rtol=0)


def test_plain_uniform_gpu_ms_counts_only_its_own_records():
    cam = core_fakes.make_camera(16)
    tel = C.CostTelemetry(use_cuda_events=False)
    r = S.trace_only_renderer(core_fakes.make_fake_engine(cam), telemetry=tel)
    first = S.plain_uniform_render(r, cam, 2, 0)
    second = S.plain_uniform_render(r, cam, 2, 2)
    assert tel.num_records == 2
    assert second.gpu_ms == pytest.approx(C.telemetry_total_ms(tel, 1)) and first.gpu_ms == pytest.approx(
        C.telemetry_total_ms(tel) - second.gpu_ms
    )


# ---------------------------------------------------------------------------------------------------------------------
# GPU: the plain renderer's first 4 samples are B's pilot on the real engine
# ---------------------------------------------------------------------------------------------------------------------


def _gpu_camera(engine):
    from kaolin.render.camera import Camera

    from threedgrut_playground.engine import OptixPrimitiveTypes

    prims = engine.primitives
    prims.objects.clear()
    prims.add_primitive("Sphere", OptixPrimitiveTypes.PBR, device=engine.device)
    prims.rebuild_bvh_if_needed(force=True)
    verts = prims.stacked_fields.vertices
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


def test_gpu_plain_uniform_first_samples_are_b_pilot(playground_engine):
    e = playground_engine
    cam = _gpu_camera(e)
    P = 64 * 48
    offset = 3 * 2**24
    pipe = _pipeline(e, C.CostTelemetry())
    seen = _capture_pilot(pipe)
    res = pipe.render(cam, budget_rays=8 * P, region="mesh", sample_offset=offset)
    pilot = _stats_of(seen, 48, 64, e.device)

    tel = C.CostTelemetry()
    plain_r = S.trace_only_renderer(e, telemetry=tel)
    plain = S.plain_uniform_render(plain_r, cam, spp=4, sample_offset=offset)
    assert plain.rays_launched == 4 * P and plain.n_nonfinite == 0 and plain.wall_ms >= plain.gpu_ms > 0
    torch.testing.assert_close(plain.mean_hdr, pilot.mean(), atol=1e-6, rtol=0)
    torch.testing.assert_close(plain.var_mean, pilot.variance_of_mean(), atol=1e-8, rtol=0)

    # The pipeline-uniform mesh mask applies to the plain run: same rays, same D6 hits.
    base = torch.full((P,), offset, dtype=torch.int64, device=e.device)
    mask = torch.zeros(P, dtype=torch.bool, device=e.device)
    for batch, out in S.traced_rounds(plain_r, cam, torch.full_like(base, 4), base, kind="plain_uniform"):
        mask |= S.union_by_pixel(batch.pixel_id.long(), S.visible_mesh_hits(out), P)
    assert torch.equal(mask.reshape(48, 64), res.mesh_mask) and mask.any() and (~mask).any()
