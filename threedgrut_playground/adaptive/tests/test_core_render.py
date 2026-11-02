# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``AdaptiveRenderer.render`` (pilot, mesh mask, adaptive loop, bias protection, redistribution) on the CPU fake.

The fake engine of ``core_fakes`` derives each ray's value from its direction and its randomness only from the ray
ids, so the tests below see the same sample values for the same ``(pixel_id, sample_index)`` regardless of launch
order. ``inner`` / ``outer`` are the pixels whose *center* lies within ``r - 0.72`` / ``r + 0.72`` of the image
center ``(8, 8)`` for the fake's default ``mesh_radius = 5`` (0.72 > half a pixel diagonal), so ``inner`` pixels are
fully inside the disk and pixels outside ``outer`` are fully outside it.
"""

import json
import logging

import pytest
import torch

import threedgrut_playground.adaptive.rays as rays_mod
import threedgrut_playground.adaptive.renderer as renderer_mod
from threedgrut_playground.adaptive.allocate import UniformAllocator, VarianceAllocator
from threedgrut_playground.adaptive.budget import SimpleBudget
from threedgrut_playground.adaptive.dump import MapDumper
from threedgrut_playground.adaptive.estimators import VarianceOfMeanEstimator
from threedgrut_playground.adaptive.renderer import MAP_KEYS, AdaptiveRenderer
from threedgrut_playground.adaptive.rounding import DeterministicRounder
from threedgrut_playground.adaptive.telemetry import CostTelemetry
from threedgrut_playground.adaptive.tests.core_fakes import (
    _renderer,
    make_camera,
    make_fake_engine,
    parity_coupled_jitter,
)

_MARGIN = 0.72


def _disk(r: float, size: int = 16) -> torch.Tensor:
    """Pixels ``[size, size]`` whose center lies within ``r`` of the image center."""
    c = torch.arange(size, dtype=torch.float64) + 0.5 - size / 2.0
    return torch.hypot(c[None, :], c[:, None]) < r


@pytest.fixture
def camera():
    return make_camera(16)


@pytest.fixture
def camera32():
    return make_camera(32)


@pytest.fixture
def inner():
    return _disk(5.0 - _MARGIN)


@pytest.fixture
def outer():
    return _disk(5.0 + _MARGIN)


@pytest.fixture
def annulus(outer):
    """Pixels fully inside the hidden band ``5 <= r < 7`` of ``make_fake_engine(..., hidden_radius=7.0)``."""
    return ~outer & _disk(7.0 - _MARGIN)


def test_budget_accounting_is_exact(camera):
    r = _renderer(
        make_fake_engine(camera),
        allocator=UniformAllocator(),
        rounder=DeterministicRounder(),
        budget=SimpleBudget(1000),
    )
    res = r.render(camera, budget_rays=16 * 256, pilot_spp=4, iterations=3, region="all", defensive_fraction=0.0)
    assert res.counts.sum().item() == 4096 == res.timings["rays_launched"] and res.timings["rays_unspent"] == 0
    assert (res.counts == 16).all() and res.timings["rays_padding"] == 0  # launches <= 4096 rays: no padding
    res2 = r.render(camera, budget_rays=16 * 256, pilot_spp=4, iterations=3, region="all")  # defensive 0.15
    assert res2.timings["rays_launched"] == 4096 and res2.counts.max() - res2.counts.min() <= 1


def test_visible_mesh_mask_is_union_over_pilot_and_excludes_hidden(camera, inner, outer, annulus):
    res = _renderer(make_fake_engine(camera, hidden_radius=7.0)).render(camera, 4096, iterations=1)
    assert (res.mesh_mask | ~inner).all() and not (res.mesh_mask & ~outer).any()
    assert not res.mesh_mask[annulus].any() and (res.aovs["first_prim_type"][annulus] == 4).all()


def test_mesh_pixels_get_more_rays_under_region_mesh(camera, inner):
    res = _renderer(make_fake_engine(camera)).render(camera, budget_rays=4096)
    m = res.mesh_mask
    assert (res.counts[~m] == 4).all() and res.counts[inner].float().mean() > 20 and res.counts.sum().item() == 4096


def test_render_is_always_jittered_and_stratified(camera):
    eng = make_fake_engine(camera)
    _renderer(eng, allocator=UniformAllocator(), rounder=DeterministicRounder()).render(
        camera, budget_rays=8192, region="all", defensive_fraction=0.0
    )  # exactly 32 rays per pixel
    for h in (0, 1):  # each half: 16 samples, 4x4-stratified
        pos = [(x, y) for p, s, x, y in eng.tracer.samples if p == 136 and s < 32 and s % 2 == h]
        cells = sorted(int((x - 8) * 4) * 4 + int((y - 8) * 4) for x, y in pos)
        assert cells == list(range(16))


def test_sample_indices_never_repeat(camera, monkeypatch):
    seen = []
    real = renderer_mod.rays_for_counts
    monkeypatch.setattr(renderer_mod, "rays_for_counts", lambda *a, **k: seen.append(real(*a, **k)) or seen[-1])
    res = _renderer(make_fake_engine(camera)).render(camera, 4096, iterations=4)
    key = torch.cat([b.pixel_id.long() * 2**32 + b.sample_index.long() for b in seen])
    assert key.unique().numel() == key.numel() == res.timings["rays_launched"] - res.timings["rays_padding"]


def test_sample_offset_repeats_and_disjointness(camera, monkeypatch):
    r = _renderer(make_fake_engine(camera))
    a, b = r.render(camera, 4096, sample_offset=0), r.render(camera, 4096, sample_offset=0)
    assert torch.equal(a.counts, b.counts) and torch.equal(a.mean_hdr, b.mean_hdr)
    eng = make_fake_engine(camera)
    c = _renderer(eng).render(camera, 4096, sample_offset=2**24)
    sidx = torch.tensor([s for _, s, _, _ in eng.tracer.samples])
    assert sidx.min() >= 2**24 and not torch.equal(a.mean_hdr, c.mean_hdr)
    with pytest.raises(ValueError, match="sample_index"):
        r.render(camera, 4096, sample_offset=2**31 - 2)


def test_shared_telemetry_accumulates_but_timings_are_per_render(camera):
    tel = CostTelemetry(use_cuda_events=False)
    r = _renderer(make_fake_engine(camera), telemetry=tel)
    r.render(camera, 4096)
    n1 = tel.num_records
    b = r.render(camera, 4096)
    assert tel.num_records == 2 * n1 and b.timings["n_launches"] == n1
    assert b.timings["gpu_ms"] == pytest.approx(tel.total_ms(n1)) and b.timings["wall_ms"] >= b.timings["gpu_ms"]
    kinds = {rec.kind for rec in tel.records()}
    assert {"pilot", "pilot_mesh", "pilot_nonmesh", "adaptive_mesh"} <= kinds and "class:mesh" in tel.ms_per_ray()


def test_decoupled_half_b_is_unbiased_where_coupled_mean_is_not(camera32):
    kw = dict(
        allocator=VarianceAllocator(),
        rounder=DeterministicRounder(),
        budget=SimpleBudget(256),
        terms=[],
        estimator=VarianceOfMeanEstimator(sigma_px=0.0),
    )
    args = dict(budget_rays=16 * 1024, pilot_spp=4, iterations=2, region="all", defensive_fraction=0.0)
    coupled = _renderer(make_fake_engine(camera32, scene="bernoulli"), **kw).render(camera32, **args)
    dec = _renderer(make_fake_engine(camera32, scene="bernoulli"), **kw).render(camera32, decouple_halves=True, **args)
    assert coupled.mean_hdr.mean().item() < 0.8  # true mean 1.0; starved pixels drag it down
    assert abs(dec.mean_hdr_half_b.mean().item() - 1.0) < 0.25  # ~4 sigma of the half-B average
    assert coupled.mean_hdr_half_b is None


def test_half_b_is_unbiased_on_subpixel_edge_and_parity_jitter_is_not(camera32, monkeypatch):
    kw = dict(
        allocator=VarianceAllocator(),
        rounder=DeterministicRounder(),
        budget=SimpleBudget(256),
        terms=[],
        estimator=VarianceOfMeanEstimator(sigma_px=0.0),
    )
    args = dict(
        budget_rays=16 * 1024, pilot_spp=4, iterations=2, region="all", defensive_fraction=0.0, decouple_halves=True
    )
    res = _renderer(make_fake_engine(camera32, scene="edge"), **kw).render(camera32, **args)
    assert abs(res.mean_hdr_half_b.mean().item() - 0.3) < 0.03  # D12 sampler: halves independent
    monkeypatch.setattr(rays_mod, "pixel_jitter", parity_coupled_jitter)  # pre-D12: one Sobol sequence split by parity
    bad = _renderer(make_fake_engine(camera32, scene="edge"), **kw).render(camera32, **args)
    assert abs(bad.mean_hdr_half_b.mean().item() - 0.3) > 0.04  # CPU toy simulation: +0.086 (new: +0.007)


def test_padding_rays_count_as_launched(camera, monkeypatch):
    monkeypatch.setattr(AdaptiveRenderer, "ROW_WIDTH", 100)  # forces padded multi-row launches
    res = _renderer(make_fake_engine(camera)).render(camera, 4096)
    t = res.timings
    assert t["rays_padding"] > 0 and res.counts.sum().item() + t["rays_padding"] == t["rays_launched"] <= 4096
    assert t["rays_launched"] + t["rays_unspent"] == 4096 and t["rays_unspent"] < 16


def test_clipped_mesh_gets_no_weighted_rays_but_budget_is_spent(camera, inner):
    res = _renderer(make_fake_engine(camera, mesh_level=2.0)).render(camera, 4096)  # samples in [1.5, 2.5)
    assert res.mean_hdr[inner].mean() > 1.2 and (res.maps[0]["weight"][inner] == 0).all()
    assert (res.maps[0]["rates"][inner] == 0).all()  # no defensive share for clipped pixels
    assert torch.isfinite(res.mean_hdr).all() and res.timings["rays_unspent"] == 0


def test_capped_topup_widens_from_unclipped_to_clipped_region_pixels(camera, inner, caplog):
    # Only a few partially covered edge pixels are unclipped, and SimpleBudget(64) caps each launch per pixel, so
    # top-up round 1 (step 3) saturates them and round 2 (step 4) widens to all eligible (mesh) pixels. Widening
    # stops at the region: rays never leave it (only an empty region falls back to all pixels).
    tel = CostTelemetry(use_cuda_events=False)
    with caplog.at_level(logging.WARNING, logger=renderer_mod.__name__):
        res = _renderer(make_fake_engine(camera, mesh_level=2.0), telemetry=tel).render(camera, 4096)
    assert all((m["rates"][inner] == 0).all() and (m["counts_new"][inner] == 0).all() for m in res.maps)
    assert (res.counts[inner] > 4).all()  # only the widened top-up round reaches the clipped pixels
    assert (res.counts[~res.mesh_mask] == 4).all()  # widening stops at the region
    assert {rec.iteration for rec in tel.records() if rec.kind.startswith("topup")} == {3, 4}
    assert res.timings["rays_unspent"] == 0 and "widening the next round to all eligible pixels" in caplog.text


def test_cap_saturated_region_never_spills_topup_outside_region(camera, caplog):
    # Small unclipped mesh (eligible & unclipped == eligible) under a 64-ray per-launch cap: the region cannot absorb
    # a 64-spp budget, and the residue must stay unspent rather than go to non-mesh pixels.
    with caplog.at_level(logging.WARNING, logger=renderer_mod.__name__):
        res = _renderer(make_fake_engine(camera, mesh_radius=2.0), budget=SimpleBudget(64)).render(camera, 64 * 256)
    m, t = res.mesh_mask, res.timings
    assert m.any() and (res.maps[-1]["weight_display_clip"][m] > 0).all()  # no clipped region pixel
    assert (res.counts[~m] == 4).all() and t["rays_unspent"] > 0  # pilot_spp only outside the region
    assert res.counts.sum().item() + t["rays_padding"] == t["rays_launched"] <= 64 * 256
    assert t["rays_launched"] + t["rays_unspent"] == 64 * 256 and "widening" not in caplog.text


def test_empty_region_budget_goes_to_topup(camera):
    res = _renderer(make_fake_engine(camera, mesh_radius=0.0)).render(camera, 4096)
    assert not res.mesh_mask.any() and res.timings["rays_unspent"] == 0
    assert res.timings["topup_rays"] == 4096 - 1024 and all(m["rates"].sum() == 0 for m in res.maps)


def test_nonfinite_sample_counts_as_launched(camera):
    res = _renderer(make_fake_engine(camera, nan_pixel=136)).render(camera, 4096)
    assert res.timings["n_nonfinite"] == 1 and torch.isfinite(res.mean_hdr).all()
    assert res.stats.counts().flatten()[136] == res.counts.flatten()[136] - 1 and res.timings["rays_launched"] == 4096


def test_small_iteration_budget_is_spent_exactly(camera):
    res = _renderer(make_fake_engine(camera), rounder=DeterministicRounder()).render(camera, 1024 + 10, iterations=2)
    assert res.timings["rays_launched"] == 1034 and res.timings["rays_unspent"] == 0  # min budget is 1024 + 6


def test_maps_and_aovs(camera, inner):
    res = _renderer(make_fake_engine(camera)).render(camera, 4096, iterations=3)
    assert len(res.maps) == 3 and res.maps[0]["rates"].device.type == "cpu"
    terms = {"weight_luminance_adaptation", "weight_contrast_masking", "weight_display_clip"}
    assert set(MAP_KEYS) | terms <= set(res.maps[0])
    assert (res.aovs["first_prim_type"][inner] == 4).all() and (res.aovs["first_hit_t"][~res.mesh_mask] == -1).all()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(budget_rays=4096, pilot_spp=3), "pilot_spp"),
        (dict(budget_rays=1000), "budget_rays"),
        (dict(budget_rays=4096, region="foo"), "region"),
        (dict(budget_rays=4096, iterations=0), "iterations"),
        (dict(budget_rays=4096, defensive_fraction=1.0), "defensive_fraction"),
        (dict(budget_rays=4096, sample_offset=-1), "sample_offset"),
    ],
)
def test_invalid_arguments_raise(camera, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _renderer(make_fake_engine(camera)).render(camera, **kwargs)


def test_non_pinhole_camera_is_rejected(camera):
    with pytest.raises(ValueError, match="Pinhole"):
        _renderer(make_fake_engine(camera, camera_type="Fisheye")).render(camera, 4096)


def test_dumper_receives_config_iterations_and_final(camera, tmp_path):
    _renderer(make_fake_engine(camera), dumper=MapDumper(tmp_path)).render(
        camera, 4096, iterations=2, decouple_halves=True
    )
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["allocator"]["class"] == "PerceptualVarianceAllocator" and cfg["seed"] == 0
    assert cfg["decouple_halves"] is True
    assert (tmp_path / "iter_001" / "rates.npy").is_file() and (tmp_path / "final" / "mean_hdr_half_b.npy").is_file()


def test_pilot_firefly_does_not_lock_a_mesh_pixel(camera, inner):
    # A single 50x sample in the 4-spp pilot of an otherwise dark mesh pixel must not mark it display-clipped (D4):
    # the clip margin comes from the raw variance, which the firefly inflates as much as the mean, so the pixel stays
    # eligible and gets adaptive rays.
    pix = int(torch.nonzero(inner.reshape(-1))[0])
    eng = make_fake_engine(camera, firefly_pixel=pix, firefly_sample=0, firefly_value=50.0)
    res = _renderer(eng).render(camera, 64 * 256, sample_offset=0)
    y, x = divmod(pix, camera.width)
    assert res.mean_hdr[y, x].min() > 1.0  # raw mean of the pilot is far above white
    assert res.maps[0]["weight_display_clip"][y, x] > 0
    assert res.counts[y, x] > 4
