# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-ray RNG keys ``(ray_pixel_id, ray_sample_index)`` of ``Tracer.render_playground``.

With ids, every random draw of a ray depends only on its ids (and its geometry), never on its launch slot, the batch
layout or ``frame_id``. Both RNG sites are covered: the alpha-test seed in ``__raygen__rg`` and the PBR BSDF draw.
"""

import pytest
import torch

from threedgrut_playground.adaptive.tests.kernel_helpers import (
    FRAME_ID,
    IMG_H,
    IMG_W,
    NO_GAUSS,
    SMOOTH,
    SPHERE_RADIUS,
    analytic_sphere,
    bit_equal,
    configure_sphere,
    default_view,
    gradient_envmap,
    grid_ids,
    make_pinhole_rays,
    requires_gpu_ckpt,
    trace_playground,
)

RNG_KEYS = ("pred_features", "pred_opacity", "last_ray_o", "last_ray_d")
N = IMG_H * IMG_W

pytestmark = requires_gpu_ckpt


def _setup(engine, prim="PBR", material="oak_wood"):
    from threedgrut_playground.engine import OptixPrimitiveTypes

    center = configure_sphere(engine, OptixPrimitiveTypes[prim], material)
    rays_o, rays_d = make_pinhole_rays(*default_view(engine))
    return rays_o, rays_d, center


def _masks(rays_o, rays_d, center):
    """``[1,H,W]`` hit / miss masks from the analytic sphere (with a margin around the silhouette)."""
    _, p = analytic_sphere(rays_o, rays_d, center, SPHERE_RADIUS)
    return p < 0.8 * SPHERE_RADIUS, p > 1.05 * SPHERE_RADIUS


def _run(engine, rays_o, rays_d, pixel_id=None, sample_index=None, *, opts=SMOOTH, frame_id=FRAME_ID, envmap=None):
    return trace_playground(
        engine,
        rays_o,
        rays_d,
        opts=opts,
        frame_id=frame_id,
        envmap=gradient_envmap() if envmap is None else envmap,
        ray_pixel_id=pixel_id,
        ray_sample_index=sample_index,
    )


def _flat(t: torch.Tensor) -> torch.Tensor:
    """``[1,H,W,C]`` -> ``[H*W, C]``."""
    return t.reshape(-1, t.shape[-1])


@pytest.mark.parametrize("material", ["oak_wood", "adaptive_test_blend"])
def test_ids_flat_shuffled_matches_grid(playground_engine, material):
    o, d, _ = _setup(playground_engine, material=material)
    pid, sidx = grid_ids(1, IMG_H, IMG_W)
    grid = _run(playground_engine, o, d, pid, sidx)

    perm = torch.randperm(N, generator=torch.Generator().manual_seed(0)).to("cuda")
    inv = torch.argsort(perm)
    flat = _run(
        playground_engine,
        o.reshape(N, 3)[perm].reshape(1, 1, N, 3).contiguous(),
        d.reshape(N, 3)[perm].reshape(1, 1, N, 3).contiguous(),
        pid.reshape(N)[perm].reshape(1, 1, N).contiguous(),
        sidx.reshape(N)[perm].reshape(1, 1, N).contiguous(),
    )
    for k in RNG_KEYS:
        c = grid[k].shape[-1]
        assert bit_equal(flat[k].reshape(-1, c)[inv], grid[k].reshape(-1, c)), k


def test_ids_row_layout_matches_grid(playground_engine):
    o, d, _ = _setup(playground_engine)
    pid, sidx = grid_ids(1, IMG_H, IMG_W)
    grid = _run(playground_engine, o, d, pid, sidx)
    rows = _run(
        playground_engine,
        o.reshape(1, 32, 128, 3).contiguous(),
        d.reshape(1, 32, 128, 3).contiguous(),
        pid.reshape(1, 32, 128).contiguous(),
        sidx.reshape(1, 32, 128).contiguous(),
    )
    for k in RNG_KEYS:
        assert bit_equal(_flat(rows[k]), _flat(grid[k])), k


def test_ids_subset_batch_matches_full(playground_engine):
    o, d, _ = _setup(playground_engine)
    pid, sidx = grid_ids(1, IMG_H, IMG_W)
    grid = _run(playground_engine, o, d, pid, sidx)
    m = N // 2
    sub = _run(
        playground_engine,
        o.reshape(N, 3)[0::2].reshape(1, 1, m, 3).contiguous(),
        d.reshape(N, 3)[0::2].reshape(1, 1, m, 3).contiguous(),
        pid.reshape(N)[0::2].reshape(1, 1, m).contiguous(),
        sidx.reshape(N)[0::2].reshape(1, 1, m).contiguous(),
    )
    for k in RNG_KEYS:
        assert bit_equal(_flat(sub[k]), _flat(grid[k])[0::2]), k


@pytest.mark.parametrize("material", ["oak_wood", "adaptive_test_blend"])
def test_ids_batch_dim_independent(playground_engine, material):
    o, d, _ = _setup(playground_engine, material=material)
    pid, _ = grid_ids(1, IMG_H, IMG_W)
    s0 = torch.zeros_like(pid)
    s5 = torch.full_like(pid, 5)
    batched = _run(
        playground_engine,
        torch.cat([o, o]),
        torch.cat([d, d]),
        torch.cat([pid, pid]),
        torch.cat([s0, s5]),
    )
    alone0 = _run(playground_engine, o, d, pid, s0)
    alone5 = _run(playground_engine, o, d, pid, s5)
    for k in RNG_KEYS:
        assert bit_equal(batched[k][0:1], alone0[k]), k
        assert bit_equal(batched[k][1:2], alone5[k]), k


@pytest.mark.parametrize("material", ["oak_wood", "adaptive_test_blend"])
def test_ids_frame_id_independent(playground_engine, material):
    o, d, _ = _setup(playground_engine, material=material)
    pid, sidx = grid_ids(1, IMG_H, IMG_W)
    a = _run(playground_engine, o, d, pid, sidx, frame_id=0)
    b = _run(playground_engine, o, d, pid, sidx, frame_id=123457)
    for k in RNG_KEYS:
        assert bit_equal(a[k], b[k]), k


def test_ids_duplicate_rays_identical(playground_engine):
    o, d, _ = _setup(playground_engine)
    pid, sidx = grid_ids(1, IMG_H, IMG_W)
    twice = _run(
        playground_engine,
        o.reshape(N, 3).repeat(2, 1).reshape(1, 1, 2 * N, 3).contiguous(),
        d.reshape(N, 3).repeat(2, 1).reshape(1, 1, 2 * N, 3).contiguous(),
        pid.reshape(N).repeat(2).reshape(1, 1, 2 * N).contiguous(),
        sidx.reshape(N).repeat(2).reshape(1, 1, 2 * N).contiguous(),
    )
    for k in RNG_KEYS:
        flat = _flat(twice[k])
        assert bit_equal(flat[:N], flat[N:]), k


@pytest.mark.parametrize("material,min_frac", [("oak_wood", 0.9), ("adaptive_test_blend", 0.8)])
def test_ids_different_sample_index_differ(playground_engine, material, min_frac):
    o, d, center = _setup(playground_engine, material=material)
    hit, miss = _masks(o, d, center)
    a = _run(playground_engine, o, d, *grid_ids(1, IMG_H, IMG_W, sample_base=0), opts=NO_GAUSS)
    b = _run(playground_engine, o, d, *grid_ids(1, IMG_H, IMG_W, sample_base=1), opts=NO_GAUSS)
    # Blend: ~1/16 of pixels pass through both surfaces in both samples and stay equal.
    differs = (a["pred_features"] != b["pred_features"]).any(-1)
    assert hit.sum() > 300 and miss.sum() > 1000
    assert differs[hit].float().mean() > min_frac
    assert bit_equal(a["pred_features"][miss], b["pred_features"][miss])  # misses use no RNG


def test_ids_pbr_key_depends_on_pixel_id(playground_engine):
    # Same rays and sample_index, pixel_id shifted by one. oak_wood is opaque (no alpha draws), so only the PBR key is
    # exercised: a PBR key built without pixel_id would give bit-identical images here.
    o, d, center = _setup(playground_engine, material="oak_wood")
    hit, miss = _masks(o, d, center)
    pid, sidx = grid_ids(1, IMG_H, IMG_W)
    a = _run(playground_engine, o, d, pid, sidx, opts=NO_GAUSS)
    b = _run(playground_engine, o, d, (pid + 1).contiguous(), sidx, opts=NO_GAUSS)
    differs = (a["pred_features"] != b["pred_features"]).any(-1)
    assert hit.sum() > 300 and miss.sum() > 1000
    assert differs[hit].float().mean() > 0.9
    assert bit_equal(a["pred_features"][miss], b["pred_features"][miss])  # misses use no RNG


def test_ids_alpha_seed_depends_on_pixel_id(playground_engine):
    # Isolates the alpha seed, as in test_ids_alpha_seed_depends_on_sample_index: the straight mask (alpha test fails
    # at both surfaces, so no BSDF draw happens) is a function of the alpha draws only. pred_features would not do:
    # BSDF draws keyed by pixel_id make images differ even if the alpha seed ignored pixel_id. A seed without
    # pixel_id gives every pixel the same alpha stream, so the straight mask is all-or-nothing in each run and
    # identical between the runs (mismatch fraction exactly 0). Correct kernel: about 2 * 0.25 * 0.75 = 0.375.
    o, d, center = _setup(playground_engine, material="adaptive_test_blend")
    hit, _ = _masks(o, d, center)
    pid, sidx = grid_ids(1, IMG_H, IMG_W)

    def straight(r):
        return (r["last_ray_d"] - d).abs().amax(-1) < 1e-6

    s0 = straight(_run(playground_engine, o, d, pid, sidx, opts=NO_GAUSS))
    s1 = straight(_run(playground_engine, o, d, (pid + 1).contiguous(), sidx, opts=NO_GAUSS))
    assert hit.sum() > 300
    assert 0.1 < s0[hit].float().mean() < 0.4  # expected 0.25 for alpha 0.5
    assert (s0 != s1)[hit].float().mean() > 0.2


def test_ids_alpha_seed_depends_on_sample_index(playground_engine):
    # A path is "straight" iff the alpha test fails at both the front and back surfaces (no BSDF draw ever happens
    # on it), so the straight mask is a function of the alpha draws only.
    o, d, center = _setup(playground_engine, material="adaptive_test_blend")
    hit, _ = _masks(o, d, center)

    def run(sample_base):
        return _run(playground_engine, o, d, *grid_ids(1, IMG_H, IMG_W, sample_base=sample_base), opts=NO_GAUSS)

    def straight(r):
        return (r["last_ray_d"] - d).abs().amax(-1) < 1e-6

    s0, s1 = straight(run(0)), straight(run(1))
    assert 0.1 < s0[hit].float().mean() < 0.4  # expected 0.25 for alpha 0.5
    assert (s0 != s1)[hit].float().mean() > 0.2  # expected 0.375 if the seed uses sample_index


@pytest.mark.parametrize("prim", ["MIRROR", "GLASS"])
def test_ids_mirror_glass_match_no_ids(playground_engine, prim):
    o, d, _ = _setup(playground_engine, prim=prim)
    plain = _run(playground_engine, o, d, frame_id=FRAME_ID)
    with_ids = _run(playground_engine, o, d, *grid_ids(1, IMG_H, IMG_W), frame_id=FRAME_ID)
    for k in RNG_KEYS:
        assert bit_equal(plain[k], with_ids[k]), k


def test_ids_expanded_tensors_match_contiguous(playground_engine):
    o, d, _ = _setup(playground_engine)
    pid, _ = grid_ids(1, IMG_H, IMG_W)
    sample_index = torch.tensor(3, dtype=torch.int32, device="cuda").expand(1, IMG_H, IMG_W)
    pixel_id = pid.transpose(1, 2).contiguous().transpose(1, 2)
    assert not sample_index.is_contiguous() and not pixel_id.is_contiguous()
    strided = _run(playground_engine, o, d, pixel_id, sample_index)
    dense = _run(playground_engine, o, d, pixel_id.contiguous(), sample_index.contiguous())
    for k in RNG_KEYS:
        assert bit_equal(strided[k], dense[k]), k


def test_ids_extreme_values(playground_engine):
    o, d, center = _setup(playground_engine)
    hit, _ = _masks(o, d, center)
    pid, _ = grid_ids(1, IMG_H, IMG_W)
    pixel_id = ((2**31 - 1) - pid).contiguous()  # (2**31 - 1) - (y*W + x), no int32 overflow
    minus_one = torch.full_like(pid, -1)
    r1 = _run(playground_engine, o, d, pixel_id, minus_one, opts=NO_GAUSS)
    r2 = _run(playground_engine, o, d, pixel_id, minus_one.clone(), opts=NO_GAUSS)
    r0 = _run(playground_engine, o, d, pixel_id, torch.zeros_like(pid), opts=NO_GAUSS)
    assert torch.isfinite(r1["pred_features"]).all() and bit_equal(r1["pred_features"], r2["pred_features"])
    assert (r1["pred_features"] != r0["pred_features"]).any(-1)[hit].float().mean() > 0.9


def test_render_playground_validation_errors(playground_engine):
    o, d, _ = _setup(playground_engine)
    pid, sidx = grid_ids(1, IMG_H, IMG_W)
    with pytest.raises(ValueError, match="both"):
        trace_playground(playground_engine, o, d, ray_pixel_id=pid)
    with pytest.raises(ValueError, match="int32"):
        trace_playground(playground_engine, o, d, ray_pixel_id=pid.long(), ray_sample_index=sidx.long())
    with pytest.raises(ValueError, match="shape"):
        trace_playground(
            playground_engine, o, d, ray_pixel_id=pid.reshape(1, 1, -1), ray_sample_index=sidx.reshape(1, 1, -1)
        )
