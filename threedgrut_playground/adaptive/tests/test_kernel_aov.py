# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-ray AOV outputs of ``Tracer.render_playground``.

The six ``aov_*`` keys are always present. They describe the first mesh hit of the input ray (``t``, normal,
primitive type), the Gaussian transmittance and depth of the first segment (input ray up to that hit), and the
number of redirections along the path. See the ``render_playground`` docstring for the exact semantics.
"""

import warnings

import pytest
import torch

from threedgrut_playground.adaptive.tests.kernel_helpers import (
    FRAME_ID,
    HARD,
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
    visible_sphere_mask,
)

AOV_KEYS = (
    "aov_first_hit_t",
    "aov_first_normal",
    "aov_first_prim_type",
    "aov_first_hit_transmittance",
    "aov_num_bounces",
    "aov_gauss_depth",
)
AOV_SPECS = {  # key -> (dtype, channels)
    "aov_first_hit_t": (torch.float32, 1),
    "aov_first_normal": (torch.float32, 3),
    "aov_first_prim_type": (torch.int32, 1),
    "aov_first_hit_transmittance": (torch.float32, 1),
    "aov_num_bounces": (torch.int32, 1),
    "aov_gauss_depth": (torch.float32, 1),
}
R = SPHERE_RADIUS
N = IMG_H * IMG_W

pytestmark = requires_gpu_ckpt


def _setup(engine, prim="MIRROR", material="solid"):
    from threedgrut_playground.engine import OptixPrimitiveTypes

    center = configure_sphere(engine, OptixPrimitiveTypes[prim], material)
    rays_o, rays_d = make_pinhole_rays(*default_view(engine))
    return rays_o, rays_d, center


def _masks(rays_o, rays_d, center, frac=0.8):
    """``[1,H,W]`` hit / miss masks from the analytic sphere (with a margin around the silhouette)."""
    _, p = analytic_sphere(rays_o, rays_d, center, R)
    return p < frac * R, p > 1.05 * R


def _run(engine, rays_o, rays_d, pixel_id=None, sample_index=None, *, opts=SMOOTH, **kwargs):
    return trace_playground(
        engine,
        rays_o,
        rays_d,
        opts=opts,
        envmap=gradient_envmap(),
        ray_pixel_id=pixel_id,
        ray_sample_index=sample_index,
        **kwargs,
    )


def _ch0(out, key):
    return out[key][..., 0]


@pytest.mark.parametrize("with_ids", [False, True])
def test_aov_keys_dtypes_shapes(playground_engine, with_ids):
    o, d, _ = _setup(playground_engine)
    o, d = torch.cat([o, o]), torch.cat([d, d])
    ids = grid_ids(2, IMG_H, IMG_W) if with_ids else (None, None)
    out = _run(playground_engine, o, d, *ids)
    assert {"pred_features", "pred_opacity", "pred_dist", "pred_normals", "hits_count", "last_ray_o", "last_ray_d"} <= (
        out.keys()
    )
    for k, (dtype, c) in AOV_SPECS.items():
        assert out[k].shape == (2, IMG_H, IMG_W, c), k
        assert out[k].dtype == dtype, k
        assert out[k].is_cuda, k


@pytest.mark.parametrize("opts,min_dot", [(SMOOTH, 0.995), (HARD, 0.99)])
def test_aov_first_hit_matches_analytic_sphere(playground_engine, opts, min_dot):
    o, d, center = _setup(playground_engine)
    hit, _ = _masks(o, d, center)
    out = _run(playground_engine, o, d, opts=opts)
    t_ana, _ = analytic_sphere(o, d, center, R)
    assert hit.sum() > 300

    dt = _ch0(out, "aov_first_hit_t")[hit] - t_ana[hit]
    assert dt.min() >= -1e-4 and dt.max() <= 0.02 * R  # inscribed mesh is never in front of the sphere

    n = out["aov_first_normal"][hit]
    p_ana = o + t_ana[..., None] * d
    n_ana = torch.nn.functional.normalize(p_ana - center.to(p_ana), dim=-1)[hit]
    assert (n.norm(dim=-1) - 1).abs().max() < 1e-4
    assert (n * n_ana).sum(-1).min() > min_dot
    assert (_ch0(out, "aov_first_prim_type")[hit] == 1).all()


def test_aov_misses_report_sentinels(playground_engine):
    o, d, center = _setup(playground_engine)
    _, miss = _masks(o, d, center)
    out = _run(playground_engine, o, d, opts=SMOOTH)
    assert miss.sum() > 1000
    assert (_ch0(out, "aov_first_hit_t")[miss] == -1.0).all()
    assert (out["aov_first_normal"][miss] == 0.0).all()
    assert (_ch0(out, "aov_first_prim_type")[miss] == 0).all()
    assert (_ch0(out, "aov_first_hit_transmittance")[miss] == 0.0).all()
    assert (_ch0(out, "aov_num_bounces")[miss] == 0).all()
    gd = _ch0(out, "aov_gauss_depth")[miss]
    assert torch.isfinite(gd).all() and (gd >= 0).all()


@pytest.mark.parametrize(
    "prim,material,region_frac,expected_nb",
    [
        ("MIRROR", "solid", 0.8, 1),
        ("GLASS", "solid", 0.5, 2),
        ("DIFFUSE", "solid", 0.8, 0),
        ("PBR", "polished_marble", 0.8, None),
    ],
)
def test_aov_prim_type_and_bounces(playground_engine, prim, material, region_frac, expected_nb):
    from threedgrut_playground.engine import OptixPrimitiveTypes

    # NO_GAUSS: with Gaussians, accumulated alpha >= 0.995 can end the loop before the glass exit hit.
    o, d, center = _setup(playground_engine, prim=prim, material=material)
    region, miss = _masks(o, d, center, frac=region_frac)
    out = _run(playground_engine, o, d, opts=NO_GAUSS)
    prim_type = _ch0(out, "aov_first_prim_type")
    nb = _ch0(out, "aov_num_bounces")
    assert region.sum() > 100 and miss.sum() > 1000
    assert (prim_type[region] == OptixPrimitiveTypes[prim].value).all()
    assert (nb[miss] == 0).all()
    if expected_nb is not None:
        assert (nb[region] == expected_nb).all(), torch.unique(nb[region])
    else:
        assert (nb[region] >= 1).all() and (nb[region] <= 15).all()
        assert (nb[region] == 1).float().mean() > 0.9


def test_aov_alpha_pass_through_not_counted(playground_engine):
    o, d, center = _setup(playground_engine, prim="PBR", material="adaptive_test_transparent")
    hit, _ = _masks(o, d, center)
    out = _run(playground_engine, o, d, opts=NO_GAUSS)
    t_ana, _ = analytic_sphere(o, d, center, R)
    assert hit.sum() > 300
    assert (_ch0(out, "aov_first_prim_type")[hit] == 4).all()
    assert (_ch0(out, "aov_num_bounces") == 0).all()
    dt = _ch0(out, "aov_first_hit_t")[hit] - t_ana[hit]
    assert dt.min() >= -1e-4 and dt.max() <= 0.02 * R  # the front surface, not the back one


def test_aov_transmittance_sentinels_and_range(playground_engine):
    o, d, center = _setup(playground_engine)
    hit, miss = _masks(o, d, center)

    out = _run(playground_engine, o, d, opts=NO_GAUSS)
    T = _ch0(out, "aov_first_hit_transmittance")
    assert bit_equal(T[hit], torch.ones_like(T[hit]))
    assert bit_equal(T[miss], torch.zeros_like(T[miss]))

    out = _run(playground_engine, o, d, opts=SMOOTH)
    T = _ch0(out, "aov_first_hit_transmittance")
    prim = _ch0(out, "aov_first_prim_type")
    assert ((T >= 0) & (T <= 1)).all()
    assert bit_equal(T[prim == 0], torch.zeros_like(T[prim == 0]))
    assert (T[prim != 0] > 0).all()


def test_aov_transmittance_matches_opacity_single_segment(playground_engine):
    # max_pbr_bounces=1: the loop stops after the first segment, so on hit pixels pred_opacity is the density of
    # that segment. Scene-independent consistency check.
    o, d, center = _setup(playground_engine, prim="PBR", material="oak_wood")
    hit, _ = _masks(o, d, center)
    out = _run(playground_engine, o, d, opts=SMOOTH, max_pbr_bounces=1)
    T = _ch0(out, "aov_first_hit_transmittance")
    assert hit.sum() > 300
    assert torch.allclose(T[hit], 1.0 - out["pred_opacity"][..., 0][hit], atol=1e-6)


def test_aov_transmittance_matches_ray_max_t_render(playground_engine):
    # D6 semantics through a pre-existing API path: with the sphere moved behind the camera, a render with
    # ray_max_t = first-hit t integrates the Gaussian segment [1e-9, t], the same segment as in the kernel.
    from threedgrut_playground.engine import OptixPrimitiveTypes

    o, d, center = _setup(playground_engine)
    out = _run(playground_engine, o, d, opts=SMOOTH)
    T = _ch0(out, "aov_first_hit_transmittance")
    t = _ch0(out, "aov_first_hit_t")
    sel = _ch0(out, "aov_first_prim_type") != 0

    # Coarse guard only: visible_sphere_mask cuts at the analytic t (the shell between mesh and sphere differs).
    visible = visible_sphere_mask(playground_engine, o, d, center)
    assert visible.sum() > 200 and sel[visible].all()

    eye = o[0, 0, 0]
    away = torch.nn.functional.normalize(eye - center.to(eye), dim=0)
    configure_sphere(playground_engine, OptixPrimitiveTypes.MIRROR, center=eye + 1000.0 * away)
    ray_max_t = torch.where(sel, t, torch.full_like(t, 1e9)).contiguous()
    clip = _run(playground_engine, o, d, opts=SMOOTH, ray_max_t=ray_max_t)
    assert (_ch0(clip, "aov_first_prim_type") == 0).all()
    assert sel.sum() > 300
    assert torch.allclose(T[sel], 1.0 - clip["pred_opacity"][..., 0][sel], atol=1e-6)


def test_aov_diffuse_first_hit_not_double_integrated(playground_engine):
    from threedgrut_playground.engine import OptixPrimitiveTypes

    o, d, center = _setup(playground_engine, prim="DIFFUSE")
    hit, _ = _masks(o, d, center)
    diffuse = _run(playground_engine, o, d, opts=SMOOTH)
    configure_sphere(playground_engine, OptixPrimitiveTypes.MIRROR, "solid")
    mirror = _run(playground_engine, o, d, opts=SMOOTH)

    T_diffuse = _ch0(diffuse, "aov_first_hit_transmittance")
    T_mirror = _ch0(mirror, "aov_first_hit_transmittance")
    gd_diffuse = _ch0(diffuse, "aov_gauss_depth")
    gd_mirror = _ch0(mirror, "aov_gauss_depth")
    assert (_ch0(diffuse, "aov_first_prim_type")[hit] == 3).all()
    assert (
        T_mirror[hit] < 0.99
    ).sum() >= 20, "view has no Gaussian occlusion in front of the sphere; test would be vacuous"
    assert torch.allclose(T_diffuse[hit], T_mirror[hit], atol=1e-6)  # would be ~T_mirror**2 without the snapshot
    assert torch.allclose(gd_diffuse[hit], gd_mirror[hit], rtol=1e-5, atol=1e-5)


def test_aov_gauss_depth_zero_without_gaussians(playground_engine):
    o, d, _ = _setup(playground_engine)
    out = _run(playground_engine, o, d, opts=NO_GAUSS)
    gd = out["aov_gauss_depth"]
    assert bit_equal(gd, torch.zeros_like(gd))


def test_aov_gauss_depth_bounded_by_first_hit(playground_engine):
    o, d, _ = _setup(playground_engine)
    out = _run(playground_engine, o, d, opts=SMOOTH)
    gd = _ch0(out, "aov_gauss_depth")
    t = _ch0(out, "aov_first_hit_t")
    hit = _ch0(out, "aov_first_prim_type") != 0
    assert hit.sum() > 300
    assert torch.isfinite(gd).all() and (gd >= 0).all()
    assert (gd[hit] <= t[hit] + 1e-4).all(), (gd[hit] - t[hit]).max()


def test_aov_gauss_depth_agrees_with_transmittance(playground_engine):
    o, d, _ = _setup(playground_engine)
    out = _run(playground_engine, o, d, opts=SMOOTH)
    gd = _ch0(out, "aov_gauss_depth")
    T = _ch0(out, "aov_first_hit_transmittance")
    hit = _ch0(out, "aov_first_prim_type") != 0
    assert hit.sum() > 300
    assert torch.equal((gd > 0)[hit], (T < 1)[hit])


def test_info_gauss_depth_vs_core_tracer(playground_engine):
    # INFORMATIONAL, never fails on the comparison. Two different hit-integration code paths (playground first
    # segment vs the core 3DGRT tracer) on one checkpoint and camera, on pixels where no mesh is hit.
    o, d, center = _setup(playground_engine)
    _, miss = _masks(o, d, center)
    out = _run(playground_engine, o, d, opts=SMOOTH)
    with torch.no_grad():
        core = playground_engine.scene_mog.trace(rays_o=o.clone(), rays_d=d.clone())
    core_dist = core["pred_dist"][..., 0]
    mask = miss & (core["pred_opacity"][..., 0] > 0.5)
    if mask.sum() < 50:
        pytest.skip("too few opaque miss pixels")
    gd = _ch0(out, "aov_gauss_depth")
    rel = (gd * out["pred_opacity"][..., 0] - core_dist).abs()[mask] / core_dist[mask]
    median = rel.median().item()
    p95 = torch.quantile(rel.float(), 0.95).item()
    print(f"\n[info] gauss depth vs core tracer: n={int(mask.sum())} median rel={median:.3e} p95 rel={p95:.3e}")
    if median > 1e-3 or p95 > 1e-2:
        warnings.warn(
            f"gauss depth differs from the core tracer: median rel {median:.3e}, p95 rel {p95:.3e}", stacklevel=2
        )


def test_aov_batch_entries_independent(playground_engine):
    o0, d0, _ = _setup(playground_engine)
    eye, at = default_view(playground_engine)
    o1, d1 = make_pinhole_rays(eye + torch.tensor([0.3, 0.0, 0.0], device=eye.device, dtype=eye.dtype), at)
    batched = _run(playground_engine, torch.cat([o0, o1]), torch.cat([d0, d1]))
    alone = [_run(playground_engine, o0, d0), _run(playground_engine, o1, d1)]
    for k in AOV_KEYS:
        for b in range(2):
            assert bit_equal(batched[k][b : b + 1], alone[b][k]), (k, b)


def test_aov_zero_pbr_bounce_budget_reports_no_hit(playground_engine):
    # Documented caveat: the path loop never runs, so every AOV is at its no-hit sentinel even with a mesh in view.
    # pred_features is not asserted on (lastRayDir is never set on this path).
    o, d, center = _setup(playground_engine)
    hit, _ = _masks(o, d, center)
    assert hit.sum() > 300
    out = _run(playground_engine, o, d, opts=SMOOTH, max_pbr_bounces=0)
    sentinels = {
        "aov_first_hit_t": -1.0,
        "aov_first_normal": 0.0,
        "aov_first_prim_type": 0,
        "aov_first_hit_transmittance": 0.0,
        "aov_num_bounces": 0,
        "aov_gauss_depth": 0.0,
    }
    for k, v in sentinels.items():
        assert bit_equal(out[k], torch.full_like(out[k], v)), k


def test_aov_ids_match_no_ids_for_mirror(playground_engine):
    o, d, _ = _setup(playground_engine)
    plain = _run(playground_engine, o, d, frame_id=FRAME_ID)
    with_ids = _run(playground_engine, o, d, *grid_ids(1, IMG_H, IMG_W), frame_id=FRAME_ID)
    for k in AOV_KEYS:
        assert bit_equal(plain[k], with_ids[k]), k


def test_aov_flat_shuffled_matches_grid_pbr(playground_engine):
    o, d, _ = _setup(playground_engine, prim="PBR", material="oak_wood")
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
    for k in AOV_KEYS:
        c = grid[k].shape[-1]
        assert bit_equal(flat[k].reshape(-1, c)[inv], grid[k].reshape(-1, c)), k
