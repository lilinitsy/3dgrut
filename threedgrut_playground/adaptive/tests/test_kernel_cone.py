# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cone parameter block and the split column of the playground cone tracing mode."""

import pytest
import torch

from threedgrut_playground.adaptive.tests.kernel_helpers import (
    IMG_H,
    IMG_W,
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
    scene_center,
    trace_playground,
)

CONE_TRACING = 8  # OptixPlaygroundRenderOptions.CONE_TRACING


def test_make_cone_params_layout():
    from threedgrut_playground.tracer import make_cone_params

    params = make_cone_params(0.1, 1.4, 0.005, 0.2, 7)
    assert params.dtype == torch.float32
    assert params.device.type == "cpu"
    assert params.shape == (5,)
    assert params.tolist() == pytest.approx([0.1, 1.4, 0.005, 0.2, 7.0])


def _render(engine, rays_o, rays_d, ids, cone_on, split_column=0):
    # max_roughness 1.0 so the roughness-0.5 test material takes the cone path (handlePBR) on the cone side.
    from threedgrut_playground.tracer import make_cone_params

    pixel_id, sample_index = ids
    return trace_playground(
        engine,
        rays_o,
        rays_d,
        opts=SMOOTH | (CONE_TRACING if cone_on else 0),
        ray_pixel_id=pixel_id,
        ray_sample_index=sample_index,
        cone_params=make_cone_params(0.0, 0.0, 0.0, 1.0, split_column),
    )["pred_features"]


def _setup(engine):
    from threedgrut_playground.engine import OptixPrimitiveTypes

    configure_sphere(engine, OptixPrimitiveTypes.PBR, "brushed_copper")
    rays_o, rays_d = make_pinhole_rays(*default_view(engine))
    return rays_o, rays_d, grid_ids(1, IMG_H, IMG_W)


@requires_gpu_ckpt
def test_split_column_halves_match_full_renders(playground_engine):
    rays_o, rays_d, ids = _setup(playground_engine)
    half = IMG_W // 2
    off = _render(playground_engine, rays_o, rays_d, ids, False)
    split = _render(playground_engine, rays_o, rays_d, ids, True, half)
    full = _render(playground_engine, rays_o, rays_d, ids, True, 0)
    assert not bit_equal(off, full), "cone mode must change the output, or the test proves nothing"
    assert bit_equal(split[..., :half, :], off[..., :half, :])
    assert bit_equal(split[..., half:, :], full[..., half:, :])


@requires_gpu_ckpt
def test_split_column_out_of_range(playground_engine):
    rays_o, rays_d, ids = _setup(playground_engine)
    off = _render(playground_engine, rays_o, rays_d, ids, False)
    past_end = _render(playground_engine, rays_o, rays_d, ids, True, IMG_W)
    assert bit_equal(past_end, off)
    full = _render(playground_engine, rays_o, rays_d, ids, True, 0)
    negative = _render(playground_engine, rays_o, rays_d, ids, True, -5)
    assert bit_equal(negative, full)


@requires_gpu_ckpt
def test_split_line_is_white_column(playground_engine):
    engine = playground_engine
    rays_o, rays_d, _ = _setup(engine)
    old = (engine.use_cone_tracing, engine.cone_split, engine.cone_max_sigma, engine.frame_id)
    try:
        engine.use_cone_tracing = True
        engine.cone_max_sigma = 0.0
        engine.cone_split = 0.0
        engine.frame_id = 5
        plain = engine._render_playground_hybrid(rays_o.clone(), rays_d.clone())["pred_features"].clone()
        engine.cone_split = 0.5
        engine.frame_id = 5
        split = engine._render_playground_hybrid(rays_o.clone(), rays_d.clone())["pred_features"].clone()
    finally:
        engine.use_cone_tracing, engine.cone_split, engine.cone_max_sigma, engine.frame_id = old
    column = IMG_W // 2
    assert (split[:, :, column, :] == 1.0).all()
    assert (plain[:, :, column, :] < 1.0).any(), "the line must not be a pre-existing white column"


@requires_gpu_ckpt
def test_cone_off_bit_identical_after_inflated_bvh(playground_engine):
    engine = playground_engine
    rays_o, rays_d, _ = _setup(engine)
    old = (engine.use_cone_tracing, engine.cone_max_sigma, engine.frame_id)
    try:
        engine.use_cone_tracing = False
        engine.frame_id = 11
        first = engine._render_playground_hybrid(rays_o.clone(), rays_d.clone())["pred_features"].clone()
        engine.use_cone_tracing = True
        engine.cone_max_sigma = 0.005
        engine.frame_id = 11
        engine._render_playground_hybrid(rays_o.clone(), rays_d.clone())
        engine.use_cone_tracing = False
        engine.frame_id = 11
        last = engine._render_playground_hybrid(rays_o.clone(), rays_d.clone())["pred_features"].clone()
    finally:
        engine.use_cone_tracing, engine.cone_max_sigma, engine.frame_id = old
        if engine._gs_bvh_inflate != 0.0:
            engine._gs_bvh_inflate = 0.0
            engine.tracer.build_gs_acc(gaussians=engine.scene_mog, rebuild=True, inflate=0.0)
    assert bit_equal(first, last)


#################################
##   --- Material parity ---    ##
#################################

DISABLE_GAUSSIAN_TRACING = 2


def _metal(engine, name, **kwargs):
    from threedgrut_playground.engine import PBRMaterial

    materials = engine.primitives.registered_materials
    fields = dict(
        diffuse_factor=torch.tensor([0.9, 0.6, 0.3, 1.0], device="cuda", dtype=torch.float32),
        emissive_factor=torch.zeros(3, device="cuda", dtype=torch.float32),
        metallic_factor=1.0,
        roughness_factor=0.1,
        transmission_factor=0.0,
        ior=1.1,
    )
    fields.update(kwargs)
    materials[name] = PBRMaterial(material_id=len(materials), **fields)


def _cone_render(engine, material, *, max_roughness=1.0, pixel_spread=0.0, max_sigma=0.0, cone_on=True, opts=SMOOTH):
    from threedgrut_playground.engine import OptixPrimitiveTypes
    from threedgrut_playground.tracer import make_cone_params

    configure_sphere(engine, OptixPrimitiveTypes.PBR, material)
    rays_o, rays_d = make_pinhole_rays(*default_view(engine))
    pixel_id, sample_index = grid_ids(1, IMG_H, IMG_W)
    return trace_playground(
        engine,
        rays_o,
        rays_d,
        opts=opts | (CONE_TRACING if cone_on else 0),
        ray_pixel_id=pixel_id,
        ray_sample_index=sample_index,
        cone_params=make_cone_params(pixel_spread, 0.0, max_sigma, max_roughness, 0),
    )["pred_features"]


def _filled(h, w, values):
    return torch.tensor(values, device="cuda", dtype=torch.float32).expand(h, w, len(values)).contiguous()


@requires_gpu_ckpt
def test_cone_diffuse_texture_matches_factor(playground_engine):
    engine = playground_engine
    _metal(engine, "cone_tex_a")
    one = torch.ones(4, device="cuda", dtype=torch.float32)
    _metal(engine, "cone_tex_b", diffuse_factor=one, diffuse_map=_filled(4, 4, [0.9, 0.6, 0.3, 1.0]))
    a = _cone_render(engine, "cone_tex_a")
    b = _cone_render(engine, "cone_tex_b")
    assert not bit_equal(a, _cone_render(engine, "cone_tex_a", cone_on=False))
    torch.testing.assert_close(a, b, atol=1e-5, rtol=0)


@requires_gpu_ckpt
def test_cone_flat_normal_map_is_identity(playground_engine):
    # The kernel applies the texel as the tangent-space normal without a 2x-1 remap (same as the MC path), so the
    # flat map is (0, 0, 1), not the (0.5, 0.5, 1) of an 8-bit image.
    engine = playground_engine
    _metal(engine, "cone_nrm_a")
    _metal(engine, "cone_nrm_b", normal_map=_filled(4, 4, [0.0, 0.0, 1.0, 1.0]))
    a = _cone_render(engine, "cone_nrm_a")
    b = _cone_render(engine, "cone_nrm_b")
    torch.testing.assert_close(a, b, atol=1e-4, rtol=0)
    _metal(engine, "cone_nrm_tilted", normal_map=_filled(4, 4, [0.5, 0.5, 1.0, 1.0]))
    assert not bit_equal(a, _cone_render(engine, "cone_nrm_tilted")), "a tilted normal map must change the render"


@requires_gpu_ckpt
def test_cone_alpha_pass_through(playground_engine):
    from threedgrut_playground.engine import OptixPrimitiveTypes
    from threedgrut_playground.tracer import make_cone_params

    engine = playground_engine
    _metal(engine, "cone_alpha_opaque")
    rays_o, rays_d = make_pinhole_rays(*default_view(engine))
    pixel_id, sample_index = grid_ids(1, IMG_H, IMG_W)
    opts = SMOOTH | DISABLE_GAUSSIAN_TRACING | CONE_TRACING

    def render(material, center=None):
        configure_sphere(engine, OptixPrimitiveTypes.PBR, material, center=center)
        return trace_playground(
            engine,
            rays_o,
            rays_d,
            opts=opts,
            envmap=gradient_envmap(),
            ray_pixel_id=pixel_id,
            ray_sample_index=sample_index,
            cone_params=make_cone_params(0.0, 0.0, 0.0, 1.0, 0),
        )["pred_features"]

    center = scene_center(engine)
    eye = rays_o[0, 0, 0]
    away = torch.nn.functional.normalize(eye - center.to(eye), dim=0)
    far = eye + 1000.0 * away
    behind = render("adaptive_test_transparent", far)
    in_view = render("adaptive_test_transparent")
    opaque = render("cone_alpha_opaque")
    t, p = analytic_sphere(rays_o, rays_d, center, SPHERE_RADIUS)
    on_sphere = (p < 0.9 * SPHERE_RADIUS)[0]
    assert on_sphere.sum() > 100
    # Sensitivity: an opaque sphere must visibly change the sphere pixels, or the pass-through proves nothing.
    assert (opaque - behind)[0][on_sphere].abs().max() > 0.05
    torch.testing.assert_close(in_view, behind, atol=1e-5, rtol=0)


@requires_gpu_ckpt
def test_rough_surface_falls_back_to_mc(playground_engine):
    engine = playground_engine
    _metal(engine, "cone_rough", roughness_factor=0.6)
    cone = _cone_render(engine, "cone_rough", max_roughness=0.2)
    off = _cone_render(engine, "cone_rough", max_roughness=0.2, cone_on=False)
    torch.testing.assert_close(cone, off, atol=1e-5, rtol=0)


@requires_gpu_ckpt
def test_textured_roughness_drives_fallback(playground_engine):
    engine = playground_engine
    _metal(engine, "cone_mr_a", roughness_factor=1.0, metallic_roughness_map=_filled(4, 4, [1.0, 0.1]))
    _metal(engine, "cone_mr_b", roughness_factor=0.1)
    a = _cone_render(engine, "cone_mr_a", max_roughness=0.2)
    b = _cone_render(engine, "cone_mr_b", max_roughness=0.2)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=0)
    mc = _cone_render(engine, "cone_mr_a", max_roughness=0.2, cone_on=False)
    assert not bit_equal(a, mc), "textured roughness 0.1 must take the cone path, not the MC fallback"


#################################
##   --- Hybrid lobes ---       ##
#################################

ENERGY_SAMPLES = 512
ENERGY_BATCH = 64


def _constant_envmap(value=0.5):
    return torch.full((8, 16, 4), value, device="cuda", dtype=torch.float32)


def _energy_mean(engine, material, cone_on):
    """Mean radiance over the visible sphere pixels and 512 samples (cone on, or the MC path with the same ray ids)."""
    from threedgrut_playground.engine import OptixPrimitiveTypes
    from threedgrut_playground.tracer import make_cone_params

    center = configure_sphere(engine, OptixPrimitiveTypes.PBR, material)
    rays_o, rays_d = make_pinhole_rays(*default_view(engine))
    t, p = analytic_sphere(rays_o, rays_d, center, SPHERE_RADIUS)
    mask = (p < 0.9 * SPHERE_RADIUS)[0]
    assert mask.sum() > 100
    rays_o = rays_o.expand(ENERGY_BATCH, -1, -1, -1).contiguous()
    rays_d = rays_d.expand(ENERGY_BATCH, -1, -1, -1).contiguous()
    total = torch.zeros(3, device="cuda", dtype=torch.float64)
    for base in range(0, ENERGY_SAMPLES, ENERGY_BATCH):
        pixel_id, sample_index = grid_ids(ENERGY_BATCH, IMG_H, IMG_W, sample_base=base)
        out = trace_playground(
            engine,
            rays_o,
            rays_d,
            opts=SMOOTH | DISABLE_GAUSSIAN_TRACING | (CONE_TRACING if cone_on else 0),
            envmap=_constant_envmap(),
            ray_pixel_id=pixel_id,
            ray_sample_index=sample_index,
            cone_params=make_cone_params(0.0, 0.0, 0.0, 1.0, 0),
        )["pred_features"]
        total += out[:, mask].double().sum(dim=(0, 1))
    return total / (ENERGY_SAMPLES * mask.sum())


def _assert_energy_matches(engine, material, tol):
    cone = _energy_mean(engine, material, True)
    mc = _energy_mean(engine, material, False)
    rel = ((cone - mc).abs() / mc.abs().clamp(min=1e-6)).max().item()
    print(f"energy {material}: cone={cone.tolist()} mc={mc.tolist()} rel={rel:.4f}")
    assert rel < tol, f"{material}: cone {cone.tolist()} vs mc {mc.tolist()} (rel {rel:.4f})"


@requires_gpu_ckpt
@pytest.mark.parametrize("roughness", [0.1, 0.3, 0.6])
def test_cone_metal_energy_matches_mc(playground_engine, roughness):
    name = f"cone_energy_metal_{roughness}"
    _metal(playground_engine, name, roughness_factor=roughness)
    _assert_energy_matches(playground_engine, name, 0.05)


@requires_gpu_ckpt
def test_cone_dielectric_energy_matches_mc(playground_engine):
    base = torch.tensor([0.8, 0.8, 0.8, 1.0], device="cuda", dtype=torch.float32)
    _metal(playground_engine, "cone_energy_dielectric", diffuse_factor=base, metallic_factor=0.0, roughness_factor=0.3)
    _assert_energy_matches(playground_engine, "cone_energy_dielectric", 0.05)


@requires_gpu_ckpt
def test_cone_transmission_energy_matches_mc(playground_engine):
    base = torch.tensor([0.8, 0.8, 0.8, 1.0], device="cuda", dtype=torch.float32)
    _metal(
        playground_engine,
        "cone_energy_glass",
        diffuse_factor=base,
        metallic_factor=0.0,
        roughness_factor=0.1,
        transmission_factor=1.0,
        ior=1.5,
    )
    _assert_energy_matches(playground_engine, "cone_energy_glass", 0.10)


@requires_gpu_ckpt
def test_cone_inside_transmissive_sphere_is_finite(playground_engine):
    from threedgrut_playground.engine import OptixPrimitiveTypes
    from threedgrut_playground.tracer import make_cone_params

    engine = playground_engine
    _metal(engine, "cone_inside_glass", metallic_factor=0.0, roughness_factor=0.1, transmission_factor=1.0, ior=1.5)
    center = configure_sphere(engine, OptixPrimitiveTypes.PBR, "cone_inside_glass")
    rays_o, _ = make_pinhole_rays(*default_view(engine))
    offset = torch.tensor([0.7 * SPHERE_RADIUS, 0.0, 0.0], device="cuda", dtype=torch.float32)
    rays_o = (center.to(rays_o) + offset).expand(1, IMG_H, IMG_W, 3).contiguous()
    gen = torch.Generator(device="cuda").manual_seed(3)
    rays_d = torch.nn.functional.normalize(torch.randn(1, IMG_H, IMG_W, 3, device="cuda", generator=gen), dim=-1)
    # Rays whose first inside hit exceeds the critical angle (ior 1.5) undergo total internal reflection.
    oc = rays_o - center.to(rays_o)
    b = (oc * rays_d).sum(dim=-1)
    t_hit = -b + torch.sqrt(b * b - ((oc * oc).sum(dim=-1) - SPHERE_RADIUS**2))
    normal = (oc + t_hit[..., None] * rays_d) / SPHERE_RADIUS
    sin_incidence = torch.sqrt(torch.clamp(1.0 - (rays_d * normal).sum(dim=-1) ** 2, min=0.0))
    assert (sin_incidence * 1.5 > 1.0).sum() > 100, "the test must reach total internal reflection"
    pixel_id, sample_index = grid_ids(1, IMG_H, IMG_W)
    out = trace_playground(
        engine,
        rays_o,
        rays_d,
        opts=SMOOTH | DISABLE_GAUSSIAN_TRACING | CONE_TRACING,
        envmap=_constant_envmap(),
        ray_pixel_id=pixel_id,
        ray_sample_index=sample_index,
        cone_params=make_cone_params(0.001, 1.0, 0.0, 1.0, 0),
    )
    assert torch.isfinite(out["pred_features"]).all()


@requires_gpu_ckpt
def test_cone_split_one_matches_cone_off(playground_engine):
    engine = playground_engine
    rays_o, rays_d, _ = _setup(engine)
    old = (engine.use_cone_tracing, engine.cone_split, engine.cone_max_sigma, engine.frame_id)
    try:
        engine.use_cone_tracing = False
        engine.frame_id = 13
        off = engine._render_playground_hybrid(rays_o.clone(), rays_d.clone())["pred_features"].clone()
        engine.use_cone_tracing = True
        engine.cone_split = 1.0
        engine.cone_max_sigma = 0.005
        engine.frame_id = 13
        split_one = engine._render_playground_hybrid(rays_o.clone(), rays_d.clone())["pred_features"].clone()
    finally:
        engine.use_cone_tracing, engine.cone_split, engine.cone_max_sigma, engine.frame_id = old
        if engine._gs_bvh_inflate != 0.0:
            engine._gs_bvh_inflate = 0.0
            engine.tracer.build_gs_acc(gaussians=engine.scene_mog, rebuild=True, inflate=0.0)
    assert bit_equal(off, split_one)
