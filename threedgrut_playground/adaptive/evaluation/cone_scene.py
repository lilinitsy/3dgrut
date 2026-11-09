# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cone test scene: a row of silver spheres of increasing roughness, placed where Gaussians surround but do not bury it.

The playground's cone tracing mode is meant for glossy surfaces whose reflections are mostly Gaussians. This scene puts
one metal sphere per roughness in a row inside the Gaussian cloud, so that the roughness at which cones stop paying off
(``cone_max_roughness``) can be compared by eye with the split view.

Placement (``r`` is the sphere radius):

- ``r = 0.05 * max over axes of (q95 - q05)`` of the Gaussian positions; spheres sit ``2.5 r`` apart along the scene
  frame's first horizontal axis.
- Candidates are the points of a 9 x 9 x 5 grid over the q20-q80 box of the positions. A candidate is rejected if any of
  its spheres has ``BURIED_COUNT`` or more Gaussians with opacity above ``OPACITY_MIN`` within ``1.1 r`` of its centre.
  The score of a candidate is the number of such Gaussians within ``4 r`` of the row centre (more scene to reflect).
- The camera looks at the row centre from ``8 r`` along one of four axis directions (the two along the row would make
  the spheres hide each other). Every legal (candidate, camera) pair is rendered once in cone mode at 64 x 64 with a
  constant envmap of 1 and of 0; the share of the sphere pixels' radiance that comes from the environment is the env
  fraction, and the pair with the highest ``score * (1 - env fraction)`` wins, so reflections are mostly Gaussians.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import List, NamedTuple, Sequence, Tuple

import torch

from threedgrut.utils.logger import logger

from .scenes import CameraPreset, _floor_frame, _new_primitive, _place_sphere

Tensor = torch.Tensor

RAMP_ROUGHNESS = (0.0, 0.05, 0.1, 0.15, 0.2)
SPHERE_SIZE = 0.05  # r, as a fraction of the q05-q95 extent
SPACING = 2.5  # in r
BURY_RADIUS = 1.1  # in r
SCORE_RADIUS = 4.0  # in r
CAMERA_DISTANCE = 8.0  # in r
CAMERA_TUBE = 0.5  # in r
CAMERA_FOV_DEG = 80.0  # the row is 12 r wide, seen from 8 r
OPACITY_MIN = 0.1
BURIED_COUNT = 50
GRID = (9, 9, 5)
ENV_RESOLUTION = 64
RANK_MAX_SIGMA = 0.005  # cap on the BVH inflation and cone std used while ranking placements
MIN_SPHERE_PIXELS = 0.015  # of the image, per sphere; the visibility test requires 0.01
UP = (0.0, -1.0, 0.0)  # COLMAP frame, as in GARDEN_GLOSSY
MATERIAL_BASE = "rose_gold"
SILVER = (0.9, 0.9, 0.9, 1.0)


@dataclass(frozen=True)
class RampPlan:
    """Where the ramp goes: sphere ``centers`` ``[n, 3]`` (CPU float32), radius, row centre, score and camera."""

    centers: Tensor
    radius: float
    row_center: Tensor
    score: int
    max_buried: int
    camera: CameraPreset


def ramp_material_name(roughness: float) -> str:
    return f"cone_ramp_r{roughness}"


def register_ramp_materials(engine, roughness: Sequence[float]) -> None:
    """One silver metal per roughness (base 0.9, metallic 1), cloned from ``rose_gold``."""
    materials = engine.primitives.registered_materials
    base = materials[MATERIAL_BASE]
    for value in roughness:
        name = ramp_material_name(value)
        if name not in materials:
            materials[name] = replace(
                base,
                material_id=len(materials),
                diffuse_factor=torch.tensor(SILVER, dtype=torch.float32, device=base.diffuse_factor.device),
                metallic_factor=1.0,
                roughness_factor=float(value),
            )


def _count_within(points: Tensor, center: Tensor, radius: float) -> int:
    return int(((points - center).square().sum(dim=1) <= radius * radius).sum())


def _row_centers(center: Tensor, axis: Tensor, count: int, spacing: float) -> Tensor:
    offsets = (torch.arange(count, dtype=center.dtype, device=center.device) - 0.5 * (count - 1)) * spacing
    return center[None, :] + offsets[:, None] * axis[None, :]


def _segment_count(points: Tensor, a: Tensor, b: Tensor, radius: float) -> int:
    d = b - a
    t = (((points - a) @ d) / d.dot(d)).clamp(0.0, 1.0)
    closest = a[None, :] + t[:, None] * d[None, :]
    return int(((points - closest).square().sum(dim=1) <= radius * radius).sum())


def row_score(opaque: Tensor, center: Tensor, radius: float) -> int:
    """Opaque Gaussians within ``SCORE_RADIUS * radius`` of the row centre."""
    return _count_within(opaque, center, SCORE_RADIUS * radius)


def _geometry(positions: Tensor, opacities: Tensor, count: int):
    """``(opaque, radius, e1, e2, n, legal, least_buried)``.

    ``legal`` is ``[(center, score, buried)]`` over the grid.
    """
    positions = positions.detach().float()
    opaque = positions[opacities.detach().reshape(-1) > OPACITY_MIN]
    device = positions.device
    quantiles = torch.tensor([0.05, 0.2, 0.8, 0.95], device=device)
    sample = positions[:: max(1, positions.shape[0] // 1_000_000)]  # torch.quantile is limited to 16M elements
    q05, q20, q80, q95 = torch.quantile(sample, quantiles, dim=0)
    radius = SPHERE_SIZE * float((q95 - q05).max())
    n_axis, e1, e2 = (v.to(device) for v in _floor_frame(torch.tensor(UP, dtype=torch.float32)))
    axes = [torch.linspace(float(q20[i]), float(q80[i]), GRID[i]) for i in range(3)]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3).to(device)
    legal = []
    least_buried = None
    for candidate in grid:
        spheres = _row_centers(candidate, e1, count, SPACING * radius)
        buried = max(_count_within(opaque, c, BURY_RADIUS * radius) for c in spheres)
        entry = (candidate, row_score(opaque, candidate, radius), buried)
        if least_buried is None or buried < least_buried[2]:
            least_buried = entry
        if buried < BURIED_COUNT:
            legal.append(entry)
    return opaque, radius, e1, e2, n_axis, legal, least_buried


def _camera(center: Tensor, eye: Tensor) -> CameraPreset:
    return CameraPreset(
        name="cone_ramp",
        eye=tuple(float(x) for x in eye),
        at=tuple(float(x) for x in center),
        up=UP,
        fov_deg=CAMERA_FOV_DEG,
    )


def _camera_directions(e2: Tensor) -> List[Tensor]:
    """Unit view directions to try: across the row (+-e2). The row axis would make the spheres hide each other, and the
    viewer's up axis (``neg_y_up``, the scene up) is excluded because the viewer cannot look along it."""
    return [e2, -e2]


def _plan(center, eye, score, buried, radius, e1, count) -> RampPlan:
    return RampPlan(
        centers=_row_centers(center, e1, count, SPACING * radius).cpu(),
        radius=radius,
        row_center=center.cpu(),
        score=score,
        max_buried=buried,
        camera=_camera(center, eye),
    )


def plan_cone_ramp(positions: Tensor, opacities: Tensor, roughness: Sequence[float] = RAMP_ROUGHNESS) -> RampPlan:
    """Geometry-only placement for Gaussians at ``positions`` ``[N, 3]`` with activated ``opacities`` ``[N]``.

    Highest score among the legal candidates; camera along the axis direction (not along the row) with the fewest opaque
    Gaussians within ``CAMERA_TUBE * r`` of the eye-to-target segment. :func:`place_cone_ramp` refines the choice with
    rendered reflections; this is its fallback.
    """
    opaque, radius, e1, e2, n_axis, legal, least_buried = _geometry(positions, opacities, len(roughness))
    if legal:
        center, score, buried = max(legal, key=lambda e: e[1])
    else:
        center, score, buried = least_buried
        logger.warning(
            f"cone test scene: every candidate is buried, using the least buried one ({buried} Gaussians within "
            f"{BURY_RADIUS} r of a sphere)"
        )
    options = _camera_directions(e2)
    tube = CAMERA_TUBE * radius
    eyes = [center + CAMERA_DISTANCE * radius * d for d in options]
    crossings = [_segment_count(opaque, eye, center, tube) for eye in eyes]
    eye = eyes[min(range(len(eyes)), key=crossings.__getitem__)]
    return _plan(center, eye, score, buried, radius, e1, len(roughness))


class RampCandidate(NamedTuple):
    center: Tensor
    eye: Tensor
    score: int
    env_fraction: float
    buried: int


def sphere_ownership(origins: Tensor, directions: Tensor, centers: Tensor, radius: float) -> Tensor:
    """``[n, H, W]`` bool: ray ``[H, W, 3]`` hits sphere ``k`` first among the ``centers`` ``[n, 3]`` (analytic)."""
    ts = []
    for center in centers:
        oc = origins - center.to(origins)
        b = (oc * directions).sum(dim=-1)
        disc = b * b - ((oc * oc).sum(dim=-1) - radius * radius)
        ts.append(torch.where(disc > 0, -b - disc.clamp_min(0).sqrt(), torch.full_like(b, float("inf"))))
    t = torch.stack(ts)
    return (t == t.min(dim=0).values) & torch.isfinite(t)


def _render(engine, rays_o, rays_d, env_value, pixel_id, sample_index, cone_params):
    """One cone-mode ``render_playground`` launch with a constant envmap.

    Uses the engine's stacked fields and smooth normals.
    """
    prims = engine.primitives
    prims.rebuild_bvh_if_needed()
    fields = prims.stacked_fields
    return engine.tracer.render_playground(
        gaussians=engine.scene_mog,
        ray_o=rays_o.clone(),
        ray_d=rays_d.clone(),
        playground_opts=1 | 8,  # SMOOTH_NORMALS | CONE_TRACING
        mesh_faces=fields.triangles,
        vertex_normals=fields.vertex_normals,
        vertex_tangents=fields.vertex_tangents,
        vertex_tangents_mask=fields.has_tangents,
        primitive_type=fields.primitive_type_tensor[:, None],
        frame_id=7,
        ray_max_t=None,
        material_uv=fields.material_uv,
        material_id=fields.material_id,
        materials=sorted(prims.registered_materials.values(), key=lambda mat: mat.material_id),
        is_sync_materials=True,
        refractive_index=fields.refractive_index_tensor[:, None],
        envmap=torch.full((8, 16, 4), env_value, device=rays_o.device, dtype=torch.float32),
        envmap_offset=torch.zeros(2, dtype=torch.float32),  # CPU: the launch reads it on the host
        max_pbr_bounces=engine.max_pbr_bounces,
        ray_pixel_id=pixel_id,
        ray_sample_index=sample_index,
        cone_params=cone_params,
    )


@torch.no_grad()
def env_fraction(engine, plan: RampPlan, resolution: int = ENV_RESOLUTION) -> float:
    """``mean(env1 - env0) / mean(env1)`` over the visible sphere pixels of ``plan`` in cone mode.

    The value is 1.0 if any sphere covers under ``MIN_SPHERE_PIXELS`` of the image.

    ``env1`` / ``env0`` are renders with a constant envmap of 1 / 0, so the value is the share of the sphere pixels'
    radiance that reaches the environment instead of Gaussians. Uses the engine's cone settings; the spheres of ``plan``
    must already be built in ``engine`` and the Gaussian proxies inflated.
    """
    from kaolin.render.camera import generate_pinhole_rays

    from threedgrut_playground.engine import OptixPrimitiveTypes
    from threedgrut_playground.tracer import make_cone_params

    device = engine.device
    camera = plan.camera.to_camera(resolution, resolution, device=device)
    centers = torch.arange(resolution, device=device, dtype=torch.float32) + 0.5
    ys, xs = torch.meshgrid(centers, centers, indexing="ij")
    o, d = generate_pinhole_rays(camera, coords_grid=[ys.reshape(-1), xs.reshape(-1)])
    o = o.float().reshape(1, resolution, resolution, 3).contiguous()
    d = d.float().reshape(1, resolution, resolution, 3).contiguous()
    pixel_id = torch.arange(resolution * resolution, device=device, dtype=torch.int32).reshape(
        1, resolution, resolution
    )
    sample_index = torch.zeros_like(pixel_id)
    spread = math.radians(plan.camera.fov_deg) / resolution / math.sqrt(12.0)
    max_sigma = min(engine.cone_max_sigma, RANK_MAX_SIGMA)  # matches the inflation of the BVH built by _rank
    params = make_cone_params(spread, engine.cone_roughness_scale, max_sigma, engine.cone_max_roughness, 0)

    lit = _render(engine, o, d, 1.0, pixel_id, sample_index, params)
    visible = (lit["aov_first_prim_type"][0, :, :, 0] == int(OptixPrimitiveTypes.PBR)) & (
        lit["aov_first_hit_transmittance"][0, :, :, 0] >= 0.5
    )
    # Every sphere must keep MIN_SPHERE_PIXELS of the image visible.
    owner = sphere_ownership(o[0], d[0], plan.centers, plan.radius)
    if bool(((owner & visible).flatten(1).float().mean(dim=1) < MIN_SPHERE_PIXELS).any()):
        return 1.0
    dark = _render(engine, o, d, 0.0, pixel_id, sample_index, params)
    env1 = float(lit["pred_features"][0][visible].double().mean())
    env0 = float(dark["pred_features"][0][visible].double().mean())
    return min(1.0, max(0.0, (env1 - env0) / env1)) if env1 > 0 else 1.0


@torch.no_grad()
def _rank(engine, roughness, resolution, prune):
    mog = engine.scene_mog
    _, radius, e1, e2, _, legal, _ = _geometry(mog.positions, mog.get_density().detach(), len(roughness))
    options = _camera_directions(e2)
    engine.tracer.build_gs_acc(gaussians=mog, rebuild=True, inflate=min(engine.cone_max_sigma, RANK_MAX_SIGMA))
    ranked: List[RampCandidate] = []
    plans: List[RampPlan] = []
    try:
        best_value = -1.0
        for center, score, buried in sorted(legal, key=lambda e: -e[1]) if prune else legal:
            if prune and score <= best_value:
                break
            for k, direction in enumerate(options):
                eye = center + CAMERA_DISTANCE * radius * direction
                plan = _plan(center, eye, score, buried, radius, e1, len(roughness))
                if k == 0:
                    apply_plan(engine, plan, roughness)
                env = env_fraction(engine, plan, resolution)
                best_value = max(best_value, score * (1.0 - env))
                ranked.append(RampCandidate(center, eye, score, env, buried))
                plans.append(plan)
    finally:
        engine.tracer.build_gs_acc(gaussians=mog, rebuild=True, inflate=engine._gs_bvh_inflate)
    return ranked, plans


def rank_cone_ramp_candidates(
    engine, roughness: Sequence[float] = RAMP_ROUGHNESS, resolution: int = ENV_RESOLUTION, prune: bool = False
) -> List[RampCandidate]:
    """Every legal (candidate, camera direction) pair with its score and rendered env fraction.

    Builds the spheres of each candidate in ``engine`` (the last one stays), so call :func:`apply_plan` afterwards.
    :func:`place_cone_ramp` picks the pair maximising ``score * (1 - env_fraction)``.

    With ``prune`` the candidates are visited by descending score and skipped once their score cannot beat the best
    value so far (``env_fraction >= 0``), so the list is partial but the maximiser is the same.
    """
    return _rank(engine, tuple(roughness), resolution, prune)[0]


def choose_cone_ramp(engine, roughness: Sequence[float] = RAMP_ROUGHNESS) -> Tuple[RampPlan, List[RampCandidate]]:
    """The chosen :class:`RampPlan` and the (pruned) ranking it came from (empty if no placement is legal).

    Maximises ``score * (1 - env_fraction)`` over the legal placements and cameras; the geometry-only
    :func:`plan_cone_ramp` is the fallback if none is legal. Leaves the engine's spheres in an unspecified state.
    """
    roughness = tuple(roughness)
    ranked, plans = _rank(engine, roughness, ENV_RESOLUTION, True)
    if not ranked:
        mog = engine.scene_mog
        return plan_cone_ramp(mog.positions, mog.get_density().detach(), roughness), ranked
    best = max(range(len(ranked)), key=lambda i: ranked[i].score * (1.0 - ranked[i].env_fraction))
    return plans[best], ranked


def place_cone_ramp(engine, roughness: Sequence[float] = RAMP_ROUGHNESS) -> CameraPreset:
    """Replace the engine's primitives by one silver sphere per roughness and return a camera that sees them.

    Takes about ten seconds on a mid-size scene. Logs the elapsed time and the chosen placement. The placement search
    inflates the Gaussian proxies by ``min(engine.cone_max_sigma, RANK_MAX_SIGMA)`` so a large slider value cannot make
    it block for minutes.
    """
    roughness = tuple(roughness)
    start = time.perf_counter()
    plan, ranked = choose_cone_ramp(engine, roughness)
    apply_plan(engine, plan, roughness)
    env = next((c.env_fraction for c in ranked if tuple(c.eye.tolist()) == plan.camera.eye), float("nan"))
    logger.info(
        f"cone test scene placed in {time.perf_counter() - start:.1f} s: centre "
        f"{tuple(round(float(x), 3) for x in plan.row_center)}, r {plan.radius:.3f}, score {plan.score}, "
        f"env fraction {env:.3f}"
    )
    return plan.camera


def apply_plan(engine, plan: RampPlan, roughness: Sequence[float] = RAMP_ROUGHNESS) -> None:
    """Build the spheres of ``plan`` in ``engine`` (primitives only) and switch it to the Pinhole camera type."""
    register_ramp_materials(engine, roughness)
    prims = engine.primitives
    for name in list(prims.objects):
        prims.remove_primitive(name)
    from threedgrut_playground.engine import OptixPrimitiveTypes

    for value, center in zip(roughness, plan.centers):
        sphere = _new_primitive(prims, "Sphere", engine.device)
        sphere.primitive_type = OptixPrimitiveTypes.PBR
        _place_sphere(sphere.transform, center, plan.radius)
        sphere.material_id[:] = prims.registered_materials[ramp_material_name(value)].material_id
    prims.rebuild_bvh_if_needed(force=True, rebuild=True)
    engine.invalidate_materials_on_gpu()
    engine.camera_type = "Pinhole"
