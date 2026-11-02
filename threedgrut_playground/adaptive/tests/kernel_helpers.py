# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scene, ray, trace and snapshot helpers for the playground kernel tests.

Frozen after Task 1 of the kernel-instrumentation plan: later tasks may add functions here, but must not change the
behavior of existing ones, because the default-path golden images depend on them.

Only stdlib, pytest and torch are imported at module level. The playground engine, its utils and kaolin are imported
lazily inside functions, so CPU-only tests can import this module without a GPU or the JIT-built plugin.
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import pytest
import torch

if TYPE_CHECKING:
    from threedgrut_playground.engine import Engine3DGRUT

REPO_ROOT = Path(__file__).resolve().parents[3]

SPHERE_RADIUS = 0.5
CAMERA_DISTANCE = 2.5

IMG_H = IMG_W = 64
FRAME_ID = 7
GOLDEN_SEED = 20261027

# OptixPlaygroundRenderOptions combinations
SMOOTH = 1  # SMOOTH_NORMALS
NO_GAUSS = 3  # SMOOTH_NORMALS | DISABLE_GAUSSIAN_TRACING
HARD = 0

MESH_VISIBLE_T = 0.5  # D6: a mesh pixel counts as visible when the Gaussian transmittance in front of it is >= this
MIN_VISIBLE_MESH_PIXELS = 200  # the sphere's analytic footprint is ~760 of the 4096 pixels

ENGINE_SETTINGS = (
    "frame_id",
    "camera_type",
    "camera_fov",
    "use_depth_of_field",
    "use_spp",
    "antialiasing_mode",
    "gamma_correction",
    "max_pbr_bounces",
    "use_optix_denoiser",
    "disable_gaussian_tracing",
    "is_materials_dirty",
)


#################################
##   --- Environment gates ---  ##
#################################


def resolve_ckpt() -> Optional[Path]:
    """Checkpoint from ``THREEDGRUT_TEST_CKPT`` (relative paths resolve against the repo root), or None."""
    value = os.environ.get("THREEDGRUT_TEST_CKPT")
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path if path.is_file() else None


def has_gpu_and_ckpt() -> bool:
    return torch.cuda.is_available() and resolve_ckpt() is not None


requires_gpu_ckpt = pytest.mark.skipif(not has_gpu_and_ckpt(), reason="needs CUDA and THREEDGRUT_TEST_CKPT")


#################################
##   --- Engine lifecycle ---   ##
#################################


def build_engine() -> Engine3DGRUT:
    from threedgrut_playground.engine import Engine3DGRUT

    return Engine3DGRUT(
        gs_object=str(resolve_ckpt()),
        mesh_assets_folder=str(REPO_ROOT / "threedgrut_playground" / "assets"),
        default_config="apps/colmap_3dgrt.yaml",
    )


def snapshot_engine(engine) -> dict:
    """Record the scene, settings and environment of ``engine`` so ``restore_engine`` can put them back."""
    prims = engine.primitives
    return dict(
        settings={name: getattr(engine, name) for name in ENGINE_SETTINGS},
        spp=dict(mode=engine.spp.mode, spp=engine.spp.spp),
        depth_of_field=dict(
            spp=engine.depth_of_field.spp,
            aperture_size=engine.depth_of_field.aperture_size,
            focus_z=engine.depth_of_field.focus_z,
        ),
        environment=copy.deepcopy(engine.environment),
        last_state=copy.copy(engine.last_state),
        objects=copy.deepcopy(prims.objects),
        instance_counter=dict(prims.instance_counter),
        registered_materials=dict(prims.registered_materials),
        use_smooth_normals=prims.use_smooth_normals,
        disable_pbr_textures=prims.disable_pbr_textures,
        enabled=prims.enabled,
    )


def restore_engine(engine, snap: dict) -> None:
    """Restore a ``snapshot_engine`` snapshot. ``spp`` and ``depth_of_field`` are rebuilt as fresh objects."""
    from threedgrut_playground.utils.depth_of_field import DepthOfField
    from threedgrut_playground.utils.spp import SPP

    for name, value in snap["settings"].items():
        setattr(engine, name, value)
    # Fresh objects: a restored reference would carry mutated StratifiedRayJitter generator state.
    engine.spp = SPP(mode=snap["spp"]["mode"], spp=snap["spp"]["spp"], device=engine.device)
    engine.depth_of_field = DepthOfField(
        spp=snap["depth_of_field"]["spp"],
        aperture_size=snap["depth_of_field"]["aperture_size"],
        focus_z=snap["depth_of_field"]["focus_z"],
    )
    engine.environment = copy.deepcopy(snap["environment"])
    engine.last_state = copy.copy(snap["last_state"])

    prims = engine.primitives
    prims.objects = copy.deepcopy(snap["objects"])
    prims.instance_counter = dict(snap["instance_counter"])
    prims.registered_materials = dict(snap["registered_materials"])
    prims.use_smooth_normals = snap["use_smooth_normals"]
    prims.disable_pbr_textures = snap["disable_pbr_textures"]
    prims.enabled = snap["enabled"]
    prims.dirty = True
    prims.rebuild_bvh_if_needed(force=True)


def ensure_test_materials(engine) -> None:
    """Idempotently register the alpha-blended test materials."""
    from threedgrut_playground.engine import PBRMaterial

    materials = engine.primitives.registered_materials
    for name, alpha in (("adaptive_test_blend", 0.5), ("adaptive_test_transparent", 0.0)):
        if name in materials:
            continue
        materials[name] = PBRMaterial(
            material_id=len(materials),
            diffuse_factor=torch.tensor([0.8, 0.8, 0.8, alpha], device="cuda", dtype=torch.float32),
            emissive_factor=torch.zeros(3, device="cuda", dtype=torch.float32),
            metallic_factor=0.0,
            roughness_factor=0.5,
            alpha_mode=1,  # GLTFBlend
            alpha_cutoff=0.5,
            transmission_factor=0.0,
            ior=1.0,
        )


#################################
##     --- Scene setup ---      ##
#################################


def scene_center(engine) -> torch.Tensor:
    return engine.scene_mog.positions.detach().median(dim=0).values.float()


def configure_sphere(engine, prim_type, material: str = "solid", radius: float = SPHERE_RADIUS, center=None):
    """Make the scene hold exactly one sphere of ``prim_type``/``material`` with ``radius`` at ``center``.

    Returns the sphere center (the scene center if ``center`` is None).
    """
    prims = engine.primitives
    spheres = [name for name, obj in prims.objects.items() if obj.geometry_type == "Sphere"]
    if len(spheres) == 1:
        sphere_name = spheres[0]
    else:
        before = set(prims.objects)
        prims.add_primitive("Sphere", prim_type, device=engine.device)  # returns None
        (sphere_name,) = set(prims.objects) - before
    for name in list(prims.objects):
        if name != sphere_name:
            del prims.objects[name]

    obj = prims.objects[sphere_name]
    obj.primitive_type = prim_type
    obj.material_id = torch.full_like(obj.material_id, prims.registered_materials[material].material_id)

    c = center if center is not None else scene_center(engine)
    # sphere.obj has radius 0.5 (1280 faces, face planes >= 0.4977 from the center)
    obj.transform.reset()
    obj.transform.scale(2.0 * radius)
    obj.transform.translate(c)
    prims.rebuild_bvh_if_needed(force=True)
    return c


def default_view(engine) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eye on the +x side of the scene center, at ``CAMERA_DISTANCE``.

    The +x side was chosen because the +z side is buried in Gaussians for the test checkpoint (0 visible sphere
    pixels at distance 2.5); +x shows 692 of ~708 sphere pixels and keeps a little Gaussian occlusion.
    """
    at = scene_center(engine)
    eye = at + torch.tensor([CAMERA_DISTANCE, 0.0, 0.0], device=at.device, dtype=at.dtype)
    return eye, at


#################################
##    --- Rays and maps ---     ##
#################################


def make_pinhole_rays(
    eye: torch.Tensor, at: torch.Tensor, height: int = IMG_H, width: int = IMG_W, fov_deg: float = 45.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pinhole rays through pixel centers, vertical fov. Returns ``[1,H,W,3]`` float32 CUDA origins and directions."""
    eye = eye.to(device="cuda", dtype=torch.float32)
    at = at.to(device="cuda", dtype=torch.float32)
    world_up = torch.tensor([0.0, 1.0, 0.0], device="cuda", dtype=torch.float32)
    f = torch.nn.functional.normalize(at - eye, dim=0)
    r = torch.nn.functional.normalize(torch.linalg.cross(f, world_up), dim=0)
    u = torch.linalg.cross(r, f)

    tan_half = math.tan(math.radians(fov_deg) / 2.0)
    aspect = width / height
    xs = ((torch.arange(width, device="cuda", dtype=torch.float32) + 0.5) / width * 2.0 - 1.0) * tan_half * aspect
    ys = (1.0 - (torch.arange(height, device="cuda", dtype=torch.float32) + 0.5) / height * 2.0) * tan_half
    d = f + xs[None, :, None] * r + ys[:, None, None] * u  # [H, W, 3]
    d = torch.nn.functional.normalize(d, dim=-1)

    rays_d = d.unsqueeze(0).contiguous()
    rays_o = eye.expand(1, height, width, 3).contiguous()
    return rays_o, rays_d


def gradient_envmap(height: int = 64, width: int = 128) -> torch.Tensor:
    """Smooth ``[H,W,4]`` float32 CUDA envmap with channels ``(u, v, 0.5*(1-u), 1)``."""
    u = ((torch.arange(width, device="cuda", dtype=torch.float32) + 0.5) / width).expand(height, width)
    v = ((torch.arange(height, device="cuda", dtype=torch.float32) + 0.5) / height)[:, None].expand(height, width)
    return torch.stack([u, v, 0.5 * (1.0 - u), torch.ones_like(u)], dim=-1).contiguous()


def analytic_sphere(rays_o: torch.Tensor, rays_d: torch.Tensor, center: torch.Tensor, radius: float):
    """Entry distance ``t`` (NaN on a miss) and impact parameter ``p`` of each ray w.r.t. a sphere."""
    oc = rays_o - center.to(rays_o)
    b = (oc * rays_d).sum(dim=-1)
    oc2 = (oc * oc).sum(dim=-1)
    t = -b - torch.sqrt(b * b - (oc2 - radius * radius))
    p = torch.sqrt(torch.clamp(oc2 - b * b, min=0.0))
    return t, p


def grid_ids(batch: int, height: int, width: int, sample_base: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """int32 CUDA ``[B,H,W]`` ids: ``pixel_id = y*W + x`` and ``sample_index = sample_base + b``."""
    pixel_id = torch.arange(height * width, device="cuda", dtype=torch.int32).reshape(1, height, width)
    pixel_id = pixel_id.expand(batch, height, width).contiguous()
    sample_index = (sample_base + torch.arange(batch, device="cuda", dtype=torch.int32)).reshape(batch, 1, 1)
    sample_index = sample_index.expand(batch, height, width).contiguous()
    return pixel_id, sample_index


_FLOAT_BITS = {
    torch.float64: torch.int64,
    torch.float32: torch.int32,
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
}


def bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Same shape, same dtype and identical bits (floats are compared through an integer view)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    a = a.detach().contiguous()
    b = b.detach().to(a.device).contiguous()
    if a.dtype in _FLOAT_BITS:
        a, b = a.view(_FLOAT_BITS[a.dtype]), b.view(_FLOAT_BITS[b.dtype])
    return torch.equal(a, b)


#################################
##       --- Tracing ---        ##
#################################


@torch.no_grad()
def trace_playground(
    engine,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    *,
    opts: int = SMOOTH,
    frame_id: int = FRAME_ID,
    envmap: Optional[torch.Tensor] = None,
    max_pbr_bounces: int = 15,
    ray_max_t: Optional[torch.Tensor] = None,
    ray_pixel_id: Optional[torch.Tensor] = None,
    ray_sample_index: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Call ``Tracer.render_playground`` directly, with the engine's stacked mesh fields.

    Mirrors ``Engine3DGRUT._render_playground_hybrid``. The rays are cloned because the kernel overwrites them.
    ``is_sync_materials`` has no effect: ``traceHybrid`` always re-syncs materials.
    """
    prims = engine.primitives
    prims.rebuild_bvh_if_needed()
    fields = prims.stacked_fields
    id_kwargs = {}
    if ray_pixel_id is not None:
        id_kwargs["ray_pixel_id"] = ray_pixel_id
    if ray_sample_index is not None:
        id_kwargs["ray_sample_index"] = ray_sample_index
    return engine.tracer.render_playground(
        gaussians=engine.scene_mog,
        ray_o=rays_o.clone(),
        ray_d=rays_d.clone(),
        playground_opts=opts,
        mesh_faces=fields.triangles,
        vertex_normals=fields.vertex_normals,
        vertex_tangents=fields.vertex_tangents,
        vertex_tangents_mask=fields.has_tangents,
        primitive_type=fields.primitive_type_tensor[:, None],
        frame_id=frame_id,
        ray_max_t=ray_max_t,
        material_uv=fields.material_uv,
        material_id=fields.material_id,
        materials=sorted(prims.registered_materials.values(), key=lambda mat: mat.material_id),
        is_sync_materials=True,
        refractive_index=fields.refractive_index_tensor[:, None],
        envmap=envmap,
        envmap_offset=torch.zeros(2, dtype=torch.float32),  # CPU: traceHybrid reads it on the host
        max_pbr_bounces=max_pbr_bounces,
        **id_kwargs,
    )


def visible_sphere_mask(engine, rays_o, rays_d, center, opts: int = SMOOTH) -> torch.Tensor:
    """Pixels where the analytic sphere is hit and the Gaussians in front of it leave transmittance >= MESH_VISIBLE_T.

    Uses only pre-change APIs: the sphere is moved behind the camera and the Gaussian segment is cut at the analytic
    sphere surface with ``ray_max_t``. The caller must reconfigure the sphere afterwards.
    """
    from threedgrut_playground.engine import OptixPrimitiveTypes

    t, p = analytic_sphere(rays_o, rays_d, center, SPHERE_RADIUS)
    eye = rays_o[0, 0, 0]
    away = torch.nn.functional.normalize(eye - center.to(eye), dim=0)
    configure_sphere(engine, OptixPrimitiveTypes.MIRROR, center=eye + 1000.0 * away)
    ray_max_t = torch.where(p < SPHERE_RADIUS, t, torch.full_like(t, 1e9)).float().contiguous()
    out = trace_playground(engine, rays_o, rays_d, opts=opts, ray_max_t=ray_max_t)
    transmittance = 1.0 - out["pred_opacity"][..., 0]
    return (p < 0.95 * SPHERE_RADIUS) & (transmittance >= MESH_VISIBLE_T)


#################################
##     --- Golden cases ---     ##
#################################


@dataclass(frozen=True)
class GoldenCase:
    name: str
    kind: str  # "tracer" | "engine"
    prim: str  # OptixPrimitiveTypes member name
    material: str
    opts: int = SMOOTH  # tracer only
    gaussians: bool = True  # engine only
    smooth: bool = True  # engine only


def has_gaussians(case: GoldenCase) -> bool:
    if case.kind == "engine":
        return case.gaussians
    return not (case.opts & 2)


GOLDEN_CASES = [
    GoldenCase("mirror_smooth", "tracer", "MIRROR", "solid", opts=SMOOTH),
    GoldenCase("mirror_hard", "tracer", "MIRROR", "solid", opts=HARD),
    GoldenCase("glass_smooth", "tracer", "GLASS", "solid", opts=SMOOTH),
    GoldenCase("glass_nogauss", "tracer", "GLASS", "solid", opts=NO_GAUSS),
    GoldenCase("diffuse_smooth", "tracer", "DIFFUSE", "solid", opts=SMOOTH),
    GoldenCase("pbr_copper_smooth", "tracer", "PBR", "brushed_copper", opts=SMOOTH),
    GoldenCase("pbr_copper_hard", "tracer", "PBR", "brushed_copper", opts=HARD),
    GoldenCase("pbr_copper_nogauss", "tracer", "PBR", "brushed_copper", opts=NO_GAUSS),
    GoldenCase("pbr_blend_smooth", "tracer", "PBR", "adaptive_test_blend", opts=SMOOTH),
    GoldenCase("pbr_blend_nogauss", "tracer", "PBR", "adaptive_test_blend", opts=NO_GAUSS),
    GoldenCase("engine_pbr", "engine", "PBR", "brushed_copper"),
    GoldenCase("engine_pbr_nogauss", "engine", "PBR", "brushed_copper", gaussians=False),
    GoldenCase("engine_blend_hard", "engine", "PBR", "adaptive_test_blend", smooth=False),
]


def visible_mesh_pixel_count(engine, case: GoldenCase) -> int:
    """Number of sphere pixels visible through the Gaussians in the default view of ``case`` (engine is restored)."""
    from threedgrut_playground.engine import OptixPrimitiveTypes

    snap = snapshot_engine(engine)
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(GOLDEN_SEED)
            ensure_test_materials(engine)  # the case material may be a test material; restored afterwards
            center = configure_sphere(engine, OptixPrimitiveTypes[case.prim], case.material)
            o, d = make_pinhole_rays(*default_view(engine))
            return int(visible_sphere_mask(engine, o, d, center).sum())
    finally:
        restore_engine(engine, snap)


def render_golden_case(engine, case: GoldenCase) -> Dict[str, torch.Tensor]:
    """Render ``case`` on ``engine`` with fixed seeds and settings; returns CPU clones. The engine is restored."""
    from kaolin.render.camera import Camera

    from threedgrut_playground.engine import OptixPrimitiveTypes
    from threedgrut_playground.utils.spp import SPP

    snap = snapshot_engine(engine)
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(GOLDEN_SEED)
            ensure_test_materials(engine)
            configure_sphere(engine, OptixPrimitiveTypes[case.prim], case.material)
            eye, at = default_view(engine)
            if case.kind == "tracer":
                o, d = make_pinhole_rays(eye, at)
                o, d = torch.cat([o, o]), torch.cat([d, d])  # B=2 exercises idx.z in getFrameNumber
                out = trace_playground(
                    engine,
                    o,
                    d,
                    opts=case.opts,
                    frame_id=FRAME_ID,
                    envmap=gradient_envmap(),
                    max_pbr_bounces=15,
                )
                keys = ("pred_features", "pred_opacity", "last_ray_o", "last_ray_d", "hits_count")
            else:
                engine.spp = SPP(mode="msaa", spp=4, device=engine.device)  # fresh StratifiedRayJitter state
                engine.frame_id, engine.use_spp, engine.antialiasing_mode = 0, True, "4x MSAA"
                engine.use_optix_denoiser, engine.use_depth_of_field, engine.camera_type = False, False, "Pinhole"
                engine.gamma_correction, engine.max_pbr_bounces = 1.0, 15
                engine.disable_gaussian_tracing = not case.gaussians
                engine.primitives.use_smooth_normals = case.smooth
                camera = Camera.from_args(
                    eye=eye.cpu(),
                    at=at.cpu(),
                    up=torch.tensor([0.0, 1.0, 0.0]),
                    fov=math.radians(45.0),
                    width=IMG_W,
                    height=IMG_H,
                    near=0.01,
                    far=1000.0,
                    dtype=torch.float32,
                    device="cpu",
                )  # render_pass moves it to CUDA
                with torch.no_grad():
                    out = engine.render(camera)
                keys = ("rgb", "opacity", "rgb_buffer")
            return {k: out[k].detach().cpu().clone() for k in keys}
    finally:
        restore_engine(engine, snap)
