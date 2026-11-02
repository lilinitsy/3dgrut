# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Glossy test scene: the ``garden_glossy`` preset, its layout math, ``setup_glossy_scene`` and the D6 visibility check.

All preset lengths are in units of the scene *spread* (the largest per-axis median absolute deviation of the Gaussian
positions around their median), so a preset scales with the checkpoint. The layout is computed on the CPU in the dtype
of the scene frame; :func:`build_glossy_scene` moves it to the engine device.

The visibility check (Review Focus 4) traces jittered samples through the core's public launch path at the dedicated
sample-index range ``[VISIBILITY_SAMPLE_BASE, 2^31)`` and counts, per object, the pixels with at least one D6-visible
mesh hit on it. A preset whose meshes are in view but hidden behind Gaussians raises :class:`SceneVisibilityError`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

import torch

from .sampling import (
    trace_only_renderer,
    traced_rounds,
    union_by_pixel,
    visible_mesh_hits,
)

Tensor = torch.Tensor
Vec3 = Tuple[float, float, float]

# Sample indices of the visibility check: [2^31 - 2^20, 2^31) (disjoint from test renders and the reference).
VISIBILITY_SAMPLE_BASE = 2**31 - 2**20
VISIBILITY_KIND = "scene_visibility"

FLOOR_NAME = "floor"


# ---------------------------------------------------------------------------------------------------------------------
# preset
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SphereSpec:
    """A sphere resting on the floor: ``material`` name, in-plane coordinates ``(a, b)`` along the floor's ``(e1, e2)``
    axes and ``radius``, all in units of spread."""

    material: str
    a: float
    b: float
    radius: float


@dataclass(frozen=True)
class GlossyPreset:
    """A glossy test scene, in units of spread.

    Attributes:
        up: world up direction; the floor normal is ``normalize(up)``.
        floor_material: registered material of the floor quad.
        floor_height: floor center offset from the scene center along the floor normal.
        floor_half_extent: half the side of the square floor quad.
        spheres: spheres resting on the floor.
        camera_offsets: ``{camera name: eye - at}`` in world axes (units of spread).
        fov_deg: vertical field of view of every camera.
        camera_target_height: camera target offset from the scene center along the floor normal (independent of
            ``floor_height`` and ``mesh_shift``).
        min_visible_fraction: ``{camera name: minimum fraction of pixels with a D6-visible mesh hit}``.
        min_object_fraction: minimum visible pixel fraction of every object in every camera.
        mesh_shift: added to every mesh (never to the cameras).
    """

    up: Vec3
    floor_material: str
    floor_height: float
    floor_half_extent: float
    spheres: Tuple[SphereSpec, ...]
    camera_offsets: Dict[str, Vec3]
    fov_deg: float
    camera_target_height: float
    min_visible_fraction: Dict[str, float]
    min_object_fraction: float
    mesh_shift: Vec3 = (0.0, 0.0, 0.0)


GARDEN_GLOSSY = GlossyPreset(
    up=(0.0, -1.0, 0.0),  # garden COLMAP frame: world -y is up
    floor_material="brushed_copper",  # metallic 1.0, roughness 0.5
    floor_height=0.05,
    floor_half_extent=1.0,
    spheres=(
        SphereSpec("rose_gold", -0.6, 0.3, 0.13),  # metallic 1.0, roughness 0.15
        SphereSpec("polished_marble", -0.2, 0.3, 0.13),  # metallic 0.0, roughness 0.1
        SphereSpec("blue_plastic", 0.2, 0.3, 0.13),  # metallic 0.0, roughness 0.4
        SphereSpec("oak_wood", 0.6, 0.3, 0.13),  # metallic 0.0, roughness 0.75
    ),
    camera_offsets={"close": (0.0, -1.0, -1.5), "overview": (0.0, -1.0, -3.0)},
    fov_deg=60.0,  # ppd 8.53 at 512 px (D14)
    camera_target_height=0.05,
    min_visible_fraction={"close": 0.25, "overview": 0.05},
    min_object_fraction=0.001,
)

PRESETS: Dict[str, GlossyPreset] = {"garden_glossy": GARDEN_GLOSSY}


@dataclass(frozen=True)
class SceneFrame:
    """Scene center ``[3]`` and spread (a Python float)."""

    center: Tensor
    spread: float


def scene_frame(positions: Tensor) -> SceneFrame:
    """``center = median(positions)``, ``spread = max over axes of median |positions - center|``."""
    positions = positions.detach()
    center = positions.median(dim=0).values
    spread = float((positions - center).abs().median(dim=0).values.max())
    return SceneFrame(center=center, spread=spread)


# ---------------------------------------------------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraPreset:
    """A pinhole camera: ``eye``, ``at``, ``up`` as world-space tuples and the vertical FOV in degrees."""

    name: str
    eye: Vec3
    at: Vec3
    up: Vec3
    fov_deg: float

    def to_camera(self, width: int, height: int, device="cuda"):
        """A kaolin pinhole ``Camera`` (float32) of ``width x height`` pixels on ``device``."""
        from kaolin.render.camera import Camera

        return Camera.from_args(
            eye=torch.tensor(self.eye, dtype=torch.float32),
            at=torch.tensor(self.at, dtype=torch.float32),
            up=torch.tensor(self.up, dtype=torch.float32),
            fov=math.radians(self.fov_deg),
            width=int(width),
            height=int(height),
            dtype=torch.float32,
            device=device,
        )

    def forward(self) -> Tensor:
        """Unit view direction ``normalize(at - eye)`` ``[3]`` float32."""
        d = torch.tensor(self.at, dtype=torch.float32) - torch.tensor(self.eye, dtype=torch.float32)
        return d / d.norm()


@dataclass
class GlossyLayout:
    """World-space geometry of a preset (CPU tensors).

    Attributes:
        floor_vertices: ``[4, 3]`` quad corners in ``create_quad_mesh`` order (winding normal ``+floor_normal``).
        floor_normal: ``[3]`` unit floor normal.
        floor_material: registered material of the floor.
        spheres: ``(material, center [3], world radius)`` per sphere, in preset order (``sphere_1..``).
        cameras: ``{name: CameraPreset}``.
    """

    floor_vertices: Tensor
    floor_normal: Tensor
    floor_material: str
    spheres: List[Tuple[str, Tensor, float]]
    cameras: Dict[str, CameraPreset] = field(default_factory=dict)

    def object_names(self) -> List[str]:
        """``["floor", "sphere_1", ...]``: object ``k`` of :func:`assign_hits_to_objects` is ``object_names()[k]``."""
        return [FLOOR_NAME] + [f"sphere_{i + 1}" for i in range(len(self.spheres))]


def _floor_frame(up: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """``(n, e1, e2)``: ``n = normalize(up)``, ``e1`` = x-hat projected onto the plane (z-hat if degenerate),
    ``e2 = e1 x n``."""
    n = up / up.norm()
    e1 = None
    for axis in (0, 2):
        a = torch.zeros(3, dtype=up.dtype)
        a[axis] = 1.0
        cand = a - (a @ n) * n
        if float(cand.norm()) > 1e-6:
            e1 = cand / cand.norm()
            break
    if e1 is None:  # unreachable for a unit n: x-hat and z-hat cannot both be parallel to it
        raise ValueError(f"cannot build a floor frame for up={up.tolist()}")
    e2 = torch.linalg.cross(e1, n)
    return n, e1, e2


def floor_quad_vertices(center: Tensor, normal: Tensor, e1: Tensor, half_extent: float) -> Tensor:
    """``[4, 3]`` corners ``c-h e1-h e2, c-h e1+h e2, c+h e1-h e2, c+h e1+h e2`` with ``e2 = e1 x normal``.

    Same vertex order as ``create_quad_mesh`` (faces ``[0,1,2], [2,1,3]``), so the winding normal is ``+normal``.
    """
    e2 = torch.linalg.cross(e1, normal)
    h = float(half_extent)
    return torch.stack(
        [
            center - h * e1 - h * e2,
            center - h * e1 + h * e2,
            center + h * e1 - h * e2,
            center + h * e1 + h * e2,
        ]
    )


def glossy_layout(frame: SceneFrame, preset: GlossyPreset) -> GlossyLayout:
    """World-space layout of ``preset`` in ``frame`` (CPU, in the dtype of ``frame.center``)."""
    center = frame.center.detach().cpu()
    dtype = center.dtype if center.dtype.is_floating_point else torch.float32
    center = center.to(dtype)
    s = float(frame.spread)
    n, e1, e2 = _floor_frame(torch.tensor(preset.up, dtype=dtype))
    shift = torch.tensor(preset.mesh_shift, dtype=dtype)

    floor_center = center + preset.floor_height * s * n + shift * s
    floor_vertices = floor_quad_vertices(floor_center, n, e1, preset.floor_half_extent * s)
    spheres = [
        (sp.material, floor_center + s * (sp.a * e1 + sp.b * e2 + sp.radius * n), sp.radius * s)
        for sp in preset.spheres
    ]

    at = center + preset.camera_target_height * s * n
    at_t = tuple(float(x) for x in at)
    up_t = tuple(float(x) for x in preset.up)
    cameras = {}
    for name, offset in preset.camera_offsets.items():
        eye = at + s * torch.tensor(offset, dtype=dtype)
        cameras[name] = CameraPreset(
            name=name, eye=tuple(float(x) for x in eye), at=at_t, up=up_t, fov_deg=float(preset.fov_deg)
        )
    return GlossyLayout(
        floor_vertices=floor_vertices,
        floor_normal=n,
        floor_material=preset.floor_material,
        spheres=spheres,
        cameras=cameras,
    )


def assign_hits_to_objects(points: Tensor, layout: GlossyLayout, tol: float = 0.02) -> Tensor:
    """Object index ``[N]`` int64 of each hit point ``[N, 3]``: ``0`` floor, ``1..`` spheres, ``-1`` none.

    Residuals are relative to each object's size: ``| |p - c| - r | / r`` for a sphere (the tessellated sphere's face
    planes lie within 0.5% of its radius) and ``|(p - c0) . n| / h`` for the floor (``h`` = half-extent, only for points
    whose in-plane offset is within ``h (1 + tol)`` of the floor center on both axes). A point goes to the object with
    the smallest residual, if that residual is ``<= tol``.
    """
    device = points.device
    p = points.detach().to(torch.float64)
    n = layout.floor_normal.to(device, torch.float64)
    verts = layout.floor_vertices.to(device, torch.float64)
    c0 = verts.mean(dim=0)
    half = 0.5 * float((verts[2] - verts[0]).norm())  # v2 - v0 = 2 h e1
    e1 = (verts[2] - verts[0]) / (2.0 * half)
    e2 = (verts[1] - verts[0]) / (2.0 * half)

    residuals = []
    d = p - c0
    inplane = torch.maximum((d @ e1).abs(), (d @ e2).abs())
    floor_res = (d @ n).abs() / half
    floor_res = torch.where(inplane <= half * (1.0 + tol), floor_res, torch.full_like(floor_res, math.inf))
    residuals.append(floor_res)
    for _, c, r in layout.spheres:
        c = c.to(device, torch.float64)
        residuals.append(((p - c).norm(dim=1) - float(r)).abs() / float(r))

    res = torch.stack(residuals, dim=1)  # [N, K]
    best, idx = res.min(dim=1)
    return torch.where(best <= tol, idx, torch.full_like(idx, -1)).to(torch.int64)


# ---------------------------------------------------------------------------------------------------------------------
# visibility (D6)
# ---------------------------------------------------------------------------------------------------------------------


class SceneVisibilityError(RuntimeError):
    """A preset's meshes are not visible enough (D6): ``camera``, ``object`` (an object name or ``"total"``) and the
    measured ``fraction``."""

    def __init__(self, camera: str, obj: str, fraction: float, threshold: float, coverage=None):
        self.camera = camera
        self.object = obj
        self.fraction = float(fraction)
        self.threshold = float(threshold)
        self.coverage = coverage
        msg = (
            f"scene visibility check failed: camera {camera!r}, {obj!r} covers {self.fraction:.4f} of the pixels "
            f"(minimum {self.threshold:g})"
        )
        if coverage is not None:
            msg += f"; measured coverage: {_fmt_coverage(coverage)}"
        super().__init__(msg)


def _fmt_coverage(coverage: Dict[str, Dict[str, float]]) -> str:
    return "; ".join(f"{cam}: " + ", ".join(f"{k}={v:.4f}" for k, v in per.items()) for cam, per in coverage.items())


@torch.no_grad()
def visible_coverage(engine, layout: GlossyLayout, camera, spp: int = 4) -> Dict[str, float]:
    """Pixel fractions of D6-visible mesh hits per object (``floor``, ``sphere_1..``) and ``total``.

    Every pixel gets ``spp`` jittered samples at sample indices ``VISIBILITY_SAMPLE_BASE + [0, spp)``. A pixel counts
    for an object if at least one of its D6-visible hits (``origin + aov_first_hit_t * direction``) is assigned to that
    object by :func:`assign_hits_to_objects`; it counts for ``total`` if it has at least one D6-visible mesh hit (the
    core's mesh-mask rule).
    """
    spp = int(spp)
    if spp < 1:
        raise ValueError(f"spp must be >= 1, got {spp}")
    device = torch.device(engine.device)
    camera = camera.to(device)
    n_pixels = int(camera.height) * int(camera.width)
    names = layout.object_names()
    renderer = trace_only_renderer(engine)
    counts = torch.full((n_pixels,), spp, dtype=torch.int64, device=device)
    base = torch.full((n_pixels,), VISIBILITY_SAMPLE_BASE, dtype=torch.int64, device=device)
    total = torch.zeros(n_pixels, dtype=torch.bool, device=device)
    per_obj = torch.zeros(len(names), n_pixels, dtype=torch.bool, device=device)
    for batch, out in traced_rounds(renderer, camera, counts, base, kind=VISIBILITY_KIND):
        vis = visible_mesh_hits(out).to(device)
        pid = batch.pixel_id.to(device).long()
        total |= union_by_pixel(pid, vis, n_pixels)
        if not bool(vis.any()):
            continue
        t = out["aov_first_hit_t"][:, 0].to(device)[vis]
        pts = batch.origins.to(device)[vis] + t[:, None] * batch.directions.to(device)[vis]
        obj = assign_hits_to_objects(pts, layout).to(device)
        pid_vis = pid[vis]
        for k in range(len(names)):
            sel = obj == k
            if bool(sel.any()):
                per_obj[k, pid_vis[sel]] = True
    cov = {name: float(per_obj[k].float().mean()) for k, name in enumerate(names)}
    cov["total"] = float(total.float().mean())
    return cov


def verify_scene_visibility(
    engine, layout: GlossyLayout, preset: GlossyPreset, resolution: int = 128
) -> Dict[str, Dict[str, float]]:
    """Measure :func:`visible_coverage` for every camera of ``layout`` at ``resolution x resolution`` pixels.

    Returns ``{camera: coverage}``. Raises :class:`SceneVisibilityError` (all coverage in the message) if a camera's
    ``total < preset.min_visible_fraction[camera]`` or any object covers ``< preset.min_object_fraction`` of a camera.
    """
    coverage: Dict[str, Dict[str, float]] = {}
    for name, cam in layout.cameras.items():
        camera = cam.to_camera(resolution, resolution, device=engine.device)
        coverage[name] = visible_coverage(engine, layout, camera)
    for name, cov in coverage.items():
        min_total = float(preset.min_visible_fraction[name])
        if cov["total"] < min_total:
            raise SceneVisibilityError(name, "total", cov["total"], min_total, coverage)
        for obj in layout.object_names():
            if cov[obj] < preset.min_object_fraction:
                raise SceneVisibilityError(name, obj, cov[obj], preset.min_object_fraction, coverage)
    return coverage


# ---------------------------------------------------------------------------------------------------------------------
# scene construction
# ---------------------------------------------------------------------------------------------------------------------


def _place_sphere(transform, center: Tensor, radius: float) -> None:
    """Unit-diameter ``sphere.obj`` (radius 0.5) scaled to ``radius`` and centered at ``center``.

    ``translate`` adds in place, so the (CPU, possibly float64) center is converted to the transform's device/dtype.
    """
    transform.reset()
    transform.scale(2.0 * float(radius))
    transform.translate(center.to(device=transform._translation.device, dtype=transform._translation.dtype))


def _new_primitive(prims, geometry_type: str, device):
    from threedgrut_playground.engine import OptixPrimitiveTypes

    prims.add_primitive(geometry_type, OptixPrimitiveTypes.PBR, device=device)
    return prims.objects[f"{geometry_type} {prims.instance_counter[geometry_type]}"]


def build_glossy_scene(engine, layout: GlossyLayout) -> None:
    """Replace every primitive by the PBR floor quad and spheres of ``layout`` (primitives only)."""
    prims = engine.primitives
    materials = [layout.floor_material] + [m for m, _, _ in layout.spheres]
    missing = sorted({m for m in materials if m not in prims.registered_materials})
    if missing:
        raise ValueError(f"materials not registered: {missing}")
    device = engine.device
    for name in list(prims.objects):
        prims.remove_primitive(name)

    floor = _new_primitive(prims, "Quad", device)
    floor.transform.reset()
    floor.vertices = layout.floor_vertices.to(device=device, dtype=torch.float32).contiguous()
    floor.vertex_normals = (
        layout.floor_normal.to(device=device, dtype=torch.float32).reshape(1, 3).expand(4, 3).contiguous()
    )
    floor.material_id[:] = prims.registered_materials[layout.floor_material].material_id

    for material, center, radius in layout.spheres:
        obj = _new_primitive(prims, "Sphere", device)
        _place_sphere(obj.transform, center, radius)
        obj.material_id[:] = prims.registered_materials[material].material_id

    prims.rebuild_bvh_if_needed(force=True, rebuild=True)
    engine.invalidate_materials_on_gpu()


def setup_glossy_scene(
    engine,
    preset: Union[str, GlossyPreset] = "garden_glossy",
    *,
    verify: bool = True,
    verify_resolution: int = 128,
) -> Dict[str, CameraPreset]:
    """Build ``preset`` in ``engine`` (Pinhole camera type) and return its cameras.

    With ``verify`` the D6 visibility check runs and raises :class:`SceneVisibilityError` on failure.
    """
    p = PRESETS[preset] if isinstance(preset, str) else preset
    engine.camera_type = "Pinhole"
    layout = glossy_layout(scene_frame(engine.scene_mog.positions.detach().cpu()), p)
    build_glossy_scene(engine, layout)
    if verify:
        verify_scene_visibility(engine, layout, p, resolution=verify_resolution)
    return dict(layout.cameras)
