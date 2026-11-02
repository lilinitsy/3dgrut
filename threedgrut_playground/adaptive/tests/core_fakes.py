# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU fake of the playground engine for the adaptive-core tests (shared by ``test_core_trace`` / ``test_core_render``).

``FakeTracer.render_playground`` checks interface A's identity contract (int32 ids of the launch shape, origins at the
camera eye, each direction inside pixel ``pixel_id``), derives the scene from the ray *direction* (not from the ids) and
draws its randomness only from the ids, as the kernel does. Ray/pixel mapping errors therefore fail loudly.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Optional

import torch
from kaolin.render.camera import Camera, CameraFOV

from threedgrut_playground.adaptive.allocate import PerceptualVarianceAllocator
from threedgrut_playground.adaptive.budget import SimpleBudget
from threedgrut_playground.adaptive.estimators import VarianceOfMeanEstimator
from threedgrut_playground.adaptive.rays import JITTER_HASH_X, JITTER_HASH_Y
from threedgrut_playground.adaptive.renderer import AdaptiveRenderer
from threedgrut_playground.adaptive.rounding import StochasticRounder
from threedgrut_playground.adaptive.telemetry import CostTelemetry
from threedgrut_playground.adaptive.weights import (
    ContrastMaskingTerm,
    DisplayClipTerm,
    LuminanceAdaptationTerm,
)
from threedgrut_playground.utils.rng import rng_torch_low_discrepancy

_M32 = 0xFFFFFFFF
PRIM_MESH = 4  # any nonzero primitive type is a mesh hit
SCENES = ("disk", "bernoulli", "edge")


def make_camera(size: int = 16, device: str = "cpu") -> Camera:
    """``size x size`` pinhole camera at ``eye=(0, 0, 5)`` looking at the origin (identity world->camera rotation)."""
    return Camera.from_args(
        eye=torch.tensor([0.0, 0.0, 5.0]),
        at=torch.zeros(3),
        up=torch.tensor([0.0, 1.0, 0.0]),
        fov=math.radians(45),
        width=size,
        height=size,
        device=device,
    )


def hash01(pid: torch.Tensor, sidx: torch.Tensor) -> torch.Tensor:
    """32-bit avalanche hash of ``(pid, sidx)`` -> float64 in ``[0, 1)``, computed in int64, masked before shifts."""
    h = (pid.long() * 0x9E3779B1 + sidx.long() * 0x85EBCA77) & _M32
    h = h ^ (h >> 15)
    h = (h * 0x2C1B3C6D) & _M32
    h = h ^ (h >> 12)
    h = (h * 0x297A2D39) & _M32
    h = h ^ (h >> 15)
    return h.double() / 2.0**32


def parity_coupled_jitter(pixel_id: torch.Tensor, sample_index: torch.Tensor, width: int) -> torch.Tensor:
    """Pre-D12 sampler (test-only): one Owen-scrambled Sobol sequence per pixel at index ``sample_index``.

    Even and odd samples are interleaved points of the *same* (0,2)-sequence, so the two parity halves are coupled.
    Same signature and output as :func:`threedgrut_playground.adaptive.rays.pixel_jitter`; it exists only to show that
    the sub-pixel edge test detects that coupling.
    """
    pid = pixel_id.long()
    px, py = pid % width, pid // width
    seed = (px * JITTER_HASH_X + py * JITTER_HASH_Y) & 0xFFFFFFFF
    u, v = rng_torch_low_discrepancy(sample_index.long(), seed)
    return torch.stack([u, v], -1).float().clamp_(0.0, 1.0 - 2.0**-24)


def project_to_pixel(camera: Camera, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """World directions ``[N, 3]`` -> continuous pixel coordinates ``(x, y)`` (inverse of kaolin's pinhole map)."""
    c = d.double() @ camera.extrinsics.R[0].double().cpu().T  # world -> camera (identity for make_camera)
    tan_h = float(camera.tan_half_fov(CameraFOV.HORIZONTAL))
    tan_v = float(camera.tan_half_fov(CameraFOV.VERTICAL))
    z = -c[:, 2]
    x = (c[:, 0] / z / tan_h + 1.0) * int(camera.width) / 2.0
    y = (-c[:, 1] / z / tan_v + 1.0) * int(camera.height) / 2.0
    return x, y


class FakeTracer:
    """Stands in for ``Tracer.render_playground`` (interface A).

    ``samples`` collects ``(pixel_id, sample_index, x, y)`` per launched ray (padding duplicates included, since the
    kernel sees them too); ``calls`` collects ``dict(frame_id, is_sync_materials, launch_shape)`` per launch.
    """

    def __init__(
        self,
        camera: Camera,
        *,
        scene: str,
        mesh_radius: float,
        hidden_radius: float,
        mesh_level: float,
        noise_amp: float,
        edge_x: float,
        nan_pixel: Optional[int],
        nan_sample: int,
        firefly_pixel: Optional[int] = None,
        firefly_sample: int = 0,
        firefly_value: float = 50.0,
    ):
        if scene not in SCENES:
            raise ValueError(f"scene must be one of {SCENES}, got {scene!r}")
        self.camera = camera
        self.width, self.height = int(camera.width), int(camera.height)
        self.eye = camera.cam_pos().reshape(3).double().cpu()
        self.scene = scene
        self.mesh_radius = mesh_radius
        self.hidden_radius = hidden_radius
        self.mesh_level = mesh_level
        self.noise_amp = noise_amp
        self.edge_x = edge_x
        self.nan_pixel = nan_pixel
        self.nan_sample = nan_sample
        self.firefly_pixel = firefly_pixel
        self.firefly_sample = firefly_sample
        self.firefly_value = firefly_value
        self.samples: list[tuple[int, int, float, float]] = []
        self.calls: list[dict] = []

    def render_playground(self, **kw) -> dict[str, torch.Tensor]:
        ray_o, ray_d = kw["ray_o"], kw["ray_d"]
        pid_l, sidx_l = kw["ray_pixel_id"], kw["ray_sample_index"]
        launch = tuple(ray_o.shape[:3])

        # 1. Identity contract.
        assert pid_l is not None and sidx_l is not None, "ray ids are required"
        assert pid_l.dtype == torch.int32 and sidx_l.dtype == torch.int32, "ray ids must be int32"
        assert tuple(pid_l.shape) == launch and tuple(sidx_l.shape) == launch, "ray ids must have shape ray_o[:3]"
        o = ray_o.reshape(-1, 3).double().cpu()
        d = ray_d.reshape(-1, 3).double().cpu()
        pid = pid_l.reshape(-1).long().cpu()
        sidx = sidx_l.reshape(-1).long().cpu()
        assert torch.allclose(o, self.eye.expand_as(o), atol=1e-5, rtol=0), "ray origins must be the camera eye"
        x, y = project_to_pixel(self.camera, d)
        px, py = (pid % self.width).double(), (pid // self.width).double()
        eps = 1e-3
        inside = (x >= px - eps) & (x <= px + 1 + eps) & (y >= py - eps) & (y <= py + 1 + eps)
        assert bool(inside.all()), f"{int((~inside).sum())} rays lie outside their pixel_id"
        self.samples.extend(zip(pid.tolist(), sidx.tolist(), x.tolist(), y.tolist()))
        self.calls.append(dict(frame_id=kw["frame_id"], is_sync_materials=kw["is_sync_materials"], launch_shape=launch))

        # 2.-3. Scene from the direction, randomness from the ids.
        m = pid.numel()
        u = hash01(pid, sidx)
        ones = torch.ones(m, dtype=torch.float64)
        if self.scene == "disk":
            r = torch.hypot(x - self.width / 2.0, y - self.height / 2.0)
            vis = r < self.mesh_radius
            hidden = ~vis & (r < self.hidden_radius)
            prim = torch.where(vis | hidden, PRIM_MESH, 0)
            trans = torch.where(vis, 1.0, torch.where(hidden, 0.3, 0.0))
            value = torch.where(vis, self.mesh_level + self.noise_amp * 2.0 * (u - 0.5), 0.3 * ones)
            opacity = torch.where(vis, 1.0, 0.5 * ones)
        else:
            prim = torch.full((m,), PRIM_MESH, dtype=torch.int64)
            trans = ones.clone()
            opacity = ones.clone()
            if self.scene == "bernoulli":
                value = 10.0 * (u < 0.1).double()
            else:  # "edge": vertical sub-pixel edge, true pixel mean edge_x
                value = ((x - x.floor()) < self.edge_x).double()
        rgb = value[:, None].expand(m, 3).clone()
        if self.nan_pixel is not None:
            rgb[(pid == self.nan_pixel) & (sidx == self.nan_sample)] = float("nan")
        if self.firefly_pixel is not None:
            rgb[(pid == self.firefly_pixel) & (sidx == self.firefly_sample)] = self.firefly_value

        # 4. Other AOVs.
        hit = prim != 0
        normal = torch.zeros(m, 3, dtype=torch.float64)
        normal[hit, 2] = 1.0

        dev = ray_o.device

        def shaped(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
            t = t.reshape(m, -1)
            return t.to(device=dev, dtype=dtype).reshape(*launch, t.shape[1]).contiguous()

        out = {
            "pred_features": shaped(rgb, torch.float32),
            "pred_opacity": shaped(opacity, torch.float32),
            "aov_first_hit_t": shaped(torch.where(hit, 1.0, -1.0 * ones), torch.float32),
            "aov_first_normal": shaped(normal, torch.float32),
            "aov_first_prim_type": shaped(prim, torch.int32),
            "aov_first_hit_transmittance": shaped(trans, torch.float32),
            "aov_num_bounces": shaped(hit.long(), torch.int32),
            "aov_gauss_depth": shaped(2.0 * ones, torch.float32),
        }

        # 5. The kernel overwrites its input rays.
        ray_o.fill_(float("nan"))
        ray_d.fill_(float("nan"))
        out["last_ray_o"] = ray_o
        out["last_ray_d"] = ray_d
        return out


def make_fake_engine(
    camera: Camera,
    *,
    scene: str = "disk",
    mesh_radius: float = 5.0,
    hidden_radius: float = 0.0,
    mesh_level: float = 0.6,
    noise_amp: float = 0.5,
    edge_x: float = 0.3,
    nan_pixel: Optional[int] = None,
    nan_sample: int = 1,
    firefly_pixel: Optional[int] = None,
    firefly_sample: int = 0,
    firefly_value: float = 50.0,
    ignore_envmap: bool = False,
    visible: bool = True,
    enabled: bool = True,
    camera_type: str = "Pinhole",
) -> SimpleNamespace:
    """Fake ``Engine3DGRUT`` exposing exactly the attributes :class:`AdaptiveRenderer` reads (see module docstring)."""
    stacked_fields = SimpleNamespace(
        triangles=torch.zeros(1, 3, dtype=torch.int32),
        vertex_normals=torch.zeros(1, 3),
        vertex_tangents=torch.zeros(1, 3),
        has_tangents=torch.zeros(1, dtype=torch.bool),
        primitive_type_tensor=torch.zeros(1, dtype=torch.int32),
        material_uv=torch.zeros(1, 2),
        material_id=torch.zeros(1, dtype=torch.int32),
        refractive_index_tensor=torch.ones(1),
    )
    primitives = SimpleNamespace(
        enabled=enabled,
        use_smooth_normals=False,
        disable_pbr_textures=False,
        has_visible_objects=lambda: visible,
        rebuild_bvh_if_needed=lambda: None,
        stacked_fields=stacked_fields,
        registered_materials={},
    )
    environment = SimpleNamespace(
        get_envmap=lambda: torch.zeros(4, 4, 4) if ignore_envmap else None,
        get_envmap_offset=lambda: torch.zeros(2),
        is_ignore_envmap=lambda: ignore_envmap,
    )
    scene_mog = SimpleNamespace(background=lambda p, d, rgb, op, train: (rgb + 0.2 * (1.0 - op), op))
    tracer = FakeTracer(
        camera,
        scene=scene,
        mesh_radius=mesh_radius,
        hidden_radius=hidden_radius,
        mesh_level=mesh_level,
        noise_amp=noise_amp,
        edge_x=edge_x,
        nan_pixel=nan_pixel,
        nan_sample=nan_sample,
        firefly_pixel=firefly_pixel,
        firefly_sample=firefly_sample,
        firefly_value=firefly_value,
    )
    return SimpleNamespace(
        device="cpu",
        frame_id=7,
        is_materials_dirty=True,
        disable_gaussian_tracing=False,
        max_pbr_bounces=15,
        camera_type=camera_type,
        primitives=primitives,
        environment=environment,
        scene_mog=scene_mog,
        tracer=tracer,
    )


def _renderer(
    engine,
    allocator=PerceptualVarianceAllocator(),
    rounder=StochasticRounder(0),
    budget=SimpleBudget(64),
    terms=None,
    estimator=None,
    telemetry=None,
    dumper=None,
) -> AdaptiveRenderer:
    """``AdaptiveRenderer`` with test defaults: ``VarianceOfMeanEstimator()``, the three default weight terms and a CPU
    (wall-clock) ``CostTelemetry`` unless given."""
    return AdaptiveRenderer(
        engine,
        estimator if estimator is not None else VarianceOfMeanEstimator(),
        terms if terms is not None else [LuminanceAdaptationTerm(), ContrastMaskingTerm(), DisplayClipTerm()],
        allocator,
        rounder,
        budget,
        telemetry=telemetry if telemetry is not None else CostTelemetry(use_cuda_events=False),
        dumper=dumper,
    )
