# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import os
import struct
from typing import List, Optional

import torch
import torch.utils.cpp_extension

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
#

_playground_plugin = None


def load_playground_plugin(conf):
    global _playground_plugin
    if _playground_plugin is None:
        try:
            from . import libplayground_cc as tdgrt  # type: ignore
        except ImportError:
            from .setup_playground import setup_playground

            tdgrt = setup_playground(conf)
        _playground_plugin = tdgrt


# ----------------------------------------------------------------------------
#


def _validate_ray_ids(
    ray_o: torch.Tensor, ray_pixel_id: Optional[torch.Tensor], ray_sample_index: Optional[torch.Tensor]
) -> bool:
    """Validates the optional per-ray RNG key tensors of ``Tracer.render_playground``.

    Returns False when both are None (default path) and True when both are valid. Raises ValueError otherwise.
    Checks run in order: both-or-neither, is a tensor, dtype int32, shape == ray_o.shape[0:3], on CUDA, same device
    as ray_o. This is the only dtype guard, because ``packed_accessor32`` does not check dtype.
    """
    if ray_pixel_id is None and ray_sample_index is None:
        return False
    if ray_pixel_id is None or ray_sample_index is None:
        raise ValueError("ray_pixel_id and ray_sample_index must be given both or neither.")
    for name, t in (("ray_pixel_id", ray_pixel_id), ("ray_sample_index", ray_sample_index)):
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor, got {type(t).__name__}.")
    for name, t in (("ray_pixel_id", ray_pixel_id), ("ray_sample_index", ray_sample_index)):
        if t.dtype != torch.int32:
            raise ValueError(f"{name} must have dtype int32, got {t.dtype}.")
    expected_shape = tuple(ray_o.shape[0:3])
    for name, t in (("ray_pixel_id", ray_pixel_id), ("ray_sample_index", ray_sample_index)):
        if tuple(t.shape) != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape} (ray_o.shape[0:3]), got {tuple(t.shape)}.")
    for name, t in (("ray_pixel_id", ray_pixel_id), ("ray_sample_index", ray_sample_index)):
        if not t.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor, got device {t.device}.")
    for name, t in (("ray_pixel_id", ray_pixel_id), ("ray_sample_index", ray_sample_index)):
        if t.device != ray_o.device:
            raise ValueError(f"{name} must be on the same device as ray_o ({ray_o.device}), got {t.device}.")
    return True


def make_cone_params(
    pixel_spread: float, spread_per_alpha: float, max_sigma: float, max_roughness: float, split_column: int
) -> torch.Tensor:
    """CPU float32 ``[5]`` tensor read by ``traceHybrid``: camera ray spread, spread per unit GGX alpha, max cone std,
    max roughness, and the first pixel column that uses cones."""
    values = [pixel_spread, spread_per_alpha, max_sigma, max_roughness, float(split_column)]
    return torch.tensor(values, dtype=torch.float32)


MAX_LOD_SLOTS = 10  # MaxLodSlots in pipelineParameters.h
MAX_LOD_CAP_VARIANTS = 8  # MaxLodCapVariants in pipelineParameters.h
_MAX_EXACT_FLOAT32_INT = 2**24  # lodParams carries the index table offsets as float32


def _as_float32(x: float) -> float:
    """``x`` rounded to float32, as the C++ slots store their inflation."""
    return struct.unpack("f", struct.pack("f", x))[0]


def _disabled_lod_params() -> torch.Tensor:
    return torch.zeros(2, dtype=torch.float32)


class Tracer:
    def __init__(self, conf):
        self.device = "cuda"
        self.conf = conf
        self.num_update_bvh = 0

        logger.info(f'🔆 Creating Optix tracing pipeline.. Using CUDA path: "{torch.utils.cpp_extension.CUDA_HOME}"')
        torch.zeros(1, device=self.device)  # Create a dummy tensor to force cuda context init
        load_playground_plugin(conf)

        playground_module_path = os.path.dirname(__file__)
        threedgrt_tracer_module_path = os.path.abspath(os.path.join(playground_module_path, "..", "threedgrt_tracer"))

        self.tracer_wrapper = _playground_plugin.HybridOptixTracer(
            threedgrt_tracer_module_path,
            playground_module_path,
            torch.utils.cpp_extension.CUDA_HOME,
            self.conf.render.pipeline_type,
            self.conf.render.backward_pipeline_type,
            self.conf.render.primitive_type,
            self.conf.render.particle_kernel_degree,
            self.conf.render.particle_kernel_min_response,
            self.conf.render.particle_kernel_max_alpha,
            self.conf.render.particle_kernel_density_clamping,
            self.conf.render.particle_radiance_sph_degree,
            self.conf.render.enable_normals,
            self.conf.render.enable_hitcounts,
        )

    def build_gs_acc(self, gaussians, rebuild=True, inflate=0.0):
        """``inflate`` (world units) widens every proxy as if its Gaussian had std ``inflate`` added in quadrature, so
        that cone tracing can find particles widened up to that std."""
        with torch.cuda.nvtx.range(f"build-bvh-full-build-{rebuild}"):
            scale = gaussians.scale_activation(gaussians.scale).view(-1, 3)
            if inflate > 0.0:
                scale = torch.sqrt(scale * scale + inflate * inflate)
            allow_bvh_update = (
                self.conf.render.max_consecutive_bvh_update > 1
            ) and not self.conf.render.particle_kernel_density_clamping
            rebuild_bvh = (
                rebuild
                or self.conf.render.particle_kernel_density_clamping
                or self.num_update_bvh >= self.conf.render.max_consecutive_bvh_update
            )
            self.tracer_wrapper.build_bvh(
                gaussians.positions.view(-1, 3).contiguous(),
                gaussians.rotation_activation(gaussians.rotation).view(-1, 4).contiguous(),
                scale.contiguous(),
                gaussians.density_activation(gaussians.density).view(-1, 1).contiguous(),
                rebuild_bvh,
                allow_bvh_update,
            )
            self.num_update_bvh = 0 if rebuild_bvh else self.num_update_bvh + 1

    def build_lod_acc(self, lod, inflations) -> List[int]:
        """Builds the LOD slot acceleration structures 1..len(inflations) of ``lod`` (a ``GaussianLOD``), a prefix of
        the slot layout. Slot 1 holds the originals inflated by ``inflations[0]``; slot 1 + k holds cut k inflated by
        ``inflations[k]``. The kernel widens a slot's Gaussians by at most its inflation. Slot 0 is the main BVH of
        ``build_gs_acc``, which is not touched.

        Incremental: a slot already built from the same ``lod`` with the same inflation is kept, so a cap change within
        a band rebuilds only the top slot and a wider cap builds only the added slots. Slots above the prefix are kept
        too (the kernel reads only the enabled ones) and are reused if they come back with the same inflation. Another
        ``lod`` frees every slot first. Returns the slots that were built. Runs when the LOD or the cap changes, never
        per frame."""
        num_levels = len(lod.slot_index)
        if not 1 <= len(inflations) <= num_levels + 1 or 1 + len(inflations) > MAX_LOD_SLOTS:
            raise ValueError(
                f"inflations must hold 1..{min(num_levels + 1, MAX_LOD_SLOTS - 1)} values, got {len(inflations)}"
            )
        inflations = [float(x) for x in inflations]
        if not all(math.isfinite(x) and x >= 0.0 for x in inflations):
            raise ValueError(f"inflations must be finite and >= 0, got {inflations}")
        if getattr(self, "_lod_acc_owner", None) is not lod:
            self.tracer_wrapper.clear_lod_bvh()
            self._lod_acc_owner = None
            self._lod_variant_keys = {}
        # What the C++ slots hold (-1: not built, or stale), so a clear_lod_bvh from elsewhere forces a rebuild
        built = self.tracer_wrapper.lod_slot_inflations()
        num_particles = lod.positions.shape[0]
        rebuilt = []
        with torch.cuda.nvtx.range("build-lod-bvh"):
            for j, inflate in enumerate(inflations):
                slot = 1 + j
                if built[slot] >= 0.0 and built[slot] == _as_float32(inflate):
                    continue
                if j == 0:
                    rows = slice(0, lod.num_original)
                else:
                    rows = lod.slot_index[j - 1].to(device=lod.positions.device, dtype=torch.long)
                    if rows.numel() > 0 and (int(rows.min()) < 0 or int(rows.max()) >= num_particles):
                        raise ValueError(f"slot_index[{j - 1}] holds indices outside [0, {num_particles})")
                scale = lod.scales[rows]
                if inflate > 0.0:
                    scale = torch.sqrt(scale * scale + inflate * inflate)
                self._lod_acc_owner = lod
                self.tracer_wrapper.build_lod_bvh(
                    slot,
                    lod.positions[rows].contiguous(),
                    lod.rotations[rows].contiguous(),
                    scale.contiguous(),
                    lod.densities[rows].contiguous(),
                    inflate,
                )
                rebuilt.append(slot)
        self._lod_acc_owner = lod
        return rebuilt

    def build_lod_cap_variants(self, lod, variants) -> List[int]:
        """Builds the cap variants of ``lod`` for a per-pixel mode: ``variants`` is a list of ``(cap, band_slot)``, and
        variant m (physical slot ``MAX_LOD_SLOTS + m``) holds the Gaussians of slot ``band_slot`` (1: the originals,
        1 + k: cut k) inflated by exactly ``cap``. A pixel capped at ``cap`` then traces it in place of ``band_slot``,
        which a frame with a larger cap inflates to the band edge. Incremental like ``build_lod_acc``; call it after
        ``build_lod_acc`` for the same ``lod``. Returns the physical slots that were built."""
        if len(variants) > MAX_LOD_CAP_VARIANTS:
            raise ValueError(f"at most {MAX_LOD_CAP_VARIANTS} cap variants, got {len(variants)}")
        if getattr(self, "_lod_acc_owner", None) is not lod:
            raise ValueError("build the LOD slots (build_lod_acc) of this lod before its cap variants")
        built = self.tracer_wrapper.lod_slot_inflations()
        keys = getattr(self, "_lod_variant_keys", None)
        if keys is None:
            keys = self._lod_variant_keys = {}
        rebuilt = []
        for m, (cap, band) in enumerate(variants):
            cap, band = float(cap), int(band)
            if not (math.isfinite(cap) and cap > 0.0) or not 1 <= band <= len(lod.slot_index) + 1:
                raise ValueError(f"cap variant {m}: cap {cap} must be > 0 and band slot {band} in [1, K + 1]")
            physical = MAX_LOD_SLOTS + m
            key = (cap, band)
            if built[physical] == _as_float32(cap) and keys.get(physical) == key:
                continue
            rows = self._lod_slot_rows(lod, band)
            scale = torch.sqrt(lod.scales[rows] ** 2 + cap * cap)
            self.tracer_wrapper.build_lod_bvh(
                physical,
                lod.positions[rows].contiguous(),
                lod.rotations[rows].contiguous(),
                scale.contiguous(),
                lod.densities[rows].contiguous(),
                cap,
            )
            keys[physical] = key
            rebuilt.append(physical)
        return rebuilt

    @staticmethod
    def _lod_slot_rows(lod, slot: int):
        """Rows of ``lod``'s particle buffer that slot ``slot`` holds: the originals for slot 1, cut k for slot 1 + k."""
        if slot == 1:
            return slice(0, lod.num_original)
        rows = lod.slot_index[slot - 2].to(device=lod.positions.device, dtype=torch.long)
        if rows.numel() > 0 and (int(rows.min()) < 0 or int(rows.max()) >= lod.positions.shape[0]):
            raise ValueError(f"slot_index[{slot - 2}] holds indices outside [0, {lod.positions.shape[0]})")
        return rows

    def clear_lod_acc(self) -> None:
        """Frees every LOD slot structure and forgets which LOD they were built from."""
        self.tracer_wrapper.clear_lod_bvh()
        self._lod_acc_owner = None
        self._lod_variant_keys = {}

    @staticmethod
    def make_lod_params(
        lod, thin_sigma: float, num_slots: Optional[int] = None, enabled: bool = True, cap_variants=()
    ) -> torch.Tensor:
        """CPU float32 ``[2 + 2S]`` tensor read by ``traceHybrid``: enabled, S, the band lower edges of slots 0..S-1
        (0, ``thin_sigma`` = s_0, s_1, ..., s_{S-2}) and their offsets into the index table that ``render_playground``
        builds from ``lod`` (slot 0 unused, slot 1 at 0, slot 1 + k after the originals and cuts 1..k-1). ``num_slots``
        counts slot 0 and defaults to every slot, 2 + K; slots 1..S-1 must have been built by ``build_lod_acc``.
        ``cap_variants`` lists the ``(cap, band_slot)`` built by ``build_lod_cap_variants``, in the same order; they
        are appended as ``[M, cap_0, band_0, ...]``."""
        num_levels = len(lod.slot_index)
        if num_slots is None:
            num_slots = 2 + num_levels
        if not 1 <= num_slots <= min(2 + num_levels, MAX_LOD_SLOTS):
            raise ValueError(f"num_slots must be in [1, {min(2 + num_levels, MAX_LOD_SLOTS)}], got {num_slots}")
        lower = [0.0, float(thin_sigma)] + [float(s) for s in lod.level_sigmas[:num_levels]]
        offsets = [0, 0]
        end = lod.num_original
        for k in range(num_levels):
            offsets.append(end)
            end += int(lod.slot_index[k].numel())
        if end > _MAX_EXACT_FLOAT32_INT:
            raise ValueError(f"the LOD index table has {end} entries; float32 offsets are exact only up to 2^24")
        values = (
            [1.0 if enabled else 0.0, float(num_slots)] + lower[:num_slots] + [float(o) for o in offsets[:num_slots]]
        )
        if cap_variants:
            values += [float(len(cap_variants))]
            for cap, band in cap_variants:
                if not 1 <= int(band) < num_slots:
                    raise ValueError(f"cap variant band slot {band} must be in [1, {num_slots})")
                values += [float(cap), float(band)]
        return torch.tensor(values, dtype=torch.float32)

    def _lod_node_index(self, lod) -> torch.Tensor:
        """The concatenated slot index tables of ``lod``, [arange(N), cut 1, ..., cut K] as CUDA int32, built once per
        LOD object."""
        cached = getattr(self, "_lod_node_index_cache", None)
        if cached is not None and cached[0] is lod:
            return cached[1]
        tables = [torch.arange(lod.num_original, dtype=torch.int32, device=self.device)]
        tables += [t.to(device=self.device, dtype=torch.int32) for t in lod.slot_index]
        node_index = torch.cat(tables).contiguous()
        self._lod_node_index_cache = (lod, node_index)
        return node_index

    def build_mesh_acc(self, mesh_vertices, mesh_faces, rebuild=True, allow_update=True):
        with torch.cuda.nvtx.range(f"build-mesh-bvh-full-build-{rebuild}"):
            self.tracer_wrapper.build_mesh_bvh(
                mesh_vertices.view(-1, 3).contiguous(), mesh_faces.view(-1, 3).contiguous(), rebuild, allow_update
            )

    def render(self, gaussians, gpu_batch, train=False, frame_id=0):
        assert ("rays_o_cam" in gpu_batch) and ("rays_d_cam" in gpu_batch) and ("poses" in gpu_batch)

        num_gaussians = gaussians.num_gaussians
        with torch.cuda.nvtx.range(f"model.forward({num_gaussians} gaussians)"):
            # The feature mask zeros out feature dims the model shouldn't use yet.
            # That introduces a curriculum way of optimizing the model
            features = gaussians.get_features()
            if gaussians.progressive_training:
                features *= gaussians.get_active_feature_mask()

            mog_pos = gaussians.positions.contiguous()
            mog_dns = gaussians.get_density().contiguous()
            mog_rot = gaussians.get_rotation().contiguous()
            mog_scl = gaussians.get_scale().contiguous()
            particle_density = torch.concat([mog_pos, mog_dns, mog_rot, mog_scl, torch.zeros_like(mog_dns)], dim=1)

            pred_features, pred_opacity, pred_dist, pred_normals, hits_count = self.tracer_wrapper.trace(
                frame_id,
                gpu_batch["poses"].contiguous(),
                gpu_batch["rays_o_cam"].contiguous(),
                gpu_batch["rays_d_cam"].contiguous(),
                particle_density,
                features.contiguous(),
                gaussians.n_active_features,
                self.conf.render.min_transmittance,
            )

            # NOTE: disable background
            pred_features, pred_opacity = gaussians.background(
                gpu_batch["poses"].contiguous(),
                gpu_batch["rays_d_cam"].contiguous(),
                pred_features,
                pred_opacity,
                train,
            )

        return {
            "pred_features": pred_features,
            "pred_opacity": pred_opacity,
            "pred_dist": pred_dist,
            "pred_normals": torch.nn.functional.normalize(pred_normals, dim=3),
            "hits_count": hits_count,
        }

    @staticmethod
    def to_native_pbr_material(pbr_mat):
        make_contiguous_tex = lambda x, c: (
            x.contiguous() if x is not None else torch.empty([0, 0, c], dtype=torch.float32)
        )
        make_contiguous_diffuse = lambda x: (
            x.cpu().contiguous() if x is not None else torch.ones(4, dtype=torch.float32)
        )
        make_contiguous_emissive = lambda x: (
            x.cpu().contiguous() if x is not None else torch.zeros(4, dtype=torch.float32)
        )
        cpbr_material = _playground_plugin.CPBRMaterial()
        cpbr_material.material_id = pbr_mat.material_id
        cpbr_material.diffuseMap = make_contiguous_tex(pbr_mat.diffuse_map, 4)
        cpbr_material.emissiveMap = make_contiguous_tex(pbr_mat.emissive_map, 4)
        cpbr_material.metallicRoughnessMap = make_contiguous_tex(pbr_mat.metallic_roughness_map, 2)
        cpbr_material.normalMap = make_contiguous_tex(pbr_mat.normal_map, 4)
        cpbr_material.diffuseFactor = make_contiguous_diffuse(pbr_mat.diffuse_factor)
        cpbr_material.emissiveFactor = make_contiguous_emissive(pbr_mat.emissive_factor)
        cpbr_material.metallicFactor = pbr_mat.metallic_factor or 0.0
        cpbr_material.roughnessFactor = pbr_mat.roughness_factor or 0.0
        cpbr_material.alphaMode = pbr_mat.alpha_mode or 0
        cpbr_material.alphaCutoff = pbr_mat.alpha_cutoff or 0.5
        cpbr_material.transmissionFactor = pbr_mat.transmission_factor or 0.0
        cpbr_material.ior = pbr_mat.ior or 0.0
        return cpbr_material

    _PARTICLE_SOURCE_FIELDS = ("positions", "rotation", "scale", "density", "features_albedo", "features_specular")

    def _particle_buffers(self, gaussians):
        """``(particle_density, features)`` for the trace, rebuilt only when the Gaussians change.

        Activating and concatenating 1.25M Gaussians costs several ms per frame, so the result is kept while every
        source tensor is the same object at the same in-place version, and the active SH degree is unchanged.
        """
        fields = [getattr(gaussians, name, None) for name in self._PARTICLE_SOURCE_FIELDS]
        key = tuple((t, t._version) if isinstance(t, torch.Tensor) else (None, -1) for t in fields)
        key += ((id(gaussians), gaussians.n_active_features, bool(gaussians.progressive_training)),)
        cached = getattr(self, "_particle_cache", None)
        if cached is not None and len(cached[0]) == len(key):
            same = all(a[0] is b[0] and a[1] == b[1] for a, b in zip(cached[0][:-1], key[:-1]))
            if same and cached[0][-1] == key[-1]:
                return cached[1], cached[2]
        with torch.no_grad():
            features = gaussians.get_features()
            if gaussians.progressive_training:
                features = features * gaussians.get_active_feature_mask()
            mog_dns = gaussians.get_density().contiguous()
            particle_density = torch.concat(
                [
                    gaussians.positions.contiguous(),
                    mog_dns,
                    gaussians.get_rotation().contiguous(),
                    gaussians.get_scale().contiguous(),
                    torch.zeros_like(mog_dns),
                ],
                dim=1,
            ).contiguous()
            features = features.contiguous()
        self._particle_cache = (key, particle_density, features)
        return particle_density, features

    def render_playground(
        self,
        gaussians,
        ray_o,  # world coords
        ray_d,  # world coords
        playground_opts,
        mesh_faces,
        vertex_normals,
        vertex_tangents,
        vertex_tangents_mask,
        primitive_type,
        frame_id=0,
        ray_max_t=None,
        material_uv=None,
        material_id=None,
        materials=None,
        is_sync_materials=True,
        refractive_index=None,
        envmap=None,
        envmap_offset=None,
        max_pbr_bounces=7,
        ray_pixel_id: Optional[torch.Tensor] = None,
        ray_sample_index: Optional[torch.Tensor] = None,
        cone_params: Optional[torch.Tensor] = None,
        lod=None,
        lod_params: Optional[torch.Tensor] = None,
        pixel_mode: Optional[torch.Tensor] = None,
    ):
        """Traces the hybrid Gaussian + mesh scene with the playground path tracer.

        ``ray_o`` and ``ray_d`` are ``[B,H,W,3]`` world-space rays; the kernel overwrites them with the last traced
        ray (returned as ``last_ray_o`` / ``last_ray_d``), so callers must pass copies if they need them afterwards.
        ``is_sync_materials`` is ignored by the host: ``traceHybrid`` always re-syncs ``materials`` (existing
        behavior, documented here, not changed).

        ``ray_pixel_id`` / ``ray_sample_index`` are optional int32 ``[B,H,W]`` CUDA tensors, given both or neither.
        When given, every random draw of a ray (alpha test, PBR BSDF sampling) is keyed by its
        ``(pixel_id, sample_index)`` only, independent of its launch slot, the batch layout and ``frame_id``. When
        omitted, the RNG is keyed by launch index and ``frame_id`` exactly as before.

        ``cone_params`` is a CPU float32 ``[5]`` tensor from ``make_cone_params`` (camera ray spread, spread per unit
        GGX alpha, max cone std, max roughness, split column), used only with the ``CONE_TRACING`` option. Zeros if
        omitted.

        ``lod`` is an optional ``GaussianLOD``. When given, the particle buffers are ``lod.particle_density()`` and
        ``lod.features`` instead of the per-frame concat of ``gaussians`` (the originals keep indices 0..N-1), and its
        concatenated index table is passed along. ``lod_params`` comes from ``Tracer.make_lod_params``; cone pixels
        trace the LOD slots when it is enabled. Disabled if omitted; an enabled ``lod_params`` needs ``lod``.

        ``pixel_mode`` is an optional float32 ``[B,H,W]`` CUDA tensor choosing each pixel's render in one launch:
        ``n > 0`` renders ``n`` Monte Carlo samples averaged in the kernel (sample ``i`` uses the random numbers of
        frame ``frame_id + i``, so it equals the mean of ``n`` frames), ``-c < 0`` one cone with its std capped at
        ``min(c, max cone std)`` (needs ``CONE_TRACING``; LOD slots must be built for the largest ``c``), ``0`` nothing.
        It replaces the split column. Off if omitted.

        Besides ``pred_features``, ``pred_opacity``, ``pred_dist``, ``pred_normals``, ``hits_count``, ``last_ray_o``
        and ``last_ray_d``, the returned dict always holds six per-ray AOVs ("first hit" is the first
        ``__closesthit__ch`` call, which always comes from the input ray; the "first segment" is the input ray from
        ``t=1e-9`` up to that hit, or up to ``ray_max_t`` on a miss):

        ``aov_first_hit_t`` (f32 ``[B,H,W,1]``)
            ``optixGetRayTmax()`` at the first hit, taken before glass adds ``REFRACTION_EPS_SHIFT``. ``-1.0`` if no
            hit. The hit is geometric: it is recorded even when Gaussians hide the mesh, and
            ``aov_first_hit_transmittance`` tells whether it is visible.
        ``aov_first_normal`` (f32 ``[B,H,W,3]``)
            The normal computed at the first hit: smooth interpolated if ``SMOOTH_NORMALS`` is set, else the hard face
            normal. Not face-forwarded, no normal-map perturbation. Zeros if no hit.
        ``aov_first_prim_type`` (i32 ``[B,H,W,1]``)
            ``primitive_type`` of the first hit triangle; alpha-test pass-through hits count. ``0`` if no hit.
        ``aov_first_hit_transmittance`` (f32 ``[B,H,W,1]``)
            ``1 - density`` of the first segment's Gaussians if there is a hit, ``0.0`` if no hit. Exactly ``1.0``
            when no Gaussian contributed (e.g. Gaussian tracing disabled).
        ``aov_num_bounces`` (i32 ``[B,H,W,1]``)
            Redirections along the whole path: +1 per Mirror or Glass hit (reflection and refraction), +1 per PBR hit
            that sampled a BSDF direction. Diffuse, alpha pass-through and None hits add 0.
        ``aov_gauss_depth`` (f32 ``[B,H,W,1]``)
            Transmittance-weighted mean Gaussian hit distance of the first segment, ``hitDistance / density``
            (``sum(w_i t_i) / sum(w_i)``), ``0`` if no Gaussian contributed.

        Diffuse snapshot rule: a Diffuse first hit already integrates the first segment inside the closest-hit
        program, and the ray-generation program then integrates it a second time (the Diffuse *color* keeps that
        upstream double integration; it is not fixed here). ``aov_first_hit_transmittance`` and ``aov_gauss_depth``
        use the single integration (a snapshot taken right after the mesh trace), so they match a Mirror or PBR hit
        on the same segment.

        Caveat, ``max_pbr_bounces=0``: the path loop never runs, so all six AOVs report their no-hit sentinels
        (``-1``, zeros, ``0``, ``0.0``, ``0``, ``0``) even if a mesh is in view. This is accepted behavior.
        """
        use_ids = _validate_ray_ids(ray_o, ray_pixel_id, ray_sample_index)
        if not use_ids:
            ray_pixel_id = torch.empty([0, 0, 0], dtype=torch.int32, device=ray_o.device)
            ray_sample_index = torch.empty([0, 0, 0], dtype=torch.int32, device=ray_o.device)
        if cone_params is None:
            cone_params = make_cone_params(0.0, 0.0, 0.0, 0.0, 0)
        if lod_params is None:
            lod_params = _disabled_lod_params()
        if lod is None:
            if lod_params[0].item() != 0.0:
                raise ValueError("enabled lod_params need the lod they were made for")
            lod_node_index = torch.empty([0], dtype=torch.int32, device=ray_o.device)
        else:
            lod_node_index = self._lod_node_index(lod)
        if ray_max_t is None:
            ray_max_t = ray_o.new_full(size=ray_o.shape[0:3], fill_value=1e9)
        if refractive_index is None:
            refractive_index = torch.ones_like(primitive_type, dtype=torch.float)
        if material_uv is None:
            material_uv = torch.empty([0, 3, 2], dtype=torch.float)
        if material_id is None:
            material_id = torch.empty([0, 0], dtype=torch.int)
        if materials is None:
            materials = []
        else:
            materials = [self.to_native_pbr_material(m) for m in materials]
        if envmap is None:
            envmap = torch.zeros([4, 4, 4], dtype=torch.float32, device="cuda")
        if envmap_offset is None:
            envmap = torch.zeros([2], dtype=torch.float32)

        poses = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=torch.float32)

        num_gaussians = gaussians.num_gaussians
        with torch.cuda.nvtx.range(f"model.forward({num_gaussians} gaussians)"):
            # The feature mask zeros out feature dims the model shouldn't use yet.
            # That introduces a curriculum way of optimizing the model
            if lod is None:
                particle_density, features = self._particle_buffers(gaussians)
            else:
                particle_density = lod.particle_density()
                features = lod.features.contiguous()
            sph_degree = gaussians.n_active_features
            if pixel_mode is None:
                pixel_mode_arg = torch.empty([0], dtype=torch.float32, device=ray_o.device)
            else:
                pixel_mode_arg = pixel_mode.to(device=ray_o.device, dtype=torch.float32).contiguous()
            min_transmittance = self.conf.render.min_transmittance
            envmap_offset = envmap_offset.contiguous()

            (
                pred_features,
                pred_opacity,
                pred_dist,
                pred_normals,
                hits_count,
                aov_first_hit_t,
                aov_first_normal,
                aov_first_prim_type,
                aov_first_hit_transmittance,
                aov_num_bounces,
                aov_gauss_depth,
            ) = self.tracer_wrapper.trace_hybrid(
                frame_id,
                poses,
                ray_o,
                ray_d,
                particle_density,
                features,
                sph_degree,
                min_transmittance,
                ray_max_t,
                playground_opts,
                mesh_faces,
                vertex_normals,
                vertex_tangents,
                vertex_tangents_mask,
                primitive_type,
                material_uv,
                material_id,
                materials,
                is_sync_materials,
                refractive_index,
                envmap,
                envmap_offset,
                max_pbr_bounces,
                ray_pixel_id,
                ray_sample_index,
                cone_params.contiguous(),
                lod_params.contiguous(),
                lod_node_index,
                pixel_mode_arg,
            )

            pred_dist = pred_dist[:, :, :, 0:1]  # return only the hit distance

        return {
            "pred_features": pred_features,
            "pred_opacity": pred_opacity,
            "pred_dist": pred_dist,
            "pred_normals": torch.nn.functional.normalize(pred_normals, dim=3),
            "hits_count": hits_count,
            "last_ray_o": ray_o,  # Rewritten by tracer
            "last_ray_d": ray_d,  # Rewritten by tracer
            "aov_first_hit_t": aov_first_hit_t,
            "aov_first_normal": aov_first_normal,
            "aov_first_prim_type": aov_first_prim_type,
            "aov_first_hit_transmittance": aov_first_hit_transmittance,
            "aov_num_bounces": aov_num_bounces,
            "aov_gauss_depth": aov_gauss_depth,
        }

    @torch.cuda.nvtx.range("denoise")
    def denoise(self, ray_radiance):
        return self.tracer_wrapper.denoise(ray_radiance.contiguous())
