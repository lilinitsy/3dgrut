# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adaptive renderer: traces flat ray batches with ``(pixel_id, sample_index)`` identities through interface A and
runs the pilot -> estimate -> weight -> allocate -> round -> enforce loop (:meth:`AdaptiveRenderer.render`).

``threedgrut_playground.engine`` is imported only inside :meth:`AdaptiveRenderer.trace`, so importing this package
never loads ``engine.py``.

Known limitations (documented, not corrected):

- **R2 (1/n error model).** The estimator models the error of a pixel mean as sigma^2 / n. Under Owen-scrambled Sobol
  jitter, pixel-edge (anti-aliasing) variance converges faster than 1/n, so edge pixels are over-allocated relative
  to their true error.
- **R3 (small-sample sigma).** sigma_hat = s, the square root of the unbiased variance, is biased low at small n
  (E[s] ~ 0.92 sigma at n = 4 for normal samples), more so for skewed MC radiance whose fireflies a 4-sample pilot
  often misses entirely. The floor is the defensive fraction plus class-aware sigma smoothing; no prior is used.
- **R5 (trimming bias).** ``StochasticRounder`` is unbiased for the expected count, but when its draw exceeds the
  budget, trimming the largest counts makes the realised counts slightly biased low on the most-sampled pixels. The
  trimmed rays are not lost: they flow to the next iteration or the final top-up.
- **R6 (top-up widening, deliberate deviation from the plan's fixed top-up target).** The final uniform top-up
  (at most 2 rounds) starts on the first non-empty of eligible & unclipped, eligible, all pixels. "All pixels" is used
  only when the region is empty. When the enforcer's per-pixel cap saturates the eligible & unclipped target, the next
  round widens to all eligible pixels, so clipped region pixels absorb the budget once the unclipped ones are capped.
  Widening never goes beyond the region: rays never leave it except in the empty-region fallback, and any cap-limited
  residue is reported as ``rays_unspent``.
- **R7 (Kirk & Arvo).** With ``decouple_halves=False`` the full mean is biased, because the samples that drove the
  allocation are reused in it; the defensive fraction only reduces this. With ``decouple_halves=True`` the estimator,
  weights, mesh mask and allocation see only half A (even ``sample_index``); ``mean_hdr`` is still the full mean and
  ``mean_hdr_half_b`` is the unbiased mean of half B (odd ``sample_index``), whose jitter is an independent
  Owen-scrambled sequence (D12).
- **R8 (luminance domains).** The allocation variance uses winsorized samples (each channel clamped to
  ``[0, 1 + WINSOR_DELTA]`` before luma, D13), an approximation of the display's ``clamp(mean, 0, 1)``.
  ``ContrastMaskingTerm`` measures contrast on the *noisy* current mean, so with a 4-spp pilot noise contrast
  self-masks (noisy pixels look textured and get a lower weight); the ``noise_var`` subtraction removes this only on
  average, because the variance estimate is itself noisy.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, ClassVar, Optional, Sequence

import torch

from threedgrut_playground.adaptive.dump import describe_component
from threedgrut_playground.adaptive.image_ops import luminance
from threedgrut_playground.adaptive.protocols import WeightContext
from threedgrut_playground.adaptive.rays import (
    DEFAULT_MAX_RAYS,
    DEFAULT_ROW_WIDTH,
    INT32_MAX,
    RayBatch,
    center_view_dirs,
    drop_padding,
    launch_padding,
    max_padding,
    rays_for_counts,
)
from threedgrut_playground.adaptive.rounding import DeterministicRounder
from threedgrut_playground.adaptive.stats import SampleStats
from threedgrut_playground.adaptive.telemetry import CostTelemetry
from threedgrut_playground.adaptive.weights import DisplayClipTerm, combine_weights

Tensor = torch.Tensor

# D6: a sample is a visible mesh hit iff aov_first_prim_type != 0 and aov_first_hit_transmittance >= MESH_VISIBLE_T.
MESH_VISIBLE_T = 0.5

# engine.py _render_playground_hybrid: poses passed to mog.background.
_BACKGROUND_POSES = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0))

# Per-iteration maps (CPU) in AdaptiveResult.maps, plus one "weight_<term.name>" map per weight term.
MAP_KEYS = ("error", "weight", "rates", "counts_before", "counts_new", "luminance", "mesh_mask", "eligible")

# R6: the final uniform top-up rounds with its own fixed rounder (steps iterations, iterations + 1).
_TOPUP_ROUNDER = DeterministicRounder(seed=0)
_TOPUP_ROUNDS = 2

# AOV buffer value for pixels whose first sample has not been traced ("none": no hit).
_AOV_NONE = {"first_hit_t": -1.0}

_log = logging.getLogger(__name__)


@dataclass
class AdaptiveResult:
    """Output of :meth:`AdaptiveRenderer.render`.

    Attributes:
        mean_hdr: ``[H, W, 3]`` float32 full mean over every finite sample, unclamped HDR.
        mean_hdr_half_b: ``[H, W, 3]`` float32 mean of half B (odd ``sample_index``) when ``decouple_halves``,
            else None.
        stats: the render's :class:`SampleStats` (with parity halves).
        counts: ``[H, W]`` int64 rays launched per pixel, non-finite samples included, padding excluded.
        mesh_mask: ``[H, W]`` bool visible-mesh union mask (D6, R9).
        aovs: ``first_hit_t``, ``first_normal``, ``first_prim_type``, ``first_hit_transmittance``, ``num_bounces``,
            ``gauss_depth``, each ``[H, W, C]`` from each pixel's ``sample_offset + 0`` sample.
        maps: per adaptive iteration, CPU tensors keyed by :data:`MAP_KEYS` plus ``"weight_<term.name>"``.
        timings: ``rays_budget``, ``rays_launched`` (padding included), ``rays_padding``, ``n_nonfinite``,
            ``rays_unspent``, ``topup_rays``, ``wall_ms``, ``gpu_ms``, ``n_launches``.
    """

    mean_hdr: Tensor
    mean_hdr_half_b: Optional[Tensor]
    stats: SampleStats
    counts: Tensor
    mesh_mask: Tensor
    aovs: dict[str, Tensor]
    maps: list[dict[str, Tensor]]
    timings: dict[str, Any]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class AdaptiveRenderer:
    """Pluggable adaptive sampler over a playground ``Engine3DGRUT``.

    ``ROW_WIDTH`` bounds the launch row width and ``MAX_RAYS_PER_LAUNCH`` the rays per tracer launch (R10).
    """

    ROW_WIDTH: ClassVar[int] = DEFAULT_ROW_WIDTH
    MAX_RAYS_PER_LAUNCH: ClassVar[int] = DEFAULT_MAX_RAYS

    def __init__(
        self,
        engine,
        estimator,
        weight_terms: Sequence,
        allocator,
        rounder,
        budget,
        telemetry: Optional[CostTelemetry] = None,
        dumper=None,
    ):
        self.engine = engine
        self.estimator = estimator
        self.weight_terms = list(weight_terms)
        self.allocator = allocator
        self.rounder = rounder
        self.budget = budget
        self.telemetry = telemetry if telemetry is not None else CostTelemetry()
        self.dumper = dumper

    @torch.no_grad()
    def trace(self, batch: RayBatch, kind: str = "external", iteration: int = -1) -> dict[str, Tensor]:
        """Trace every ray of ``batch`` once, like ``_render_playground_hybrid`` but keyed by the batch ids.

        Same render options, stacked fields, materials, envmap and ``mog.background`` blend as the engine, but the RGB
        is **not clamped** (HDR) and ``engine.frame_id`` is not advanced (every random draw is keyed by
        ``(pixel_id, sample_index)``). ``engine.is_materials_dirty`` is cleared after each launch, as the engine does.

        The batch is launched in chunks of ``<= MAX_RAYS_PER_LAUNCH`` rays, each packed by
        :meth:`RayBatch.to_launch` into fresh tensors (the kernel overwrites its input rays) with padding ``< R``.
        Each chunk adds exactly one (non-nested) telemetry record ``(kind, iteration, n_chunk)``; a chunk whose launch
        raises leaves no record.

        Returns, in batch order with padding dropped: ``"rgb"`` ``[N, 3]`` f32 (background-blended, unclamped),
        ``"opacity"`` ``[N, 1]`` f32 and every ``"aov_*"`` key of interface A flattened to ``[N, C]`` (dtype kept).
        """
        engine = self.engine
        prims = engine.primitives
        if not (prims.enabled and prims.has_visible_objects()):  # engine.py:899/1062 would fall back to scene_mog.trace
            raise ValueError(
                "AdaptiveRenderer requires mesh primitives enabled and at least one visible mesh primitive"
            )
        if batch.num_rays == 0:
            raise ValueError("cannot trace an empty RayBatch")

        from threedgrut_playground.engine import (
            OptixPlaygroundRenderOptions,  # local: keeps engine.py out of imports
        )

        opts = 0  # engine.py: playground_render_opts
        if prims.use_smooth_normals:
            opts |= OptixPlaygroundRenderOptions.SMOOTH_NORMALS
        if engine.disable_gaussian_tracing:
            opts |= OptixPlaygroundRenderOptions.DISABLE_GAUSSIAN_TRACING
        if prims.disable_pbr_textures:
            opts |= OptixPlaygroundRenderOptions.DISABLE_PBR_TEXTURES

        env = engine.environment
        mog = engine.scene_mog
        device = engine.device
        parts: list[dict[str, Tensor]] = []
        for chunk in batch.split(self.MAX_RAYS_PER_LAUNCH):
            prims.rebuild_bvh_if_needed()
            fields = prims.stacked_fields
            # Fresh tensors (the kernel overwrites ray_o / ray_d); .to() is a no-op when already on the engine device.
            ray_o, ray_d, pid, sidx, n = chunk.to_launch(self.ROW_WIDTH)
            ray_o, ray_d, pid, sidx = (t.to(device) for t in (ray_o, ray_d, pid, sidx))
            envmap = env.get_envmap()
            envmap_offset = env.get_envmap_offset()
            with self.telemetry.launch(kind, iteration, n):
                out = engine.tracer.render_playground(
                    gaussians=mog,
                    ray_o=ray_o,
                    ray_d=ray_d,
                    playground_opts=opts,
                    mesh_faces=fields.triangles,
                    vertex_normals=fields.vertex_normals,
                    vertex_tangents=fields.vertex_tangents,
                    vertex_tangents_mask=fields.has_tangents,
                    primitive_type=fields.primitive_type_tensor[:, None],
                    frame_id=engine.frame_id,
                    ray_max_t=None,
                    material_uv=fields.material_uv,
                    material_id=fields.material_id,
                    materials=sorted(prims.registered_materials.values(), key=lambda m: m.material_id),
                    is_sync_materials=engine.is_materials_dirty,
                    refractive_index=fields.refractive_index_tensor[:, None],
                    envmap=envmap,
                    envmap_offset=envmap_offset,
                    max_pbr_bounces=engine.max_pbr_bounces,
                    ray_pixel_id=pid,
                    ray_sample_index=sidx,
                )
            rgb, op = out["pred_features"], out["pred_opacity"]
            if envmap is None or not env.is_ignore_envmap():  # engine.py: same condition, poses and call
                poses = torch.tensor(_BACKGROUND_POSES, dtype=torch.float32)
                rgb, op = mog.background(poses.contiguous(), out["last_ray_d"].contiguous(), rgb, op, False)
            engine.is_materials_dirty = False  # unlike the engine: no frame_id advance and no clamp
            part = {"rgb": drop_padding(rgb, n), "opacity": drop_padding(op, n)}
            part.update({k: drop_padding(v, n) for k, v in out.items() if k.startswith("aov_")})
            parts.append(part)

        if len(parts) == 1:
            return parts[0]
        return {k: torch.cat([p[k] for p in parts], dim=0) for k in parts[0]}

    # ------------------------------------------------------------------ adaptive loop

    def _validate(
        self,
        num_pixels: int,
        budget_rays: int,
        pilot_spp: int,
        iterations: int,
        region: str,
        sample_offset: int,
        defensive_fraction: float,
    ) -> None:
        engine = self.engine
        if engine.camera_type != "Pinhole":
            raise ValueError(
                f"AdaptiveRenderer.render supports only engine.camera_type == 'Pinhole' (R10), "
                f"got {engine.camera_type!r}"
            )
        if region not in ("mesh", "all"):
            raise ValueError(f"region must be 'mesh' or 'all', got {region!r}")
        if pilot_spp < 4:
            raise ValueError(f"pilot_spp must be >= 4 (D5), got {pilot_spp}")
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}")
        if not 0.0 <= defensive_fraction < 1.0:
            raise ValueError(f"defensive_fraction must be in [0, 1), got {defensive_fraction}")
        if not 0 <= sample_offset <= INT32_MAX:
            raise ValueError(f"sample_offset must be in [0, {INT32_MAX}], got {sample_offset}")
        rw, mr = self.ROW_WIDTH, self.MAX_RAYS_PER_LAUNCH
        min_budget = (
            pilot_spp * num_pixels
            + max_padding(num_pixels, rw, mr)
            + 2 * max_padding(num_pixels * (pilot_spp - 1), rw, mr)
        )
        if budget_rays < min_budget:
            raise ValueError(
                f"budget_rays={budget_rays} is below the pilot's worst case {min_budget} "
                f"(pilot_spp * H * W = {pilot_spp * num_pixels} plus launch padding; R1, R6)"
            )
        prims = engine.primitives
        if not (prims.enabled and prims.has_visible_objects()):
            raise ValueError(
                "AdaptiveRenderer requires mesh primitives enabled and at least one visible mesh primitive"
            )

    @torch.no_grad()
    def render(
        self,
        camera,
        budget_rays: int,
        pilot_spp: int = 4,
        iterations: int = 3,
        region: str = "mesh",
        sample_offset: int = 0,
        defensive_fraction: float = 0.15,
        decouple_halves: bool = False,
    ) -> AdaptiveResult:
        """Render one adaptive frame with at most ``budget_rays`` launched rays (pilot and padding included, R1, R6).

        Every call starts from fresh state and is reproducible for a given ``sample_offset`` (D2): pixel ``p``'s rays
        get consecutive sample indices ``sample_offset, sample_offset + 1, ...``, always jittered (D1) with the
        half-split sampler (D12).

        1. **Pilot (D5, D6, R9):** 1 spp over all pixels (kind ``pilot``), then ``pilot_spp - 1`` spp split by the
           provisional mesh mask (``pilot_mesh`` / ``pilot_nonmesh``). The mesh mask is the union over the pilot
           samples (only even ``sample_index`` when ``decouple_halves``) of ``prim_type != 0`` and
           ``transmittance >= MESH_VISIBLE_T``. ``region="mesh"`` restricts allocation to it, ``"all"`` does not.
        2. **Iterations:** iteration ``k`` gets ``remaining // (iterations - k)`` rays (leftovers flow forward). The
           estimator, weights and allocator see half A only when ``decouple_halves`` (R7). A ``defensive_fraction``
           share is spread uniformly over eligible pixels that are not display-clipped (D4, D5). Counts are rounded,
           capped by the budget enforcer, and shrunk until rays plus exact launch padding fit (R6). Each iteration is
           launched as ``adaptive_mesh`` / ``adaptive_nonmesh``.
        3. **Top-up (R6):** at most 2 uniform rounds spend what the iterations left (``topup_mesh`` /
           ``topup_nonmesh``). The target is the first non-empty of eligible & unclipped, eligible, all pixels;
           "all pixels" is only the empty-region fallback. If the budget enforcer's per-pixel cap saturates the
           eligible & unclipped target, the next round widens to all eligible pixels and logs a warning. This
           deliberately deviates from a fixed target, so that clipped region pixels absorb the budget when the few
           unclipped ones are capped. Widening never leaves the region; any cap-limited residue is reported as
           ``rays_unspent``.

        Invariants: ``counts.sum() + rays_padding == rays_launched <= budget_rays`` and
        ``rays_unspent == budget_rays - rays_launched``. Non-finite samples count as launched (``counts``) but are
        excluded from ``stats`` and counted in ``n_nonfinite``. Timings cover only this render's telemetry records.

        Raises:
            ValueError: naming the offending argument (non-Pinhole camera, ``region``, ``pilot_spp < 4``,
                ``iterations < 1``, ``defensive_fraction`` outside ``[0, 1)``, ``sample_offset`` outside the int32
                range, ``budget_rays`` below the pilot's worst case, no enabled visible primitives), or naming
                ``sample_index`` when an issued sample index would exceed the int32 range.
        """
        engine = self.engine
        height, width = int(camera.height), int(camera.width)
        num_pixels = height * width
        budget_rays, pilot_spp, iterations = int(budget_rays), int(pilot_spp), int(iterations)
        sample_offset, defensive_fraction = int(sample_offset), float(defensive_fraction)
        self._validate(num_pixels, budget_rays, pilot_spp, iterations, region, sample_offset, defensive_fraction)
        row_width, max_rays = self.ROW_WIDTH, self.MAX_RAYS_PER_LAUNCH

        if self.dumper is not None:  # written at render start
            self.dumper.dump_config(
                dict(
                    budget_rays=budget_rays,
                    pilot_spp=pilot_spp,
                    iterations=iterations,
                    region=region,
                    sample_offset=sample_offset,
                    defensive_fraction=defensive_fraction,
                    decouple_halves=bool(decouple_halves),
                    height=height,
                    width=width,
                    row_width=row_width,
                    max_rays_per_launch=max_rays,
                    estimator=describe_component(self.estimator),
                    allocator=describe_component(self.allocator),
                    rounder=describe_component(self.rounder),
                    budget=describe_component(self.budget),
                    weight_terms=[dict(describe_component(t), name=t.name) for t in self.weight_terms],
                    seed=getattr(self.rounder, "seed", None),
                )
            )

        device = torch.device(engine.device)
        camera = camera.to(device)
        _sync(device)
        t0 = time.perf_counter()
        rec0 = self.telemetry.num_records  # the telemetry may be shared across renders

        stats = SampleStats(height, width, device)
        counter = torch.zeros(num_pixels, dtype=torch.int64, device=device)  # rays launched per pixel
        visible = torch.zeros(num_pixels, dtype=torch.bool, device=device)
        aov_buf: dict[str, Tensor] = {}
        launched = 0
        padding = 0
        mesh_mask = visible.clone()

        def trace_counts(n: Tensor, kind: str, it: int, in_pilot: bool) -> None:
            """One logical launch of ``n[p]`` new rays per pixel (chunked inside :meth:`trace`)."""
            nonlocal launched, padding
            b = rays_for_counts(camera, n, sample_offset + counter)  # always jittered (D1), half-split (D12)
            out = self.trace(b, kind, it)
            pid, sidx = b.pixel_id.long(), b.sample_index.long()
            stats.add(pid, out["rgb"], sidx)
            if in_pilot:
                hit = (out["aov_first_prim_type"][:, 0] != 0) & (
                    out["aov_first_hit_transmittance"][:, 0] >= MESH_VISIBLE_T
                )
                if decouple_halves:
                    hit &= (sidx & 1) == 0
                visible[pid[hit]] = True
            first = sidx == sample_offset
            if bool(first.any()):
                for key, value in out.items():
                    if not key.startswith("aov_"):
                        continue
                    name = key[len("aov_") :]
                    if name not in aov_buf:
                        fill = _AOV_NONE.get(name, 0)
                        aov_buf[name] = torch.full((num_pixels, value.shape[1]), fill, dtype=value.dtype, device=device)
                    aov_buf[name][pid[first]] = value[first].to(device)
            pad = launch_padding(b.num_rays, row_width, max_rays)  # the same formula trace() launches with
            counter.add_(n)
            launched += int(n.sum()) + pad  # padding rays count as launched (R6)
            padding += pad

        def split_pad(n: Tensor) -> int:
            """Exact padding of ``trace_split(n)``."""
            return sum(
                launch_padding(int(torch.where(sel, n, 0).sum()), row_width, max_rays)
                for sel in (mesh_mask, ~mesh_mask)
            )

        def trace_split(n: Tensor, base: str, it: int, in_pilot: bool = False) -> None:
            """Separate mesh / non-mesh launches, so the cost telemetry can fit the two classes (R9)."""
            for suffix, sel in (("_mesh", mesh_mask), ("_nonmesh", ~mesh_mask)):
                part = torch.where(sel, n, 0)
                if int(part.sum()) > 0:
                    trace_counts(part, base + suffix, it, in_pilot)

        def split_max_pad(n: Tensor) -> int:
            """Upper bound (:func:`max_padding`) on the padding of ``trace_split(m)`` for every ``m <= n``."""
            return sum(
                max_padding(int(torch.where(sel, n, 0).sum()), row_width, max_rays) for sel in (mesh_mask, ~mesh_mask)
            )

        def fit(n: Tensor, avail: int) -> Tensor:
            """Shrink ``n`` (via the budget enforcer) until its rays plus exact launch padding fit ``avail`` (R6).

            The plan's version (3 exact shrinks, then zeros "unreachable in practice: padding shrinks with n") was
            wrong. Exact padding ``R * ceil(N / R) - N`` is not monotone in ``N``: at a fixed row count ``R`` it
            *grows* as rays are removed. For example, with ``ROW_WIDTH = 100`` and ``avail = 1022``, 1022 rays need 1
            padding ray, 1021 need 2 and 1020 need 3, so every shrink missed and whole iterations launched nothing.
            So after 3 exact attempts ``n`` is shrunk once against the monotone bound :func:`max_padding` of its
            current mesh / non-mesh parts. The enforcer never raises a pixel's count, so each part's exact padding
            stays within that bound and the result always fits. The few rays lost flow to the next iteration or
            top-up; any final residue is reported as ``rays_unspent``.
            """
            for _ in range(3):
                pad = split_pad(n)
                if int(n.sum()) + pad <= avail:
                    return n
                n = self.budget.enforce(n, avail - pad)
            if int(n.sum()) + split_pad(n) <= avail:
                return n
            n = self.budget.enforce(n, avail - split_max_pad(n))
            return n if int(n.sum()) + split_pad(n) <= avail else torch.zeros_like(n)

        # --- pilot (D5, D6, R9) ---
        ones = torch.ones(num_pixels, dtype=torch.int64, device=device)
        trace_counts(ones, "pilot", -1, True)
        mesh_mask = visible.clone()  # provisional: only splits the remaining pilot launches
        trace_split(ones * (pilot_spp - 1), "pilot", -1, True)
        mesh_mask = visible.clone()  # final: union over all (decoupled: even-index) pilot samples
        mesh_hw = mesh_mask.reshape(height, width)
        eligible_hw = mesh_hw if region == "mesh" else torch.ones_like(mesh_hw)
        view_dirs = center_view_dirs(camera)
        unclipped_hw = torch.ones_like(mesh_hw)

        # --- adaptive iterations ---
        maps: list[dict[str, Tensor]] = []
        for k in range(iterations):
            b_k = (budget_rays - launched) // (iterations - k)  # leftovers flow forward (R6)
            s_alloc = stats.half(0) if decouple_halves else stats
            mean = s_alloc.mean()
            ctx = WeightContext(
                mean,
                luminance(mean),
                mesh_hw,
                view_dirs,
                camera,
                None,
                s_alloc.display_variance_of_mean(),  # D13: winsorized noise variance
                s_alloc.variance_of_mean(),  # D4: raw variance for the clip margin (robust to pilot outliers)
            )
            error = self.estimator.estimate(s_alloc, ctx)
            if decouple_halves:  # rescale half A's error of the mean to the full counts (R2, R7)
                error = error * s_alloc.counts() / stats.counts().clamp_min(1)
            w, per_term = combine_weights(self.weight_terms, ctx)
            clip = per_term["display_clip"] if "display_clip" in per_term else DisplayClipTerm()(ctx)
            unclipped_hw = clip > 0
            defend = eligible_hw & unclipped_hw  # the defensive share never goes to clipped pixels (D4)
            n_defend = int(defend.sum())
            d = int(math.floor(defensive_fraction * b_k)) if n_defend > 0 else 0
            rates = self.allocator.rates(
                torch.where(eligible_hw, error, torch.zeros_like(error)),
                torch.where(eligible_hw, w, torch.zeros_like(w)),
                None,
                stats.counts(),
                b_k - d,
            )
            rates = rates + torch.where(defend, torch.full_like(rates, d / max(n_defend, 1)), torch.zeros_like(rates))
            n_new = self.budget.enforce(self.rounder.round(rates, k), b_k)
            n_new = fit(torch.where(eligible_hw, n_new, torch.zeros_like(n_new)).reshape(-1), b_k)
            m = {
                "error": error,
                "weight": w,
                "rates": rates,
                "counts_before": counter.reshape(height, width),
                "counts_new": n_new.reshape(height, width),
                "luminance": ctx.luminance,
                "mesh_mask": mesh_hw,
                "eligible": eligible_hw,
                **{f"weight_{name}": t for name, t in per_term.items()},
            }
            maps.append({key: t.detach().to("cpu", copy=True) for key, t in m.items()})
            if self.dumper is not None:
                self.dumper.dump_iteration(k, maps[-1])
            if int(n_new.sum()) > 0:
                trace_split(n_new, "adaptive", k)

        # --- final uniform top-up (R6): at most 2 rounds ---
        # Initial target: first non-empty of (eligible & unclipped, eligible, all pixels); "all pixels" only when the
        # region is empty. A round whose target was saturated by the enforcer's per-pixel cap widens the next round
        # from eligible & unclipped to eligible (deliberate deviation from the brief's fixed target, R6, D4), never
        # beyond the region: cap-limited residue stays in rays_unspent.
        ladder = [
            ("eligible unclipped pixels", (eligible_hw & unclipped_hw).reshape(-1)),
            ("all eligible pixels", eligible_hw.reshape(-1)),
            ("all pixels", torch.ones(num_pixels, dtype=torch.bool, device=device)),
        ]
        sizes = [int(sel.sum()) for _, sel in ladder]
        level = 0
        if sizes[0] == 0:
            _log.warning("adaptive top-up: no unclipped eligible pixel; falling back to all eligible pixels")
            level = 1
            if sizes[1] == 0:
                _log.warning("adaptive top-up: the region is empty; falling back to all pixels")
                level = 2
        topup = 0
        for r in range(_TOPUP_ROUNDS):
            rem = budget_rays - launched
            if rem <= 0:
                break
            target, n_target = ladder[level][1], sizes[level]
            t_rates = torch.where(
                target,
                torch.full((num_pixels,), rem / n_target, dtype=torch.float64, device=device),
                torch.zeros(num_pixels, dtype=torch.float64, device=device),
            )
            rounded = _TOPUP_ROUNDER.round(t_rates, iterations + r)
            capped = self.budget.enforce(rounded, rem)
            n_top = fit(capped, rem)
            if int(n_top.sum()) == 0:
                break
            trace_split(n_top, "topup", iterations + r)
            topup += int(n_top.sum())
            saturated = int(capped.sum()) < int(rounded.sum())  # the per-pixel cap limited this round's target
            if saturated and r + 1 < _TOPUP_ROUNDS and level == 0 and sizes[1] > sizes[0]:
                _log.warning(
                    "adaptive top-up: %s saturated by the per-pixel cap; widening the next round to %s",
                    ladder[0][0],
                    ladder[1][0],
                )
                level = 1  # widening stops at the region (never to all pixels)

        _sync(device)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        gpu_ms = self.telemetry.total_ms(rec0)
        timings = dict(
            rays_budget=budget_rays,
            rays_launched=launched,
            rays_padding=padding,
            n_nonfinite=stats.num_nonfinite,
            rays_unspent=budget_rays - launched,
            topup_rays=topup,
            wall_ms=wall_ms,
            gpu_ms=gpu_ms,
            n_launches=self.telemetry.num_records - rec0,
        )
        mean_hdr = stats.mean()
        mean_hdr_half_b = stats.half(1).mean() if decouple_halves else None
        counts = counter.reshape(height, width)
        aovs = {name: buf.reshape(height, width, -1) for name, buf in aov_buf.items()}
        if self.dumper is not None:
            self.dumper.dump_final(mean_hdr, counts, mesh_hw, aovs, timings, mean_hdr_half_b)
        return AdaptiveResult(
            mean_hdr=mean_hdr,
            mean_hdr_half_b=mean_hdr_half_b,
            stats=stats,
            counts=counts,
            mesh_mask=mesh_hw,
            aovs=aovs,
            maps=maps,
            timings=timings,
        )
