# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded ``trace`` rounds, the D6 visible-hit rule, the plain uniform renderer and the plumbing statistics.

Everything that is traced here goes through the core's public launch path ``AdaptiveRenderer.trace``, fed by the
core's jittered ``rays_for_counts`` (D1) at disjoint sample indices.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch

from .core_api import (
    MESH_VISIBLE_T,
    AdaptiveRenderer,
    CostTelemetry,
    DeterministicRounder,
    RayBatch,
    SampleStats,
    SimpleBudget,
    UniformAllocator,
    VarianceOfMeanEstimator,
    rays_for_counts,
    telemetry_total_ms,
)

Tensor = torch.Tensor

# Rays per bounded round (equal to the core's per-launch cap, so a round is normally one launch).
MAX_RAYS_PER_ROUND = 2**22

# Median of the chi-squared distribution with one degree of freedom.
CHI2_1_MEDIAN = 0.4549

PLAIN_KIND = "plain_uniform"


# ---------------------------------------------------------------------------------------------------------------------
# bounded rounds
# ---------------------------------------------------------------------------------------------------------------------


def _iter_rounds(counts: Tensor, max_rays: int) -> Iterator[Tensor]:
    """Lazily yields the rounds of :func:`count_rounds` (one ``[P]`` int64 tensor alive at a time)."""
    if max_rays < 1:
        raise ValueError(f"max_rays must be >= 1, got {max_rays}")
    if counts.dim() != 1:
        raise ValueError(f"counts must be a 1-D per-pixel tensor, got shape {tuple(counts.shape)}")
    remaining = counts.to(torch.int64).clone()
    if bool((remaining < 0).any()):
        raise ValueError("counts must be non-negative")
    while True:
        pending = torch.nonzero(remaining > 0).flatten()
        n_pending = int(pending.numel())
        if n_pending == 0:
            return
        if n_pending > max_rays:  # one ray per pending pixel, in consecutive groups of max_rays pixels
            for i in range(0, n_pending, max_rays):
                r = torch.zeros_like(remaining)
                r[pending[i : i + max_rays]] = 1
                remaining -= r
                yield r
            continue
        quota = max(1, max_rays // n_pending)
        r = torch.zeros_like(remaining)
        r[pending] = remaining[pending].clamp(max=quota)
        remaining -= r
        yield r


def count_rounds(counts: Tensor, max_rays: int) -> List[Tensor]:
    """Split per-pixel ``counts`` ``[P]`` into rounds that sum to ``counts``, each with a total in ``(0, max_rays]``.

    Per round every pending pixel (remaining count > 0) gets ``min(remaining, max(1, max_rays // n_pending))`` rays;
    when ``n_pending > max_rays`` the pending pixels go in consecutive groups of ``max_rays`` (one ray each).
    """
    return list(_iter_rounds(counts, int(max_rays)))


def traced_rounds(
    renderer,
    camera,
    counts: Tensor,
    sample_base: Tensor,
    *,
    kind: str,
    max_rays: int = MAX_RAYS_PER_ROUND,
) -> Iterator[Tuple[RayBatch, Dict[str, Tensor]]]:
    """Trace ``counts[p]`` jittered rays per pixel in bounded rounds, yielding ``(batch, renderer.trace(batch))``.

    Round rays of pixel ``p`` get sample indices ``sample_base[p] + k``; after each yielded round ``sample_base`` is
    advanced **in place** by that round's counts, so consecutive calls issue disjoint sample indices.
    """
    for r in _iter_rounds(counts, int(max_rays)):
        batch = rays_for_counts(camera, r, sample_base, jitter=True)
        out = renderer.trace(batch, kind=kind)
        try:
            yield batch, out
        finally:  # the round was traced: its indices are spent even if the consumer stops here
            sample_base.add_(r.to(sample_base.device))


# ---------------------------------------------------------------------------------------------------------------------
# D6 visibility
# ---------------------------------------------------------------------------------------------------------------------


def visible_mesh_hits(out: Dict[str, Tensor]) -> Tensor:
    """D6: a sample is a visible mesh hit iff ``first_prim_type != 0`` and ``first_hit_transmittance >= T``."""
    return (out["aov_first_prim_type"][:, 0] != 0) & (out["aov_first_hit_transmittance"][:, 0] >= MESH_VISIBLE_T)


def union_by_pixel(pixel_id: Tensor, hits: Tensor, n_pixels: int) -> Tensor:
    """``[n_pixels]`` bool: True for every pixel with at least one hit sample."""
    mask = torch.zeros(int(n_pixels), dtype=torch.bool, device=pixel_id.device)
    mask[pixel_id.long()[hits.to(pixel_id.device)]] = True
    return mask


# ---------------------------------------------------------------------------------------------------------------------
# plain uniform renderer (time-axis baseline)
# ---------------------------------------------------------------------------------------------------------------------


def trace_only_renderer(engine, telemetry: Optional[CostTelemetry] = None) -> AdaptiveRenderer:
    """An ``AdaptiveRenderer`` used only for :meth:`AdaptiveRenderer.trace` (its adaptive components are inert)."""
    return AdaptiveRenderer(
        engine,
        VarianceOfMeanEstimator(),
        [],
        UniformAllocator(),
        DeterministicRounder(),
        SimpleBudget(1),
        telemetry=telemetry,
    )


@dataclass
class PlainResult:
    """Output of :func:`plain_uniform_render`.

    Attributes:
        mean_hdr: ``[H, W, 3]`` float32 mean of every finite sample, unclamped HDR.
        var_mean: ``[H, W]`` float32 variance of the mean of raw luma (``SampleStats.variance_of_mean``).
        rays_launched: rays traced, ``spp * H * W`` (launch padding excluded).
        wall_ms: wall-clock time of the whole call, device-synchronized at both ends.
        gpu_ms: sum of this call's telemetry launch times.
        n_nonfinite: samples with a non-finite channel (counted as launched, excluded from the statistics).
    """

    mean_hdr: Tensor
    var_mean: Tensor
    rays_launched: int
    wall_ms: float
    gpu_ms: float
    n_nonfinite: int


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def plain_uniform_render(renderer, camera, spp: int, sample_offset: int) -> PlainResult:
    """Every pixel gets ``spp`` jittered rays at sample indices ``sample_offset + [0, spp)``: no pilot, estimator,
    weights or mesh / non-mesh split, accumulated in ``SampleStats``.

    Rays are keyed by ``(pixel_id, sample_index)`` like the core's pilot, so the first 4 samples per pixel are the same
    rays as the pilot of ``AdaptiveRenderer.render(..., sample_offset=sample_offset)`` and a pipeline run's
    ``mesh_mask`` applies to this image.
    """
    spp, sample_offset = int(spp), int(sample_offset)
    if spp < 1:
        raise ValueError(f"spp must be >= 1, got {spp}")
    device = torch.device(renderer.engine.device)
    camera = camera.to(device)
    height, width = int(camera.height), int(camera.width)
    n_pixels = height * width
    _sync(device)
    t0 = time.perf_counter()
    start = renderer.telemetry.num_records
    stats = SampleStats(height, width, device, track_halves=False)
    counts = torch.full((n_pixels,), spp, dtype=torch.int64, device=device)
    base = torch.full((n_pixels,), sample_offset, dtype=torch.int64, device=device)
    for batch, out in traced_rounds(renderer, camera, counts, base, kind=PLAIN_KIND, max_rays=MAX_RAYS_PER_ROUND):
        stats.add(batch.pixel_id.long(), out["rgb"], batch.sample_index.long())
    mean_hdr, var_mean = stats.mean(), stats.variance_of_mean()
    _sync(device)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return PlainResult(
        mean_hdr=mean_hdr,
        var_mean=var_mean,
        rays_launched=spp * n_pixels,
        wall_ms=wall_ms,
        gpu_ms=telemetry_total_ms(renderer.telemetry, start),
        n_nonfinite=stats.num_nonfinite,
    )


# ---------------------------------------------------------------------------------------------------------------------
# plumbing statistics (pure CPU)
# ---------------------------------------------------------------------------------------------------------------------


class PlumbingError(AssertionError):
    """A plumbing statistic is outside its hard bound (a gross sampling / blending / indexing bug)."""


# Soft (warn) and hard (fail) ranges, set outside the worst 0.5%-99.5% row of the firefly simulation
# (simulate_firefly_plumbing; see the plan's Task 4 table).
PLUMBING_BOUNDS: Dict[str, Tuple[float, float]] = {
    "z_mean_soft": (0.90, 1.25),
    "z_mean_hard": (0.5, 3.0),
    "z_median_soft": (0.90, 2.30),
    "z_median_hard": (0.3, 5.0),
    "slope_soft": (-1.25, -0.75),
    "slope_hard": (-2.0, -0.4),
    "exact_mismatch_hard": (0.0, 1e-8),
}

_CHECKED = ("z_mean", "z_median", "slope", "exact_mismatch")


def z_statistics(d2: Tensor, v: Tensor, floor: float = 1e-10) -> Dict[str, float]:
    """Plumbing z statistics of squared differences ``d2`` against their expected value ``v`` (per pixel).

    Over pixels with ``v > floor``: ``z_mean = mean(d2 / v)`` and ``z_median = median(d2 / v) / 0.4549`` (the
    chi-squared(1) median), both 1 for Gaussian noise; ``nan`` when there is no such pixel. ``exact_mismatch`` is
    ``max(d2)`` over the noise-free pixels (``v <= floor``), 0 when there is none.
    """
    d2 = d2.detach().reshape(-1).to("cpu", torch.float64)
    v = v.detach().reshape(-1).to("cpu", torch.float64)
    if d2.shape != v.shape:
        raise ValueError(f"d2 and v must have the same number of elements, got {d2.numel()} and {v.numel()}")
    noisy = v > floor
    if bool(noisy.any()):
        z = d2[noisy] / v[noisy]
        z_mean = float(z.mean())
        z_median = float(z.median()) / CHI2_1_MEDIAN
    else:
        z_mean = z_median = math.nan
    exact = d2[~noisy]
    return {
        "z_mean": z_mean,
        "z_median": z_median,
        "exact_mismatch": float(exact.max()) if exact.numel() else 0.0,
    }


def _fmt(stats: Dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.6g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in stats.items())


def check_plumbing(stats: Dict[str, float]) -> List[str]:
    """Check the plumbing statistics present in ``stats`` (``z_mean``, ``z_median``, ``slope``, ``exact_mismatch``)
    against :data:`PLUMBING_BOUNDS`.

    Only the keys present are checked (other keys are carried into the messages). Returns one warning per soft
    violation; raises :class:`PlumbingError` if any statistic is non-finite or outside its hard bound. Every message
    lists all statistics. Raises ``ValueError`` if none of the checked keys is present.
    """
    present = [k for k in _CHECKED if k in stats]
    if not present:
        raise ValueError(f"check_plumbing: none of {_CHECKED} in stats ({_fmt(stats)})")
    hard: List[str] = []
    soft: List[str] = []
    for key in present:
        value = float(stats[key])
        lo, hi = PLUMBING_BOUNDS[f"{key}_hard"]
        if not math.isfinite(value) or not lo <= value <= hi:
            hard.append(f"{key}={value:.6g} outside hard bound [{lo:g}, {hi:g}]")
            continue
        soft_bounds = PLUMBING_BOUNDS.get(f"{key}_soft")
        if soft_bounds is not None and not soft_bounds[0] <= value <= soft_bounds[1]:
            soft.append(f"{key}={value:.6g} outside soft bound [{soft_bounds[0]:g}, {soft_bounds[1]:g}]")
    everything = _fmt(stats)
    if hard:
        raise PlumbingError(f"plumbing check failed: {'; '.join(hard + soft)} (statistics: {everything})")
    return [f"plumbing warning: {msg} (statistics: {everything})" for msg in soft]


def _convergence_slope(ns: Sequence[float], mses: Sequence[float], floor: float) -> float:
    """Least-squares slope of ``log(mse - floor)`` vs ``log n`` over ``mse > floor``; ``nan`` with < 2 points."""
    pts = [(math.log(n), math.log(m - floor)) for n, m in zip(ns, mses) if m > floor]
    if len(pts) < 2:
        return math.nan
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    if sxx == 0.0:
        return math.nan
    return sum((x - mx) * (y - my) for x, y in pts) / sxx


def _firefly_samples(gen: torch.Generator, P: int, n: int, p: float, F: float, sigma: float) -> Tensor:
    """``[P, n]`` float64 samples ``N(0, sigma) + F * Bernoulli(p)``."""
    x = torch.randn(P, n, generator=gen, dtype=torch.float64) * sigma
    if p > 0.0 and F != 0.0:
        x += F * (torch.rand(P, n, generator=gen, dtype=torch.float64) < p).to(torch.float64)
    return x


def _mean_and_var_of_mean(x: Tensor) -> Tuple[Tensor, Tensor]:
    n = x.shape[1]
    return x.mean(dim=1), x.var(dim=1, unbiased=True) / n


def simulate_firefly_plumbing(
    n_sims: int,
    seed: int,
    *,
    p: float,
    F: float,
    sigma: float,
    P: int,
    n_test: int,
    n_ref: int,
    ns: Sequence[int],
) -> Dict[str, List[float]]:
    """Monte-Carlo of the plumbing statistics for ``P`` pixels whose samples are ``N(0, sigma) + F * Bernoulli(p)``.

    Per simulation, one ``n_ref``-sample reference per pixel and independent test estimates:

    - z: an ``n_test``-sample estimate; ``d2 = (mean_test - mean_ref)^2``, ``v = var_mean_test + var_mean_ref``
      (unbiased sample variances over the sample counts) -> :func:`z_statistics` (``z_mean``, ``z_median``);
    - slope: for each ``n`` in ``ns`` an ``n``-sample estimate, ``mse = mean_p (mean_n - mean_ref)^2``, ``floor =
      mean_p var_mean_ref`` -> least-squares slope of ``log(mse - floor)`` vs ``log n`` (``slope``), mirroring the
      GPU 1/n plumbing test.

    Returns ``{"z_mean": [...], "z_median": [...], "slope": [...]}`` with ``n_sims`` entries each.
    """
    gen = torch.Generator().manual_seed(int(seed))
    out: Dict[str, List[float]] = {"z_mean": [], "z_median": [], "slope": []}
    for _ in range(int(n_sims)):
        ref_mean, ref_vm = _mean_and_var_of_mean(_firefly_samples(gen, P, n_ref, p, F, sigma))
        test_mean, test_vm = _mean_and_var_of_mean(_firefly_samples(gen, P, n_test, p, F, sigma))
        z = z_statistics((test_mean - ref_mean) ** 2, test_vm + ref_vm)
        out["z_mean"].append(z["z_mean"])
        out["z_median"].append(z["z_median"])
        mses = []
        for n in ns:
            m = _firefly_samples(gen, P, int(n), p, F, sigma).mean(dim=1)
            mses.append(float(((m - ref_mean) ** 2).mean()))
        out["slope"].append(_convergence_slope([float(n) for n in ns], mses, float(ref_vm.mean())))
    return out
