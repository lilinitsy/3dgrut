# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Oracle study: how much error would uniform, sigma-only and sigma*sqrt(w) allocation give at equal ray budgets?

Sections appear in task order. This module must not import anything from plan B (adaptive core); the pieces of plan B
that the study needs are kept as local copies marked ``# local copy of plan B <module>.<name>; keep identical``.
"""

from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from kaolin.render.camera import generate_pinhole_rays

from threedgrut_playground.utils.rng import rng_torch_low_discrepancy

# ---------------------------------------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------------------------------------

# Sample-index layout (int32): pilot 0 .. pilot_spp-1; oracle pool and independent reference at fixed offsets.
ORACLE_SAMPLE_OFFSET = 2**30
REF_SAMPLE_OFFSET = 2**30 + 2**29

# D6: a sample is a visible mesh hit iff aov_first_prim_type != 0 and aov_first_hit_transmittance >= MESH_VISIBLE_T.
MESH_VISIBLE_T = 0.5

# local copy of plan B image_ops.LUMA_REC709; keep identical
LUMA_REC709 = (0.2126, 0.7152, 0.0722)

# local copy of plan B display.CIELAB_EPS / display.CIELAB_KAPPA; keep identical
CIELAB_EPS = (6 / 29) ** 3
CIELAB_KAPPA = 24389 / 27

# D9 default luminance-adaptation reference luma, and D13 winsorization margin.
Y_REF = 0.5
# local copy of plan B estimators.WINSOR_DELTA; keep identical
WINSOR_DELTA = 0.05

# D14: allocation exponents (1.0 = Neyman, 0.5 = sub-linear pooling like FLIP).
GAMMAS = (1.0, 0.5)

# Display clip term defaults (plan B DisplayClipTerm(margin_min=0.05, noise_k=2)).
CLIP_MARGIN_MIN = 0.05
CLIP_NOISE_K = 2.0

# Plan B VarianceOfMeanEstimator(sigma_px=1.0).
SIGMA_SMOOTH_PX = 1.0

# Plan B Task 12 renderer: number of uniform top-up rounds.
TOPUP_ROUNDS = 2

# local copy of plan B rays.JITTER_HASH_X/Y, rays.HALF_SALT; keep identical
JITTER_HASH_X = 19349663
JITTER_HASH_Y = 96925573
HALF_SALT = 0x9E3779B9

# Launch packing (plan B RayBatch.to_launch): rows of width <= ROW_WIDTH, <= MAX_RAYS_PER_LAUNCH rays per launch.
ROW_WIDTH = 4096
MAX_RAYS_PER_LAUNCH = 2**22

RULES = ("uniform", "sigma", "sigma_sqrt_w")
SIGMA_SOURCES = ("oracle", "pilot")

# Four PBR spheres (metal/dielectric x smooth/rough): (material name, horizontal image position u in [0, 1]).
ORACLE_SPHERES = (
    ("rose_gold", 0.2),
    ("brushed_copper", 0.4),
    ("polished_marble", 0.6),
    ("oak_wood", 0.8),
)
ORACLE_FOV_DEG = 60.0

REPO_ROOT = Path(__file__).resolve().parents[2]

# Spec D10 / D11 are deferred; every output carries these caveats.
KNOWN_LIMITATIONS: tuple[str, ...] = (
    "(a) Independence assumption (D10): the pool + bootstrap and the analytic sigma^2/n assume independent samples, "
    "but D1's sampler is stratified Owen-Sobol consumed sequentially per pixel, which overstates uniform's error on "
    "edge and texture pixels (in the reviewer's CPU simulation the true prefix MSE was 0.08-0.20x sigma^2/n while the "
    "bootstrap gave 0.64-0.74x), so headroom figures are biased toward adaptive rules wherever jitter variance "
    "dominates.",
    "(b) Pool cap (D10): the per-pixel cap of pilot_spp + oracle_spp (4 + 256) samples clips the allocations that "
    "high budgets want, which understates headroom at high budgets (see n_capped).",
    "(c) Single pilot: the pilot is one realization shared by all repeats, so repeat std reflects only pool "
    "subsampling and understates the uncertainty of the pilot-sigma results.",
    "(d) Analytic self-reference: the analytic oracle error uses sigma from the same reference set that the "
    "empirical metrics are scored against.",
    "(e) Which variance each quantity uses (LDR vs HDR, D13): allocation (both sigma sources and the clip term's "
    "noise_var) and sanity_objective use the display-winsorized luma variance, the analytic error uses the raw HDR "
    "per-channel RGB variance of the reference, and empirical PSNR/FLIP are measured on clamp(mean, 0, 1), so the "
    "three diverge on highlights and on dark pixels whose samples go negative.",
    "(f) Scene (D11): the four-sphere scene of this study differs from plan C's garden_glossy, so the numbers do not "
    "transfer directly to plan C's sweep.",
    "(g) Wording and scope: results are estimates under the assumptions above, not a ceiling on what plan B can "
    "reach, and savings are reported both per mesh pixel and per image (whole-frame rays including the non-mesh "
    "pilot).",
)


# ---------------------------------------------------------------------------------------------------------------------
# Repo guard
# ---------------------------------------------------------------------------------------------------------------------


def ensure_outside_repo(path: str | Path) -> Path:
    """Resolve ``path`` (relative to CWD) and refuse it if it is the repository root or lies below it.

    Comparison uses ``os.path.normcase`` so that case differences on Windows do not bypass the guard.
    """
    resolved = Path(path).resolve()
    p = os.path.normcase(str(resolved))
    root = os.path.normcase(str(REPO_ROOT)).rstrip("\\/")
    if p == root or p.startswith(root + os.sep) or p.startswith(root + "/"):
        raise ValueError(f"output directory {resolved} is inside the repository {REPO_ROOT}; choose a path outside it")
    return resolved


# ---------------------------------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------------------------------


def psnr_from_mse(mse: float) -> float:
    """PSNR in dB for a signal range of 1, with the MSE floored at 1e-12."""
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def _ldr_numpy(img: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(img.detach().cpu().float().clamp(0, 1).numpy(), dtype=np.float32)


def flip_map(test_ldr: torch.Tensor, ref_ldr: torch.Tensor, ppd: float) -> torch.Tensor:
    """Per-pixel FLIP error map ``[H, W]`` (float32, CPU) of two sRGB-encoded LDR images ``[H, W, 3]`` (D14)."""
    try:
        import flip_evaluator
    except ImportError as e:  # pragma: no cover - exercised only without the dependency
        raise ImportError("FLIP is required for the oracle study (spec D3): pip install flip-evaluator") from e
    ref_np, test_np = _ldr_numpy(ref_ldr), _ldr_numpy(test_ldr)
    out = flip_evaluator.evaluate(
        ref_np,
        test_np,
        "LDR",
        inputsRGB=True,
        applyMagma=False,
        computeMeanError=False,
        parameters={"ppd": float(ppd)},
    )[0]
    return torch.from_numpy(np.ascontiguousarray(np.asarray(out, dtype=np.float32).squeeze())).reshape(
        ref_np.shape[0], ref_np.shape[1]
    )


# ---------------------------------------------------------------------------------------------------------------------
# Image ops (local copies of plan B image_ops)
# ---------------------------------------------------------------------------------------------------------------------


# local copy of plan B image_ops.luminance; keep identical
def luminance(rgb: torch.Tensor) -> torch.Tensor:
    """Rec.709 luma of display-encoded RGB ``[..., 3]`` -> ``[...]``."""
    w = torch.tensor(LUMA_REC709, dtype=rgb.dtype, device=rgb.device)
    return (rgb * w).sum(dim=-1)


# local copy of plan B image_ops.gaussian_blur2d; keep identical
def gaussian_blur2d(x: torch.Tensor, sigma_px: float) -> torch.Tensor:
    """Separable Gaussian blur of ``x [H, W]``: taps ``exp(-k^2 / 2 sigma^2)``, ``|k| <= ceil(3 sigma)``, normalized.

    Replicate padding, computed in the input dtype. ``sigma_px <= 0`` returns a clone.
    """
    if sigma_px <= 0:
        return x.clone()
    r = int(math.ceil(3.0 * sigma_px))
    k = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    taps = torch.exp(-(k * k) / (2.0 * sigma_px * sigma_px))
    taps = taps / taps.sum()
    y = F.pad(x[None, None], (r, r, r, r), mode="replicate")
    y = F.conv2d(y, taps.view(1, 1, 1, -1))
    y = F.conv2d(y, taps.view(1, 1, -1, 1))
    return y[0, 0]


# local copy of plan B image_ops.normalized_blur2d; keep identical
def normalized_blur2d(x: torch.Tensor, mask: torch.Tensor, sigma_px: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``blur(x * m) / blur(m)`` where ``blur(m) > 1e-12`` else 0. Returns ``(blurred, support = blur(m))``."""
    m = mask.to(x.dtype)
    if sigma_px <= 0:
        return torch.where(mask, x, torch.zeros_like(x)), m
    num = gaussian_blur2d(torch.where(mask, x, torch.zeros_like(x)), sigma_px)
    sup = gaussian_blur2d(m, sigma_px)
    ok = sup > 1e-12
    out = torch.where(ok, num / torch.where(ok, sup, torch.ones_like(sup)), torch.zeros_like(num))
    return out, sup


# local copy of plan B image_ops.class_blur2d; keep identical
def class_blur2d(
    x: torch.Tensor, classes: torch.Tensor, sigma_px: float, valid: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blur each pixel only over pixels of its own class that are also ``valid``. Returns ``(blurred, support)``."""
    if valid is None:
        valid = torch.ones_like(classes, dtype=torch.bool)
    out_a, sup_a = normalized_blur2d(x, classes & valid, sigma_px)
    out_b, sup_b = normalized_blur2d(x, ~classes & valid, sigma_px)
    return torch.where(classes, out_a, out_b), torch.where(classes, sup_a, sup_b)


# ---------------------------------------------------------------------------------------------------------------------
# Display model (local copies of plan B display)
# ---------------------------------------------------------------------------------------------------------------------


# local copy of plan B display.srgb_eotf; keep identical
def srgb_eotf(v: torch.Tensor) -> torch.Tensor:
    """sRGB EOTF (encoded -> linear); input clamped to [0, 1]."""
    v = v.clamp(0.0, 1.0)
    return torch.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


# local copy of plan B display.srgb_eotf_derivative; keep identical
def srgb_eotf_derivative(v: torch.Tensor) -> torch.Tensor:
    """d(linear)/d(encoded) of the sRGB EOTF; input clamped to [0, 1]."""
    v = v.clamp(0.0, 1.0)
    return torch.where(v <= 0.04045, torch.full_like(v, 1.0 / 12.92), (2.4 / 1.055) * ((v + 0.055) / 1.055) ** 1.4)


# local copy of plan B display.lstar_slope; keep identical
def lstar_slope(y_prime: torch.Tensor) -> torch.Tensor:
    """``dL*/dY'``: CIELAB L* slope per unit encoded luma (white Y_n = 1)."""
    y = srgb_eotf(y_prime)
    safe = torch.where(y <= CIELAB_EPS, torch.ones_like(y), y)
    dl_dy = torch.where(y <= CIELAB_EPS, torch.full_like(y, CIELAB_KAPPA), (116.0 / 3.0) * safe ** (-2.0 / 3.0))
    return dl_dy * srgb_eotf_derivative(y_prime)


# ---------------------------------------------------------------------------------------------------------------------
# Perceptual weight terms (local copies of plan B weights; O6)
# ---------------------------------------------------------------------------------------------------------------------


def _finite_f64(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x.to(torch.float64), nan=0.0, posinf=0.0, neginf=0.0)


# local copy of plan B weights.LuminanceAdaptationTerm (default, y_ref=0.5); keep identical
def luminance_adaptation_weight(mean_rgb: torch.Tensor) -> torch.Tensor:
    """D9 default: ``(lstar_slope(Y') / lstar_slope(Y_REF))^2`` with ``Y' = clamp(luma(mean), 0, 1)``. Pointwise, f64."""
    yp = luminance(_finite_f64(mean_rgb)).clamp(0.0, 1.0)
    ref = lstar_slope(torch.tensor(Y_REF, dtype=torch.float64, device=yp.device))
    return (lstar_slope(yp) / ref) ** 2


# local copy of plan B weights.DisplayClipTerm(margin_min=0.05, noise_k=2); keep identical
def display_clip_weight(mean_rgb: torch.Tensor, noise_var: Optional[torch.Tensor]) -> torch.Tensor:
    """Smoothstep fade to 0 once the minimum channel mean exceeds display white by ``delta``. f64 ``[H, W]``."""
    m = _finite_f64(mean_rgb).min(dim=-1).values
    if noise_var is None:
        delta = torch.full_like(m, CLIP_MARGIN_MIN)
    else:
        delta = (CLIP_NOISE_K * noise_var.to(torch.float64).sqrt()).clamp_min(CLIP_MARGIN_MIN)
    t = ((m - 1.0) / delta).clamp(0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


# local copy of plan B weights.combine_weights sanitising; keep identical
def perceptual_weight(mean_rgb: torch.Tensor, noise_var: Optional[torch.Tensor]) -> torch.Tensor:
    """O6 weight ``w = nan_to_num(w_lum * w_clip, nan=0, posinf=0).clamp_min(0)``, f64 ``[H, W]``."""
    w = luminance_adaptation_weight(mean_rgb) * display_clip_weight(mean_rgb, noise_var)
    return torch.nan_to_num(w, nan=0.0, posinf=0.0).clamp_min(0.0)


# ---------------------------------------------------------------------------------------------------------------------
# Allocation rules (local copies of plan B allocate / rounding / renderer top-up; O4, D14)
# ---------------------------------------------------------------------------------------------------------------------


# local copy of plan B allocate.neyman_fill; keep identical
def neyman_fill(priority: torch.Tensor, counts: torch.Tensor, budget: float) -> torch.Tensor:
    """Fill totals proportional to ``priority`` given existing ``counts``: ``n_i = max(c_i, lam * a_i)``, sum = budget.

    Returns the additions ``x >= 0``. For gamma = 1 priorities this is the Neyman solution of ``min sum s^2 w / n``;
    for other gamma it is the heuristic power allocation of D14 (no optimality claim).
    """
    x = torch.zeros_like(priority)
    idx = priority > 0
    if budget <= 0 or not idx.any():
        return x
    a, c = priority[idx], counts[idx]
    bp = c / a
    order = torch.argsort(bp)
    a_s, c_s, bp_s = a[order], c[order], bp[order]
    lam = (budget + torch.cumsum(c_s, 0)) / torch.cumsum(a_s, 0)
    k = int(torch.nonzero(bp_s < lam).max()) + 1  # largest consistent active set; k = 1 is always valid
    x[idx] = torch.clamp(lam[k - 1] * a - c, min=0)
    return x


# local copy of plan B rounding.DeterministicRounder(seed).round(rates, step); keep identical
def deterministic_round(rates: torch.Tensor, seed: int, step: int) -> torch.Tensor:
    """Largest-remainder rounding with an exact total ``round(sum(rates))`` and seeded random tie-breaks.

    Sanitises first (non-finite -> 0, clamp >= 0, float64). Ties between equal fractions fall in the order of a
    seeded random permutation, so they have no spatial bias.
    """
    shape = rates.shape
    r = torch.nan_to_num(rates.to(torch.float64), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0).reshape(-1)
    g = torch.Generator(device=r.device).manual_seed((seed * 1_000_003 + step) % 2**63)
    base = r.floor()
    frac = r - base
    total = int(round(float(r.sum())))
    k = max(total - int(base.sum()), 0)
    perm = torch.randperm(r.numel(), generator=g, device=r.device)
    _, order = torch.sort(frac[perm], descending=True, stable=True)
    add = torch.zeros(r.numel(), dtype=torch.int64, device=r.device)
    add[perm[order[:k]]] = 1
    return (base.to(torch.int64) + add).reshape(shape)


def _finite_nonneg_f64(x: torch.Tensor) -> torch.Tensor:
    return _finite_f64(x).clamp_min(0.0)


# local copy of plan B allocate._NeymanAllocator + renderer.AdaptiveRenderer.render (iterations=1, region="mesh",
# DeterministicRounder(seed=0)); keep identical
def allocate(
    rule: str,
    sigma: torch.Tensor,
    weight: torch.Tensor,
    clip: torch.Tensor,
    base: torch.Tensor,
    cap: torch.Tensor,
    total: int,
    defensive_fraction: float,
    seed: int = 0,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, int]:
    """One-shot allocation of ``total`` samples over the ``M`` mesh pixels (O4). Returns ``(counts, unspent)``.

    ``weight`` is the full ``w_lum * w_clip`` and is used only by ``sigma_sqrt_w``; ``clip`` is ``w_clip`` and drives
    the defensive share and the top-up target for every rule. ``uniform`` and ``sigma`` treat the weight as ones.
    ``counts = base + new`` never exceeds ``cap``; ``unspent = total - counts.sum()`` is non-zero only when every
    pixel with pool room is exhausted.
    """
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")
    if not (0.0 <= defensive_fraction < 1.0):
        raise ValueError(f"defensive_fraction must be in [0, 1), got {defensive_fraction}")
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError(f"gamma must be finite and >= 0, got {gamma}")
    sigma, weight, clip = _finite_nonneg_f64(sigma), _finite_nonneg_f64(weight), _finite_nonneg_f64(clip)
    base64 = base.to(torch.int64)
    extra = int(total) - int(base64.sum())
    if extra < 0:
        raise ValueError(f"total {total} is below the pilot samples already taken ({int(base64.sum())})")

    # eligibility and priority (O4): uniform / sigma use w = 1, sigma_sqrt_w uses the full weight
    w_alloc = weight if rule == "sigma_sqrt_w" else torch.ones_like(sigma)
    eligible = w_alloc > 0
    b = {"uniform": torch.ones_like(sigma), "sigma": sigma, "sigma_sqrt_w": sigma * w_alloc.sqrt()}[rule]
    g = 0.0 if rule == "uniform" else float(gamma)
    a = torch.where(eligible, b**g, torch.zeros_like(b))
    if a.sum() == 0:
        a = eligible.to(torch.float64)  # B's uniform fallback (may itself be all-zero)

    unclipped = clip > 0
    defend = unclipped  # B's eligible & unclipped; every pixel here is a mesh pixel
    d = int(math.floor(defensive_fraction * extra)) if bool(defend.any()) else 0
    x = neyman_fill(a, base64.to(torch.float64), float(extra - d))
    if x.sum() > 0:
        x = x * ((extra - d) / x.sum())
    rates = x + torch.where(defend, torch.full_like(x, d / max(int(defend.sum()), 1)), torch.zeros_like(x))

    room = (cap.to(torch.int64) - base64).clamp_min(0)
    new = torch.minimum(deterministic_round(rates, 0, 0), room)

    # uniform top-up, at most TOPUP_ROUNDS rounds (B R6)
    for r in range(TOPUP_ROUNDS):
        rem = extra - int(new.sum())
        if rem <= 0:
            break
        has_room = new < room
        target = unclipped & has_room
        if not bool(target.any()):
            target = has_room
        if not bool(target.any()):
            break
        t_rates = torch.where(target, torch.full_like(x, rem / int(target.sum())), torch.zeros_like(x))
        top = torch.minimum(deterministic_round(t_rates, 0, 1 + r), room - new)
        if int(top.sum()) == 0:
            break
        new = new + top

    counts = base64 + new
    return counts, int(total) - int(counts.sum())


# ---------------------------------------------------------------------------------------------------------------------
# Pilot statistics, plan-B sigma estimate, analytic error, bootstrap subsampling (Task 4)
# ---------------------------------------------------------------------------------------------------------------------


def pilot_stats(
    samples: torch.Tensor, winsor_delta: Optional[float] = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-pixel pilot statistics over the finite samples of ``samples [N, S, 3]``.

    Returns ``(mean [N, 3] f64, sigma_luma [N] f64, n_valid [N] int64)``. A sample is valid when all 3 channels are
    finite. The mean is always the raw RGB mean (0 where ``n_valid == 0``). ``sigma_luma`` is the unbiased std of the
    Rec.709 luma (0 where ``n_valid < 2``); with ``winsor_delta`` the luma is taken of ``clamp(rgb, 0, 1 + delta)``
    per channel (D13).
    """
    ok = torch.isfinite(samples).all(dim=-1)
    x = torch.where(ok[..., None], samples.to(torch.float64), torch.zeros((), dtype=torch.float64))
    n = ok.sum(dim=1)
    nf = n.clamp_min(1).to(torch.float64)
    mean = x.sum(dim=1) / nf[:, None]
    lum = luminance(x if winsor_delta is None else x.clamp(0.0, 1.0 + winsor_delta))
    lmean = (lum * ok).sum(dim=1) / nf
    ss = (((lum - lmean[:, None]) * ok) ** 2).sum(dim=1)
    sigma = torch.where(n >= 2, (ss / (n - 1).clamp_min(1).to(torch.float64)).sqrt(), torch.zeros_like(ss))
    return mean, sigma, n.to(torch.int64)


def _median(x: torch.Tensor) -> float:
    return float(torch.quantile(x, 0.5))


# local copy of plan B estimators.VarianceOfMeanEstimator(sigma_px=1.0), sigma step only (no c4, B's R3); keep identical
def smooth_sigma_like_b(
    sigma: torch.Tensor, n: torch.Tensor, mesh_mask: torch.Tensor, sigma_px: float = SIGMA_SMOOTH_PX
) -> torch.Tensor:
    """Mask-aware smoothing of the pilot sigma: ``class_blur2d(sigma, mesh_mask, sigma_px, valid = n >= 2)``.

    Where the support is ``<= 1e-12`` the value falls back to the median of the valid same-class sigma, then of all
    valid sigma, then 0. ``sigma`` is used as is (no small-sample correction). f64 ``[H, W]``.
    """
    sigma = sigma.to(torch.float64)
    valid = n >= 2
    out, sup = class_blur2d(sigma, mesh_mask, sigma_px, valid)
    gap = sup <= 1e-12
    if bool(gap.any()):
        fb = torch.zeros_like(sigma)
        all_valid = _median(sigma[valid]) if bool(valid.any()) else 0.0
        for cls in (mesh_mask, ~mesh_mask):
            sel = valid & cls
            fb = torch.where(cls, torch.full_like(sigma, _median(sigma[sel]) if bool(sel.any()) else all_valid), fb)
        out = torch.where(gap, fb, out)
    return out


def analytic_mse(var_rgb: torch.Tensor, counts: torch.Tensor) -> float:
    """``mean_{i,c}(var_ic / counts_i)`` for per-sample variances ``var_rgb [M, 3]`` and counts ``[M]``."""
    v = var_rgb.to(torch.float64) / counts.to(torch.float64).clamp_min(1)[:, None]
    return float(v.mean())


def bootstrap_means(
    pool: torch.Tensor,
    pilot: torch.Tensor,
    counts: torch.Tensor,
    generator: torch.Generator,
    *,
    chunk_pixels: int = 16384,
    device="cpu",
) -> torch.Tensor:
    """Subsampled estimates ``[K, M, 3]`` (f32, CPU) of each pixel's mean, drawn WITHOUT replacement (O2).

    ``counts [K, M]`` are total sample counts including the pilot: ``est[k, i] = (pilot_sum_i + sum of the first
    counts[k, i] - n_pilot_i finite pool samples of a random permutation of pixel i) / counts[k, i]``. Non-finite
    samples are skipped. Every ``k`` shares the permutation of its pixel chunk (common random numbers, O11), so
    differences between rules are paired. Chunks run in index order on one generator; results depend on
    ``chunk_pixels``. Raises ValueError ("pool") when a pixel needs more pool samples than it has finite ones, or
    fewer than zero (count below its finite pilot samples).
    """
    K, M = counts.shape
    S = pool.shape[1]
    out = torch.empty(K, M, 3, dtype=torch.float32)
    counts = counts.to(torch.int64)
    for s0 in range(0, M, chunk_pixels):
        s1 = min(s0 + chunk_pixels, M)
        c = s1 - s0
        pl = pool[s0:s1].to(device)
        pi = pilot[s0:s1].to(device)
        ok = torch.isfinite(pl).all(dim=-1)
        pool_valid = ok.sum(dim=1)
        pok = torch.isfinite(pi).all(dim=-1)
        pvalid = pok.sum(dim=1)
        psum = torch.where(pok[..., None], pi.to(torch.float64), torch.zeros((), dtype=torch.float64, device=device))
        psum = psum.sum(dim=1)
        cnt = counts[:, s0:s1].to(device)
        extra = cnt - pvalid[None]
        if bool((extra < 0).any()) or bool((extra > pool_valid[None]).any()):
            bad = int(((extra < 0) | (extra > pool_valid[None])).any(dim=0).nonzero()[0]) + s0
            raise ValueError(
                f"counts exceed what the pool can supply (pixel {bad}): need 0 <= counts - finite pilot samples "
                f"<= finite pool samples (draws are without replacement)"
            )
        keys = torch.rand(c, S, generator=generator, device=device)
        keys[~ok] = float("inf")
        perm = keys.argsort(dim=1)
        xs = pl.gather(1, perm[..., None].expand(c, S, 3)).nan_to_num(0.0, 0.0, 0.0).to(torch.float64)
        prefix = torch.cat([torch.zeros(c, 1, 3, dtype=torch.float64, device=device), xs.cumsum(dim=1)], dim=1)
        idx = extra.t()[..., None].expand(c, K, 3)  # [c, K, 3]
        drawn = prefix.gather(1, idx)  # [c, K, 3]
        est = (psum[:, None, :] + drawn) / cnt.t().to(torch.float64)[..., None]
        out[:, s0:s1] = est.permute(1, 0, 2).to(torch.float32).cpu()
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Configuration, data container, evaluation (Task 5)
# ---------------------------------------------------------------------------------------------------------------------

RUN_COLUMNS = (
    "rule",
    "sigma_source",
    "gamma",
    "budget_spp_mesh",
    "budget_spp_image",
    "repeat",
    "rays_mesh",
    "rays_unspent",
    "n_capped",
    "mse_analytic_mesh",
    "psnr_analytic_mesh",
    "psnr_analytic_all",
    "sanity_objective",
    "mse_mesh",
    "psnr_mesh",
    "psnr_all",
    "flip_mesh",
    "flip_all",
    "ref_noise_mse_mesh",
)

HEADROOM_COLUMNS = (
    "rule",
    "sigma_source",
    "gamma",
    "budget_spp_mesh",
    "budget_spp_image",
    "n_repeats",
    "psnr_mesh_mean",
    "psnr_mesh_std",
    "flip_mesh_mean",
    "flip_mesh_std",
    "dpsnr_mesh_mean",
    "dpsnr_mesh_std",
    "dpsnr_all_mean",
    "dflip_mesh_mean",
    "dflip_mesh_std",
    "dflip_mesh_rel_mean",
    "dflip_all_mean",
    "dpsnr_analytic_mesh",
    "ray_saving_empirical",
    "ray_saving_analytic",
    "ray_saving_empirical_image",
    "ray_saving_analytic_image",
    "estimation_loss_db",
)

WEIGHTS_DESCRIPTION = "luminance_adaptation[L*] x display_clip (plan B defaults; contrast_masking excluded)"
PLOT_CAPTION = "estimate; see known limitations in summary.json"
_MAX_SPP = 2**28


@dataclass(frozen=True)
class OracleConfig:
    oracle_spp: int = 256
    ref_spp: int = 2048
    pilot_spp: int = 4
    budgets: tuple[float, ...] = (8, 16, 32, 64)
    gammas: tuple[float, ...] = GAMMAS
    repeats: int = 8
    defensive_fraction: float = 0.15
    seed: int = 0
    row_width: int = ROW_WIDTH
    max_rays_per_launch: int = MAX_RAYS_PER_LAUNCH
    max_pool_gib: float = 2.0

    def validate(self) -> None:
        """Raise ValueError for settings the study cannot run (O1, D5, D14)."""
        if self.pilot_spp < 4:
            raise ValueError(f"pilot_spp must be >= 4, got {self.pilot_spp}")
        if not (1 <= self.oracle_spp <= _MAX_SPP):
            raise ValueError(f"oracle_spp must be in [1, 2**28], got {self.oracle_spp}")
        if not (2 <= self.ref_spp <= _MAX_SPP):
            raise ValueError(f"ref_spp must be in [2, 2**28], got {self.ref_spp}")
        if len(self.budgets) == 0 or len(set(self.budgets)) != len(self.budgets):
            raise ValueError(f"budgets must be non-empty and distinct, got {self.budgets}")
        lo, hi = self.pilot_spp, self.pilot_spp + self.oracle_spp
        for b in self.budgets:
            if not (lo <= b <= hi):  # also rejects NaN
                raise ValueError(f"budget {b} is outside [pilot_spp, pilot_spp + oracle_spp] = [{lo}, {hi}] (O1)")
        if len(self.gammas) == 0 or not all(g > 0 and math.isfinite(g) for g in self.gammas):
            raise ValueError(f"gammas must be non-empty with every gamma > 0, got {self.gammas}")
        if self.repeats < 1:
            raise ValueError(f"repeats must be >= 1, got {self.repeats}")
        if not (0.0 <= self.defensive_fraction < 1.0):
            raise ValueError(f"defensive_fraction must be in [0, 1), got {self.defensive_fraction}")
        if not (1 <= self.row_width < 2**24):
            raise ValueError(f"row_width must be in [1, 2**24), got {self.row_width}")


@dataclass
class OracleData:
    """Everything the evaluation needs, on the CPU. ``M`` mesh pixels, listed in row-major order of the mask."""

    height: int
    width: int
    ppd: float
    mesh_mask: torch.Tensor  # [H, W] bool
    pilot: torch.Tensor  # [P, pilot_spp, 3] f32, all pixels, NaN kept
    pool: torch.Tensor  # [M, oracle_spp, 3] f32
    ref_sum: torch.Tensor  # [M, 3] f64
    ref_sumsq: torch.Tensor  # [M, 3] f64
    ref_luma_sum: torch.Tensor  # [M] f64, raw HDR luma (reporting)
    ref_luma_sumsq: torch.Tensor  # [M] f64
    ref_wluma_sum: torch.Tensor  # [M] f64, luma of clamp(rgb, 0, 1 + WINSOR_DELTA) (D13)
    ref_wluma_sumsq: torch.Tensor  # [M] f64
    ref_count: torch.Tensor  # [M] int64, finite reference samples
    n_nonfinite: dict[str, int] = field(default_factory=dict)  # keys pilot, pool, ref
    timings: dict[str, float] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def mesh_index(self) -> torch.Tensor:
        """Flat (row-major) indices ``[M]`` int64 of the mesh pixels."""
        return torch.nonzero(self.mesh_mask.reshape(-1)).reshape(-1)

    def ref_mean(self) -> torch.Tensor:
        """Raw reference mean ``[M, 3]`` f64."""
        return self.ref_sum.to(torch.float64) / self.ref_count.to(torch.float64).clamp_min(1)[:, None]

    @staticmethod
    def _unbiased_var(s: torch.Tensor, ss: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        s, ss, n = s.to(torch.float64), ss.to(torch.float64), n.to(torch.float64)
        if s.dim() == 2:
            n = n[:, None]
        var = (ss - s * s / n.clamp_min(1)) / (n - 1).clamp_min(1)
        return torch.where(n >= 2, var.clamp_min(0.0), torch.zeros_like(var))

    def ref_var_rgb(self) -> torch.Tensor:
        """Unbiased raw HDR per-channel variance of one sample, ``[M, 3]`` f64 (analytic error)."""
        return self._unbiased_var(self.ref_sum, self.ref_sumsq, self.ref_count)

    def ref_sigma_luma(self, winsorized: bool = True) -> torch.Tensor:
        """Unbiased std ``[M]`` f64 of the Rec.709 luma of one sample: winsorized (D13, allocation) or raw (reporting)."""
        if winsorized:
            s, ss = self.ref_wluma_sum, self.ref_wluma_sumsq
        else:
            s, ss = self.ref_luma_sum, self.ref_luma_sumsq
        return self._unbiased_var(s, ss, self.ref_count).sqrt()


@dataclass
class OracleReport:
    runs: list[dict]
    headroom: list[dict]
    summary_text: str
    files: list[Path]


def _clean(o):
    """JSON-safe copy: non-finite floats become None, tuples lists, numpy/torch scalars Python numbers."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.generic, torch.Tensor)) and np.ndim(o) == 0:
        o = o.item()
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    return o


def _std(a) -> float:
    a = np.asarray(a, dtype=np.float64)
    return float(np.std(a, ddof=1)) if a.size > 1 else 0.0


def _csv_value(v):
    return repr(float(v)) if isinstance(v, (float, np.floating)) else v


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(columns))
        wr.writeheader()
        for r in rows:
            wr.writerow({c: _csv_value(r[c]) for c in columns})


def _series_style(rule: str, gamma: float) -> dict:
    if rule == "uniform":
        return dict(color="black", linestyle="-", linewidth=2.0)
    return dict(color={"sigma": "tab:blue", "sigma_sqrt_w": "tab:orange"}[rule], linewidth=2.5 if gamma == 1.0 else 1.0)


def _plot_metric(path: Path, head: list[dict], budgets, ycol: str, ylabel: str, title: str, analytic=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scol = ycol.replace("_mean", "_std")
    delta = ycol.startswith("d")
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    series: dict[tuple, list[dict]] = {}
    for r in head:
        series.setdefault((r["rule"], r["sigma_source"], r["gamma"]), []).append(r)
    for (rule, src, g), rows in series.items():
        if delta and rule == "uniform":
            continue
        rows = sorted(rows, key=lambda r: r["budget_spp_mesh"])
        x = [r["budget_spp_mesh"] for r in rows]
        st = _series_style(rule, g)
        if src == "pilot":
            st["linestyle"] = "--"
        label = "uniform" if rule == "uniform" else f"{rule} / {src} / g={g:g}"
        ax.errorbar(
            x, [r[ycol] for r in rows], yerr=[r[scol] for r in rows], label=label, capsize=2, marker="o", ms=3, **st
        )
        if analytic is not None and not delta:
            ax.plot(x, [analytic[(rule, src, g, b)] for b in x], color=st["color"], linewidth=0.8, linestyle=":")
    if delta:
        ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(budgets))
    ax.set_xticklabels([f"{b:g}" for b in budgets])
    ax.set_xlabel("budget [samples per mesh pixel, incl. pilot]")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    fig.text(0.5, 0.01, PLOT_CAPTION, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _fmt(v: float, spec: str = "+.2f") -> str:
    return "n/a" if not math.isfinite(v) else format(v, spec)


def evaluate_oracle(
    data: OracleData, cfg: OracleConfig, out_dir: Path, *, device: str | torch.device = "cpu"
) -> OracleReport:
    """Run rules x sigma sources x gamma x budgets x repeats on ``data``; write CSVs, plots and ``summary.json``."""
    t_start = time.perf_counter()
    out_dir = ensure_outside_repo(out_dir)
    cfg.validate()
    H, W = int(data.height), int(data.width)
    P = H * W
    mask = data.mesh_mask
    if tuple(mask.shape) != (H, W) or mask.dtype != torch.bool:
        raise ValueError(f"mesh_mask must be a bool tensor of shape ({H}, {W})")
    idx = data.mesh_index()
    M = int(idx.numel())
    if M == 0:
        raise ValueError("no visible mesh pixels")
    if tuple(data.pilot.shape) != (P, cfg.pilot_spp, 3):
        raise ValueError(
            f"pilot has shape {tuple(data.pilot.shape)}, expected ({P}, {cfg.pilot_spp}, 3) (cfg.pilot_spp)"
        )
    if tuple(data.pool.shape) != (M, cfg.oracle_spp, 3):
        raise ValueError(
            f"pool has shape {tuple(data.pool.shape)}, expected ({M}, {cfg.oracle_spp}, 3) (cfg.oracle_spp)"
        )
    for name in ("ref_sum", "ref_sumsq"):
        if tuple(getattr(data, name).shape) != (M, 3):
            raise ValueError(f"{name} must have shape ({M}, 3)")
    for name in ("ref_luma_sum", "ref_luma_sumsq", "ref_wluma_sum", "ref_wluma_sumsq", "ref_count"):
        if tuple(getattr(data, name).shape) != (M,):
            raise ValueError(f"{name} must have shape ({M},)")
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    # --- per-pixel inputs (O5) ----------------------------------------------------------------------------------------
    pilot_mean_all, sig_raw_all, n_all = pilot_stats(data.pilot, winsor_delta=WINSOR_DELTA)
    base = n_all[idx]
    cap = base + torch.isfinite(data.pool).all(dim=-1).sum(dim=1)
    ref_mean, var_rgb = data.ref_mean(), data.ref_var_rgb()
    sig_o = data.ref_sigma_luma(winsorized=True)

    mean_o = pilot_mean_all.clone()
    mean_o[idx] = ref_mean
    nv_o = torch.zeros(P, dtype=torch.float64)
    nv_o[idx] = sig_o**2 / base.clamp_min(1).to(torch.float64)
    w_o = perceptual_weight(mean_o.view(H, W, 3), nv_o.view(H, W)).flatten()[idx]
    clip_o = display_clip_weight(mean_o.view(H, W, 3), nv_o.view(H, W)).flatten()[idx]

    sig_p = smooth_sigma_like_b(sig_raw_all.view(H, W), n_all.view(H, W), mask).flatten()[idx]
    nv_p = torch.where(n_all >= 2, sig_raw_all**2 / n_all.clamp_min(1).to(torch.float64), torch.zeros_like(sig_raw_all))
    w_p = perceptual_weight(pilot_mean_all.view(H, W, 3), nv_p.view(H, W)).flatten()[idx]
    clip_p = display_clip_weight(pilot_mean_all.view(H, W, 3), nv_p.view(H, W)).flatten()[idx]
    inputs = {"oracle": (sig_o, w_o, clip_o), "pilot": (sig_p, w_p, clip_p)}

    # --- allocations (O4) ---------------------------------------------------------------------------------------------
    configs: list[tuple[str, str, float]] = [("uniform", "none", 0.0)]
    configs += [
        (rule, src, float(g)) for rule in ("sigma", "sigma_sqrt_w") for src in SIGMA_SOURCES for g in cfg.gammas
    ]
    budgets = [float(b) for b in cfg.budgets]
    n_pilot_rest = cfg.pilot_spp * (P - M)
    alloc: dict[tuple, dict] = {}
    count_list: list[torch.Tensor] = []
    order: list[tuple] = []
    for rule, src, g in configs:
        sig_src, w_src, clip_src = inputs["oracle" if src == "none" else src]
        for b in budgets:
            counts, unspent = allocate(
                rule, sig_src, w_src, clip_src, base, cap, round(b * M), cfg.defensive_fraction, seed=0, gamma=g
            )
            if bool((counts > cap).any()) or bool((counts < base).any()):
                raise ValueError("internal error: allocation outside [finite pilot samples, pool cap]")
            mse_a = analytic_mse(var_rgb, counts)
            alloc[(rule, src, g, b)] = dict(
                k=len(count_list),
                budget_spp_image=(n_pilot_rest + b * M) / P,
                rays_mesh=int(counts.sum()),
                rays_unspent=int(unspent),
                n_capped=int(((counts == cap) & (counts > base)).sum()),
                mse_analytic_mesh=mse_a,
                psnr_analytic_mesh=psnr_from_mse(mse_a),
                psnr_analytic_all=psnr_from_mse(mse_a * M / P),
                sanity_objective=float((w_o * sig_o**2 / counts.clamp_min(1).to(torch.float64)).mean()),
            )
            count_list.append(counts)
            order.append((rule, src, g, b))
    all_counts = torch.stack(count_list)  # [K, M]
    empty_any = (all_counts == 0).any(dim=0)  # mesh pixels with no finite sample under some allocation
    n_empty = int(empty_any.sum())

    # --- reference image and its noise floor --------------------------------------------------------------------------
    ref_img = pilot_mean_all.clone()
    ref_img[idx] = ref_mean
    ref_ldr = ref_img.view(H, W, 3).clamp(0.0, 1.0)
    ref_ldr_f32 = ref_ldr.to(torch.float32)
    ref_ldr_mesh = ref_ldr.reshape(P, 3)[idx]
    ref_noise_mse_mesh = float((var_rgb / data.ref_count.to(torch.float64).clamp_min(1)[:, None]).mean())
    pilot_mean_f32 = pilot_mean_all.to(torch.float32)
    pilot_mean_mesh = pilot_mean_f32[idx]

    # --- repeats ------------------------------------------------------------------------------------------------------
    runs: list[dict] = []
    pilot_mesh = data.pilot[idx]
    for r in range(cfg.repeats):
        gen = torch.Generator(device=device).manual_seed(cfg.seed * 1000 + r)
        est = bootstrap_means(data.pool, pilot_mesh, all_counts, gen, device=device)  # [K, M, 3] f32
        # a pixel without any finite sample has no estimate (0/0): keep its pilot mean (0), report it in n_empty
        est = torch.where(all_counts[..., None] == 0, pilot_mean_mesh[None], est)
        for (rule, src, g, b), a in zip(order, (alloc[k] for k in order)):
            e = est[a["k"]]
            img = pilot_mean_f32.clone()
            img[idx] = e
            ldr = img.view(H, W, 3).clamp(0.0, 1.0)
            err2 = (e.to(torch.float64).clamp(0.0, 1.0) - ref_ldr_mesh) ** 2
            mse_mesh = float(err2.mean())
            mse_all = float(err2.sum()) / (P * 3)
            fm = flip_map(ldr, ref_ldr_f32, ppd=data.ppd)
            row = dict(
                rule=rule,
                sigma_source=src,
                gamma=g,
                budget_spp_mesh=b,
                repeat=r,
                mse_mesh=mse_mesh,
                psnr_mesh=psnr_from_mse(mse_mesh),
                psnr_all=psnr_from_mse(mse_all),
                flip_mesh=float(fm[mask].mean()),
                flip_all=float(fm.mean()),
                ref_noise_mse_mesh=ref_noise_mse_mesh,
            )
            row.update({k: v for k, v in a.items() if k != "k"})
            runs.append({c: row[c] for c in RUN_COLUMNS})

    # --- headroom -----------------------------------------------------------------------------------------------------
    groups: dict[tuple, list[dict]] = {}
    for row in runs:
        groups.setdefault((row["rule"], row["sigma_source"], row["gamma"], row["budget_spp_mesh"]), []).append(row)

    def col(key, name):
        return np.array([x[name] for x in sorted(groups[key], key=lambda x: x["repeat"])], dtype=np.float64)

    sorted_budgets = sorted(budgets)
    u_psnr = np.array([col(("uniform", "none", 0.0, b), "psnr_mesh").mean() for b in sorted_budgets])
    u_monotone = len(sorted_budgets) >= 2 and bool(np.all(np.diff(u_psnr) > 0))
    log_b = np.log2(np.array(sorted_budgets))
    mean_var = float(var_rgb.mean())

    def equivalent_uniform_budget(psnr: float) -> float:
        if not u_monotone or not (u_psnr[0] <= psnr <= u_psnr[-1]):
            return float("nan")
        return float(2.0 ** np.interp(psnr, u_psnr, log_b))

    def image_saving(b_eq: float, b: float) -> float:
        return (n_pilot_rest + b_eq * M) / (n_pilot_rest + b * M)

    headroom: list[dict] = []
    for key in order:
        rule, src, g, b = key
        ukey = ("uniform", "none", 0.0, b)
        a = alloc[key]
        psnr, flip = col(key, "psnr_mesh"), col(key, "flip_mesh")
        d_psnr, d_psnr_all = psnr - col(ukey, "psnr_mesh"), col(key, "psnr_all") - col(ukey, "psnr_all")
        d_flip, d_flip_all = flip - col(ukey, "flip_mesh"), col(key, "flip_all") - col(ukey, "flip_all")
        d_flip_rel = d_flip / np.maximum(col(ukey, "flip_mesh"), 1e-12)
        b_u = b if rule == "uniform" else equivalent_uniform_budget(float(psnr.mean()))
        b_ua = mean_var / max(a["mse_analytic_mesh"], 1e-300)
        headroom.append(
            dict(
                rule=rule,
                sigma_source=src,
                gamma=g,
                budget_spp_mesh=b,
                budget_spp_image=a["budget_spp_image"],
                n_repeats=int(psnr.size),
                psnr_mesh_mean=float(psnr.mean()),
                psnr_mesh_std=_std(psnr),
                flip_mesh_mean=float(flip.mean()),
                flip_mesh_std=_std(flip),
                dpsnr_mesh_mean=float(d_psnr.mean()),
                dpsnr_mesh_std=_std(d_psnr),
                dpsnr_all_mean=float(d_psnr_all.mean()),
                dflip_mesh_mean=float(d_flip.mean()),
                dflip_mesh_std=_std(d_flip),
                dflip_mesh_rel_mean=float(d_flip_rel.mean()),
                dflip_all_mean=float(d_flip_all.mean()),
                dpsnr_analytic_mesh=a["psnr_analytic_mesh"] - alloc[ukey]["psnr_analytic_mesh"],
                ray_saving_empirical=b_u / b,
                ray_saving_analytic=b_ua / b,
                ray_saving_empirical_image=image_saving(b_u, b),
                ray_saving_analytic_image=image_saving(b_ua, b),
                estimation_loss_db=float("nan"),
            )
        )
    by_key = {(h["rule"], h["sigma_source"], h["gamma"], h["budget_spp_mesh"]): h for h in headroom}
    for (rule, src, g, b), h in by_key.items():
        if src == "pilot":
            h["estimation_loss_db"] = by_key[(rule, "oracle", g, b)]["dpsnr_mesh_mean"] - h["dpsnr_mesh_mean"]

    # --- files --------------------------------------------------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files: list[Path] = []
    _write_csv(out_dir / "oracle_runs.csv", RUN_COLUMNS, runs)
    _write_csv(out_dir / "oracle_headroom.csv", HEADROOM_COLUMNS, headroom)
    files += [out_dir / "oracle_runs.csv", out_dir / "oracle_headroom.csv"]
    plt.imsave(out_dir / "reference_ldr.png", ref_ldr_f32.numpy())
    plt.imsave(out_dir / "mesh_mask.png", mask.to(torch.float32).numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    files += [out_dir / "reference_ldr.png", out_dir / "mesh_mask.png"]

    sig_raw_luma = data.ref_sigma_luma(winsorized=False)
    ok = sig_o > 0
    sigma_ratio_median = float((sig_raw_luma[ok] / sig_o[ok]).median()) if bool(ok.any()) else float("nan")
    ref_noise_share_max = max(r["ref_noise_mse_mesh"] / max(r["mse_mesh"], 1e-30) for r in runs)
    n_capped_max = max(r["n_capped"] for r in runs)
    try:
        flip_version = importlib.metadata.version("flip-evaluator")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        flip_version = None
    timings = {**{k: float(v) for k, v in data.timings.items()}, "evaluate_s": time.perf_counter() - t_start}
    summary = dict(
        config=asdict(cfg),
        height=H,
        width=W,
        P=P,
        M=M,
        mesh_fraction=M / P,
        ppd=float(data.ppd),
        n_nonfinite=dict(data.n_nonfinite),
        n_mesh_pixels_without_samples=n_empty,
        timings=timings,
        meta=data.meta,
        flip_evaluator_version=flip_version,
        ref_noise_share_max=ref_noise_share_max,
        n_capped_max=n_capped_max,
        winsor_delta=WINSOR_DELTA,
        sigma_raw_over_winsorized_median=sigma_ratio_median,
        weights=WEIGHTS_DESCRIPTION,
        known_limitations=list(KNOWN_LIMITATIONS),
    )
    (out_dir / "summary.json").write_text(json.dumps(_clean(summary), indent=2, default=str), encoding="utf-8")
    files.append(out_dir / "summary.json")

    analytic = {k: a["psnr_analytic_mesh"] for k, a in alloc.items()}
    for name, ycol, ylabel, title in (
        (
            "psnr_mesh_vs_budget",
            "psnr_mesh_mean",
            "PSNR over mesh pixels [dB]",
            "Mesh PSNR vs budget (dotted: analytic)",
        ),
        ("flip_mesh_vs_budget", "flip_mesh_mean", "mean FLIP over mesh pixels", "Mesh FLIP vs budget"),
        ("dpsnr_mesh_vs_budget", "dpsnr_mesh_mean", "dPSNR vs uniform, mesh [dB]", "Mesh dPSNR vs uniform (paired)"),
        (
            "dflip_mesh_vs_budget",
            "dflip_mesh_mean",
            "dFLIP vs uniform, mesh",
            "Mesh dFLIP vs uniform (paired; < 0 is better)",
        ),
    ):
        p = out_dir / "plots" / f"{name}.png"
        _plot_metric(
            p, headroom, sorted_budgets, ycol, ylabel, title, analytic=analytic if ycol == "psnr_mesh_mean" else None
        )
        files.append(p)

    text = _summary_text(data, cfg, H, W, M, P, headroom, summary)
    return OracleReport(runs=runs, headroom=headroom, summary_text=text, files=files)


def _summary_text(data, cfg, H, W, M, P, headroom, summary) -> str:
    by_key = {(h["rule"], h["sigma_source"], h["gamma"], h["budget_spp_mesh"]): h for h in headroom}
    budgets = sorted({h["budget_spp_mesh"] for h in headroom})
    gammas = [float(g) for g in cfg.gammas]
    L: list[str] = []
    L.append(
        f"Oracle study: {W}x{H} px, mesh pixels M={M} ({100.0 * M / P:.1f}% of {P}), ppd={data.ppd:.2f}, "
        f"pilot_spp={cfg.pilot_spp}, oracle_spp={cfg.oracle_spp}, ref_spp={cfg.ref_spp}, "
        f"defensive_fraction={cfg.defensive_fraction:g}, gammas={list(cfg.gammas)}, repeats={cfg.repeats}, "
        f"budgets={[float(b) for b in cfg.budgets]}"
    )
    L.append(f"weights: {WEIGHTS_DESCRIPTION}")
    L.append("")
    L.append("Known limitations (D10, D11 — deferred):")
    L.extend(KNOWN_LIMITATIONS)
    L.append("")
    L.append("Note: pilot-σ is a one-shot lower bound for plan B's iterative loop.")
    L.append("")
    L.append(
        "budget rule sigma_source γ | dPSNR_mesh mean±std [dB] | dPSNR_all [dB] | dFLIP_mesh rel mean [%] | "
        "ray saving mesh (emp / analytic) | ray saving image (emp / analytic) | analytic dB | est. loss"
    )
    for h in headroom:
        if h["rule"] == "uniform":
            continue
        L.append(
            f"{h['budget_spp_mesh']:g} {h['rule']} {h['sigma_source']} {h['gamma']:g} | "
            f"{_fmt(h['dpsnr_mesh_mean'])}±{_fmt(h['dpsnr_mesh_std'], '.2f')} | {_fmt(h['dpsnr_all_mean'], '+.3f')} | "
            f"{_fmt(100.0 * h['dflip_mesh_rel_mean'], '+.1f')} | "
            f"{_fmt(h['ray_saving_empirical'], '.2f')} / {_fmt(h['ray_saving_analytic'], '.2f')} | "
            f"{_fmt(h['ray_saving_empirical_image'], '.2f')} / {_fmt(h['ray_saving_analytic_image'], '.2f')} | "
            f"{_fmt(h['dpsnr_analytic_mesh'])} | "
            f"{_fmt(h['estimation_loss_db']) if h['sigma_source'] == 'pilot' else 'n/a'}"
        )
    L.append("")
    L.append("Numbers to look at (no thresholds; you decide):")
    L.append("1. sigma_sqrt_w with oracle sigma vs uniform: an estimate under limitations (a)-(f).")
    for g in gammas:
        for b in budgets:
            h = by_key[("sigma_sqrt_w", "oracle", g, b)]
            L.append(
                f"   γ={g:g} budget={b:g}: dPSNR_mesh {_fmt(h['dpsnr_mesh_mean'])}±{_fmt(h['dpsnr_mesh_std'], '.2f')} dB, "
                f"dFLIP_mesh {_fmt(h['dflip_mesh_mean'], '+.4f')} ({_fmt(100.0 * h['dflip_mesh_rel_mean'], '+.1f')}%)"
            )
    L.append("2. Estimation loss, oracle minus pilot dPSNR_mesh [dB]:")
    for rule in ("sigma", "sigma_sqrt_w"):
        for g in gammas:
            L.append(
                f"   {rule} γ={g:g}: "
                + ", ".join(f"b={b:g}: {_fmt(by_key[(rule, 'pilot', g, b)]['estimation_loss_db'])}" for b in budgets)
            )
    L.append("3. Value of the D9 weight, sigma_sqrt_w minus sigma (oracle sigma), dPSNR_mesh [dB] / dFLIP_mesh:")
    for g in gammas:
        parts = []
        for b in budgets:
            hw, hs = by_key[("sigma_sqrt_w", "oracle", g, b)], by_key[("sigma", "oracle", g, b)]
            parts.append(
                f"b={b:g}: {_fmt(hw['dpsnr_mesh_mean'] - hs['dpsnr_mesh_mean'])} / "
                f"{_fmt(hw['dflip_mesh_mean'] - hs['dflip_mesh_mean'], '+.4f')}"
            )
        L.append(f"   γ={g:g}: " + ", ".join(parts))
    L.append("4. Ray-saving factors (emp / analytic), per mesh pixel and per image (oracle sigma):")
    for rule in ("sigma", "sigma_sqrt_w"):
        for g in gammas:
            for b in budgets:
                h = by_key[(rule, "oracle", g, b)]
                L.append(
                    f"   {rule} γ={g:g} budget={b:g}: mesh {_fmt(h['ray_saving_empirical'], '.2f')} / "
                    f"{_fmt(h['ray_saving_analytic'], '.2f')}, image {_fmt(h['ray_saving_empirical_image'], '.2f')} / "
                    f"{_fmt(h['ray_saving_analytic_image'], '.2f')}"
                )
    L.append("5. Validity:")
    share = summary["ref_noise_share_max"]
    L.append(
        f"   ref_noise_share_max = {_fmt(share, '.3f')}"
        + ("  WARNING: reference noise exceeds 10% of the measured MSE; increase --ref-spp" if share > 0.10 else "")
    )
    L.append(f"   n_capped_max = {summary['n_capped_max']} (pool cap, limitation b)")
    L.append(f"   n_nonfinite = {dict(data.n_nonfinite)}")
    L.append(
        f"   mesh pixels with no finite samples (kept at the pilot mean) = {summary['n_mesh_pixels_without_samples']}"
    )
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------------------------------
# Jittered ray generation and row packing (Task 6)
# ---------------------------------------------------------------------------------------------------------------------


# local copy of plan B rays.pixel_jitter (half-split Owen-scrambled Sobol, D1/D12); keep identical
def pixel_jitter(pixel_id: torch.Tensor, sample_index: torch.Tensor, width: int) -> torch.Tensor:
    """Sub-pixel offsets in [0, 1) for rays keyed by (pixel_id, sample_index); a pure function of its inputs."""
    pid = pixel_id.long()
    px, py = pid % width, pid // width
    s = sample_index.long()
    base = (px * JITTER_HASH_X + py * JITTER_HASH_Y) & 0xFFFFFFFF  # same per-pixel seed hash as SPP's LD mode
    seed = (base ^ ((s & 1) * HALF_SALT)) & 0xFFFFFFFF  # D12: own scramble per half h = s & 1
    u, v = rng_torch_low_discrepancy(s >> 1, seed)  # Owen-scrambled Sobol at index s >> 1
    return torch.stack([u, v], -1).float().clamp_(0.0, 1.0 - 2.0**-24)  # float32 rounding of x*2^-32 can hit 1.0


def oracle_rays(
    camera, pixel_id: torch.Tensor, sample_index: torch.Tensor, jitter: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pinhole rays (float32 origins, directions) for pixels ``pixel_id`` at jitter ``sample_index``.

    Coordinates are ``x = px + u``, ``y = py + v`` (u, v from :func:`pixel_jitter`, or 0.5 when ``jitter=False``): the
    engine's centered pixel coordinate plus a jitter in [-0.5, 0.5].
    """
    device = camera.device
    pid = pixel_id.to(device).long()
    px, py = pid % camera.width, pid // camera.width
    if jitter:
        uv = pixel_jitter(pid, sample_index.to(device), camera.width)
        u, v = uv[:, 0], uv[:, 1]
    else:
        u = v = 0.5
    x = px.float() + u
    y = py.float() + v
    o, d = generate_pinhole_rays(camera, coords_grid=[y, x])
    return o.float(), d.float()


# local copy of plan B rays.launch_shape / RayBatch.to_launch row width; keep identical
def pack_rows(
    o: torch.Tensor,
    d: torch.Tensor,
    pixel_id: torch.Tensor,
    sample_index: torch.Tensor,
    row_width: int = ROW_WIDTH,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pack N rays into ``R = ceil(N/row_width)`` rows of width ``rw = ceil(N/R)`` (padding < R, padding = last ray).

    Returns fresh contiguous tensors ``(o [1,R,rw,3], d [1,R,rw,3], pixel_id int32 [1,R,rw], sample_index int32
    [1,R,rw], n_valid)``; the kernel overwrites its inputs, so nothing here aliases the arguments.
    """
    n = o.shape[0]
    if n == 0:
        raise ValueError("pack_rows needs at least one ray")
    rows = -(-n // row_width)
    rw = -(-n // rows)
    idx = torch.arange(rows * rw, device=o.device).clamp_(max=n - 1)

    def take(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        out = t.to(o.device)[idx].to(dtype)
        return out.reshape(1, rows, rw, *out.shape[1:]).contiguous()

    return (
        take(o, o.dtype),
        take(d, d.dtype),
        take(pixel_id, torch.int32),
        take(sample_index, torch.int32),
        n,
    )


# ---------------------------------------------------------------------------------------------------------------------
# GPU collection and oracle scene (Task 7)
# ---------------------------------------------------------------------------------------------------------------------

_LAUNCH_POSES = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0))  # engine.py: mog.background poses


def _check_traceable(engine) -> None:
    """Raise ValueError unless ``engine`` would take ``_render_playground_hybrid``'s path with a usable D6 mask."""
    if engine.camera_type != "Pinhole":
        raise ValueError(f"the oracle study needs engine.camera_type == 'Pinhole', got {engine.camera_type!r}")
    prims = engine.primitives
    if not prims.enabled or not prims.has_visible_objects():
        raise ValueError("the oracle study needs enabled primitives with at least one visible object")
    if engine.max_pbr_bounces < 1:  # 0 makes every AOV a sentinel, so the D6 mask would be empty
        raise ValueError(f"the oracle study needs engine.max_pbr_bounces >= 1, got {engine.max_pbr_bounces}")


def _sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def trace_samples(
    engine,
    camera,
    pixel_id: torch.Tensor,
    sample_index: torch.Tensor,
    *,
    row_width: int = ROW_WIDTH,
    max_rays_per_launch: int = MAX_RAYS_PER_LAUNCH,
) -> dict[str, torch.Tensor]:
    """Trace one jittered primary ray per ``(pixel_id[i], sample_index[i])`` like ``_render_playground_hybrid``.

    Same render options, stacked fields, materials, envmap and ``mog.background`` blend as the engine, but **no clamp**
    (the returned RGB is HDR), ``engine.frame_id`` is left unchanged, and every random draw is keyed by the ids
    (interface A). Returns ``rgb [N, 3] f32``, ``prim_type [N] int32`` (``aov_first_prim_type``) and
    ``transmittance [N] f32`` (``aov_first_hit_transmittance``) on the engine device.
    """
    from threedgrut_playground.engine import OptixPlaygroundRenderOptions

    _check_traceable(engine)
    if pixel_id.dim() != 1 or pixel_id.shape != sample_index.shape:
        raise ValueError(
            f"pixel_id and sample_index must be 1-D of equal length, got {tuple(pixel_id.shape)}, "
            f"{tuple(sample_index.shape)}"
        )
    if max_rays_per_launch < 1:
        raise ValueError(f"max_rays_per_launch must be >= 1, got {max_rays_per_launch}")
    n = int(pixel_id.numel())
    device = engine.device
    rgb = torch.empty(n, 3, dtype=torch.float32, device=device)
    prim_type = torch.empty(n, dtype=torch.int32, device=device)
    transmittance = torch.empty(n, dtype=torch.float32, device=device)
    if n == 0:
        return dict(rgb=rgb, prim_type=prim_type, transmittance=transmittance)
    P = int(camera.width) * int(camera.height)
    if int(pixel_id.min()) < 0 or int(pixel_id.max()) >= P:
        raise ValueError(f"pixel_id must lie in [0, {P})")
    if int(sample_index.min()) < 0 or int(sample_index.max()) >= 2**31:
        raise ValueError("sample_index must lie in [0, 2**31) (int32 ids)")

    mog = engine.scene_mog
    prims = engine.primitives
    opts = 0  # engine.py: playground_render_opts
    if prims.use_smooth_normals:
        opts |= OptixPlaygroundRenderOptions.SMOOTH_NORMALS
    if engine.disable_gaussian_tracing:
        opts |= OptixPlaygroundRenderOptions.DISABLE_GAUSSIAN_TRACING
    if prims.disable_pbr_textures:
        opts |= OptixPlaygroundRenderOptions.DISABLE_PBR_TEXTURES

    prims.rebuild_bvh_if_needed()
    envmap = engine.environment.get_envmap()
    envmap_offset = engine.environment.get_envmap_offset().to(device="cpu", dtype=torch.float32)  # host reads it
    blend_background = envmap is None or not engine.environment.is_ignore_envmap()

    for start in range(0, n, max_rays_per_launch):
        stop = min(n, start + max_rays_per_launch)
        pid, sidx = pixel_id[start:stop], sample_index[start:stop]
        o, d = oracle_rays(camera, pid, sidx)
        ray_o, ray_d, ray_pid, ray_sidx, n_valid = pack_rows(o, d, pid, sidx, row_width=row_width)  # fresh tensors
        fields = prims.stacked_fields
        out = engine.tracer.render_playground(
            gaussians=mog,
            ray_o=ray_o.to(device),
            ray_d=ray_d.to(device),
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
            materials=sorted(prims.registered_materials.values(), key=lambda mat: mat.material_id),
            is_sync_materials=engine.is_materials_dirty,
            refractive_index=fields.refractive_index_tensor[:, None],
            envmap=envmap,
            envmap_offset=envmap_offset,
            max_pbr_bounces=engine.max_pbr_bounces,
            ray_pixel_id=ray_pid.to(device),
            ray_sample_index=ray_sidx.to(device),
        )
        pred_features, pred_opacity = out["pred_features"], out["pred_opacity"]
        if blend_background:  # engine.py: same condition, poses and call
            poses = torch.tensor(_LAUNCH_POSES, dtype=torch.float32)
            pred_features, pred_opacity = mog.background(
                poses.contiguous(), out["last_ray_d"].contiguous(), pred_features, pred_opacity, False
            )
        engine.is_materials_dirty = False  # unlike the engine: no frame_id advance and no clamp
        rgb[start:stop] = pred_features.reshape(-1, 3)[:n_valid]
        prim_type[start:stop] = out["aov_first_prim_type"].reshape(-1, 1)[:n_valid, 0]
        transmittance[start:stop] = out["aov_first_hit_transmittance"].reshape(-1, 1)[:n_valid, 0]
        del out, pred_features, pred_opacity, ray_o, ray_d, o, d
    return dict(rgb=rgb, prim_type=prim_type, transmittance=transmittance)


def _n_nonfinite(rgb: torch.Tensor) -> int:
    """Number of samples (last dim = RGB) with any non-finite channel."""
    return int((~torch.isfinite(rgb).all(dim=-1)).sum())


@torch.no_grad()
def collect_oracle_data(engine, camera, cfg: OracleConfig, *, fov_deg: float = ORACLE_FOV_DEG) -> OracleData:
    """Trace the pilot (all pixels), then the oracle pool and the independent reference of the D6 mesh pixels.

    O2, O8, O9. ``fov_deg`` is the camera's vertical FOV, used only for ``ppd = H / fov_deg`` (D14) and the metadata.
    Every returned tensor is on the CPU.
    """
    cfg.validate()
    _check_traceable(engine)
    if not (math.isfinite(fov_deg) and fov_deg > 0):
        raise ValueError(f"fov_deg must be finite and > 0, got {fov_deg}")
    device = engine.device
    H, W = int(camera.height), int(camera.width)
    P = H * W
    Sp, S, R = cfg.pilot_spp, cfg.oracle_spp, cfg.ref_spp
    max_rays = cfg.max_rays_per_launch

    def trace(pid: torch.Tensor, sidx: torch.Tensor) -> dict[str, torch.Tensor]:
        return trace_samples(engine, camera, pid, sidx, row_width=cfg.row_width, max_rays_per_launch=max_rays)

    timings: dict[str, float] = {}
    n_nonfinite: dict[str, int] = {}

    # 1. Pilot: every pixel, sample indices 0 .. Sp-1; D6 mask = union of visible mesh hits over the pilot samples.
    _sync(device)
    t0 = time.perf_counter()
    out = trace(torch.arange(P, device=device).repeat_interleave(Sp), torch.arange(Sp, device=device).repeat(P))
    visible = (out["prim_type"] != 0) & (out["transmittance"] >= MESH_VISIBLE_T)
    mesh_flat = visible.view(P, Sp).any(dim=1)
    pilot = out["rgb"].view(P, Sp, 3).cpu()
    n_nonfinite["pilot"] = _n_nonfinite(pilot)
    del out, visible
    _sync(device)
    timings["pilot_s"] = time.perf_counter() - t0

    # 2. Guards.
    mesh_idx = torch.nonzero(mesh_flat).reshape(-1)
    M = int(mesh_idx.numel())
    if M == 0:
        raise ValueError(
            f"no visible mesh pixels (D6: prim_type != 0 and transmittance >= {MESH_VISIBLE_T} in any of the "
            f"{Sp} pilot samples of the {P} pixels)"
        )
    pool_bytes = M * S * 12
    if pool_bytes > cfg.max_pool_gib * 2**30:
        raise ValueError(
            f"the pool needs {pool_bytes / 2**30:.2f} GiB ({M} mesh pixels x {S} samples x 12 B), "
            f"above max_pool_gib = {cfg.max_pool_gib} GiB"
        )

    # 3. Pool: S samples per mesh pixel at ORACLE_SAMPLE_OFFSET + k, kept as float32 on the CPU (O8).
    _sync(device)
    t0 = time.perf_counter()
    pool = torch.empty(M, S, 3, dtype=torch.float32)
    s_pool = ORACLE_SAMPLE_OFFSET + torch.arange(S, device=device)
    chunk = max(1, max_rays // S)
    n_nonfinite["pool"] = 0
    for a in range(0, M, chunk):
        px = mesh_idx[a : a + chunk]
        k = int(px.numel())
        rgb = trace(px.repeat_interleave(S), s_pool.repeat(k))["rgb"].view(k, S, 3)
        n_nonfinite["pool"] += _n_nonfinite(rgb)
        pool[a : a + k] = rgb.cpu()
        del rgb
    _sync(device)
    timings["pool_s"] = time.perf_counter() - t0

    # 4. Independent reference (O2): R samples at REF_SAMPLE_OFFSET + k, float64 moments over finite samples only.
    _sync(device)
    t0 = time.perf_counter()
    f64 = dict(dtype=torch.float64, device=device)
    ref_sum, ref_sumsq = torch.zeros(M, 3, **f64), torch.zeros(M, 3, **f64)
    ref_luma_sum, ref_luma_sumsq = torch.zeros(M, **f64), torch.zeros(M, **f64)
    ref_wluma_sum, ref_wluma_sumsq = torch.zeros(M, **f64), torch.zeros(M, **f64)
    ref_count = torch.zeros(M, dtype=torch.int64, device=device)
    blk = min(R, max_rays)
    chunk = max(1, max_rays // blk)
    n_nonfinite["ref"] = 0
    for k0 in range(0, R, blk):
        b = min(blk, R - k0)
        s_ref = REF_SAMPLE_OFFSET + k0 + torch.arange(b, device=device)
        for a in range(0, M, chunk):
            px = mesh_idx[a : a + chunk]
            k = int(px.numel())
            x = trace(px.repeat_interleave(b), s_ref.repeat(k))["rgb"].view(k, b, 3).to(torch.float64)
            fin = torch.isfinite(x).all(dim=-1)  # [k, b]: a sample counts only if all 3 channels are finite
            n_nonfinite["ref"] += int((~fin).sum())
            x = torch.where(fin[..., None], x, torch.zeros_like(x))
            luma = luminance(x)
            wluma = luminance(x.clamp(0.0, 1.0 + WINSOR_DELTA)) * fin  # D13 winsorized luma, 0 where non-finite
            ref_sum[a : a + k] += x.sum(dim=1)
            ref_sumsq[a : a + k] += (x * x).sum(dim=1)
            ref_luma_sum[a : a + k] += luma.sum(dim=1)
            ref_luma_sumsq[a : a + k] += (luma * luma).sum(dim=1)
            ref_wluma_sum[a : a + k] += wluma.sum(dim=1)
            ref_wluma_sumsq[a : a + k] += (wluma * wluma).sum(dim=1)
            ref_count[a : a + k] += fin.sum(dim=1)
            del x, fin, luma, wluma
    _sync(device)
    timings["ref_s"] = time.perf_counter() - t0

    # 5. Metadata. kaolin's Camera keeps no look-at point: "at" is the point at unit distance on the optical axis.
    eye = camera.cam_pos().reshape(-1)[:3].float()
    _, d_axis = generate_pinhole_rays(
        camera,
        coords_grid=[torch.full((1,), 0.5 * H, device=camera.device), torch.full((1,), 0.5 * W, device=camera.device)],
    )
    at = eye + d_axis[0].to(eye)
    meta = {
        "camera": {"eye": eye.tolist(), "at": at.tolist(), "fov_deg": float(fov_deg), "width": W, "height": H},
        "rays": {"pilot": P * Sp, "pool": M * S, "ref": M * R},
    }
    return OracleData(
        height=H,
        width=W,
        ppd=H / fov_deg,
        mesh_mask=mesh_flat.view(H, W).cpu(),
        pilot=pilot,
        pool=pool,
        ref_sum=ref_sum.cpu(),
        ref_sumsq=ref_sumsq.cpu(),
        ref_luma_sum=ref_luma_sum.cpu(),
        ref_luma_sumsq=ref_luma_sumsq.cpu(),
        ref_wluma_sum=ref_wluma_sum.cpu(),
        ref_wluma_sumsq=ref_wluma_sumsq.cpu(),
        ref_count=ref_count.cpu(),
        n_nonfinite=n_nonfinite,
        timings=timings,
        meta=meta,
    )


def oracle_camera(engine, width: int, height: int):
    """O10: plan C's ``garden_glossy`` close camera, vertical FOV ``ORACLE_FOV_DEG``. Returns ``(camera, |at - eye|)``.

    The camera lives on ``engine.device``.
    """
    from kaolin.render.camera import Camera

    pos = engine.scene_mog.positions.detach().float().cpu()
    center = pos.median(dim=0).values
    spread = (pos - center).abs().median(dim=0).values.max()
    up = torch.tensor([0.0, -1.0, 0.0])
    at = center + 0.05 * spread * up
    eye = at + spread * torch.tensor([0.0, -1.0, -1.5])
    camera = Camera.from_args(
        eye=eye,
        at=at,
        up=up,
        fov=math.radians(ORACLE_FOV_DEG),
        width=int(width),
        height=int(height),
        dtype=torch.float32,
        device=engine.device,
    )
    return camera, float((at - eye).norm())


def setup_oracle_scene(engine, camera, focus_distance: float) -> list[str]:
    """O10: replace every primitive by the four PBR ``ORACLE_SPHERES``. Returns the names of the added spheres.

    Sphere ``k`` is centred on the unjittered ray through image point ``(u_k W, H/2)`` at distance
    ``D = 0.5 * focus_distance``, with radius ``0.1 * D``.
    """
    from threedgrut_playground.engine import OptixPrimitiveTypes

    prims = engine.primitives
    missing = [m for m, _ in ORACLE_SPHERES if m not in prims.registered_materials]
    if missing:
        raise ValueError(f"materials not registered: {missing}")
    if not (math.isfinite(focus_distance) and focus_distance > 0):
        raise ValueError(f"focus_distance must be finite and > 0, got {focus_distance}")
    for name in list(prims.objects):
        prims.remove_primitive(name)

    H, W = int(camera.height), int(camera.width)
    D = 0.5 * float(focus_distance)
    radius = 0.1 * D
    eye = camera.cam_pos().reshape(-1)[:3]
    names = []
    for material, u in ORACLE_SPHERES:
        _, d = generate_pinhole_rays(
            camera,
            coords_grid=[
                torch.full((1,), 0.5 * H, device=camera.device),
                torch.full((1,), u * W, device=camera.device),
            ],
        )
        center = eye + D * d[0].to(eye)
        prims.add_primitive("Sphere", OptixPrimitiveTypes.PBR, device=engine.device)
        name = f"Sphere {prims.instance_counter['Sphere']}"
        obj = prims.objects[name]
        tr = obj.transform
        tr.reset()
        tr.scale(2.0 * radius)  # sphere.obj has radius 0.5
        tr.translate(center.to(device=tr._translation.device, dtype=tr._translation.dtype))
        obj.material_id[:] = prims.registered_materials[material].material_id
        names.append(name)
    prims.rebuild_bvh_if_needed(force=True, rebuild=True)
    engine.invalidate_materials_on_gpu()
    return names
