# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image-quality metrics for the adaptive-sampling evaluation (D3, D9, D14).

Every perceptual metric (FLIP, CVVDP, PSNR) sees display-referred LDR images (``to_display_ldr``) viewed at a
``ppd`` derived from the camera (see ``viewing.py``). The weighted sanity objective is computed on HDR and is a sanity
check only, never evidence (D3).
"""

from __future__ import annotations

import importlib.util
import math
from typing import Dict, Optional

import flip_evaluator
import numpy as np
import torch
from torchmetrics.functional.image import peak_signal_noise_ratio

from .core_api import L_BLACK, L_PEAK, WeightContext, combine_weights, luminance
from .ldr import count_nonfinite, to_display_ldr
from .viewing import band_masks, band_names

REF_NOISE_FLAG_RATIO = 0.1

# D9: FLIP judges the default CIELAB weight; CVVDP judges TVI-weighted runs.
PRIMARY_METRIC = {
    "cielab": ("flip_mesh", "lower"),
    "cielab_nomask": ("flip_mesh", "lower"),
    "tvi": ("cvvdp_mesh", "higher"),
}


def _region_names() -> tuple:
    bands = band_names()
    return ("all", "mesh", "mesh_clean", *(f"mesh_{b}" for b in bands), *(f"all_{b}" for b in bands))


METRIC_KEYS = (
    *(f"flip_{r}" for r in _region_names()),
    *(f"psnr_{r}" for r in _region_names()),
    "cvvdp_all",
    "cvvdp_mesh",
    "sanity_objective_all",
    "sanity_objective_mesh",
    "mse_lum_mesh",
    "ref_noise_floor_mesh",
    "ref_noise_share_mesh",
    "ref_flagged_frac_mesh",
    "mask_px_outside_ref_hi",
    "nonfinite_px",
)


# ---------------------------------------------------------------------------------------------------------------------
# FLIP
# ---------------------------------------------------------------------------------------------------------------------


def _to_numpy(img: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(img.detach().cpu().float().clamp(0.0, 1.0).numpy(), dtype=np.float32)


def flip_error_map(test_ldr: torch.Tensor, ref_ldr: torch.Tensor, ppd: float) -> torch.Tensor:
    """Per-pixel FLIP error ``[H, W]`` (float32, on the input device) of two sRGB-encoded LDR images ``[H, W, 3]``."""
    ref_np, test_np = _to_numpy(ref_ldr), _to_numpy(test_ldr)
    # flip-evaluator 1.7 returns mean_error = -1 with computeMeanError=False; the caller averages the map.
    out = flip_evaluator.evaluate(
        ref_np,
        test_np,
        "LDR",
        inputsRGB=True,
        applyMagma=False,
        computeMeanError=False,
        parameters={"ppd": float(ppd)},
    )[0]
    arr = np.ascontiguousarray(np.asarray(out, dtype=np.float32).squeeze())
    return torch.from_numpy(arr).reshape(ref_np.shape[0], ref_np.shape[1]).to(test_ldr.device)


# ---------------------------------------------------------------------------------------------------------------------
# CVVDP (optional)
# ---------------------------------------------------------------------------------------------------------------------

_CVVDP_CACHE: Dict[tuple, object] = {}


def cvvdp_available() -> bool:
    """True if ``pycvvdp`` can be imported."""
    return importlib.util.find_spec("pycvvdp") is not None


def _cvvdp_metric(ppd: float, h: int, w: int, device: torch.device):
    key = (float(ppd), int(h), int(w))
    metric = _CVVDP_CACHE.get(key)
    if metric is None:
        import pycvvdp

        photometry = pycvvdp.vvdp_display_photo_eotf(
            Y_peak=float(L_PEAK),
            contrast=float(L_PEAK) / float(L_BLACK),
            EOTF="sRGB",
            E_ambient=0.0,
            source_colorspace="sRGB",
        )
        geometry = pycvvdp.vvdp_display_geometry((int(w), int(h)), ppd=float(ppd))
        metric = pycvvdp.cvvdp(
            display_name="adaptive_eval",
            display_photometry=photometry,
            display_geometry=geometry,
            device=device,
        )
        _CVVDP_CACHE[key] = metric
    return metric


def cvvdp_jod(test_ldr: torch.Tensor, ref_ldr: torch.Tensor, ppd: float) -> Optional[float]:
    """ColorVideoVDP JOD (higher is better) of two LDR images ``[H, W, 3]``; ``None`` if pycvvdp is not installed."""
    if not cvvdp_available():
        return None
    h, w = int(ref_ldr.shape[0]), int(ref_ldr.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = _cvvdp_metric(ppd, h, w, device)
    jod, _ = metric.predict(test_ldr.float().to(device), ref_ldr.float().to(device), dim_order="HWC")
    return float(jod)


# ---------------------------------------------------------------------------------------------------------------------
# Simple metrics
# ---------------------------------------------------------------------------------------------------------------------


def psnr(test_ldr: torch.Tensor, ref_ldr: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """PSNR in dB (``data_range=1``) over ``mask`` (all pixels if ``None``); ``nan`` on an empty mask."""
    if mask is not None:
        if not bool(mask.any()):
            return math.nan
        test_ldr, ref_ldr = test_ldr[mask], ref_ldr[mask]
    return float(peak_signal_noise_ratio(test_ldr.float(), ref_ldr.float(), data_range=1.0))


def reference_weights(
    terms,
    ref_hdr: torch.Tensor,
    mesh_mask: torch.Tensor,
    view_dirs: torch.Tensor,
    camera,
    ref_noise_var: Optional[torch.Tensor] = None,
    gaze_dir: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Weight map ``[H, W]`` evaluated on the reference image. ``ref_noise_var`` is the winsorized variance of the
    mean (``SampleStats.display_variance_of_mean()``), never the raw one."""
    ctx = WeightContext(
        mean_rgb=ref_hdr,
        luminance=luminance(ref_hdr),
        mesh_mask=mesh_mask,
        view_dirs=view_dirs,
        camera=camera,
        gaze_dir=gaze_dir,
        noise_var=ref_noise_var,
    )
    return combine_weights(terms, ctx)[0]


def _hdr_luma(hdr: torch.Tensor) -> torch.Tensor:
    return luminance(torch.nan_to_num(hdr.float()))


def sanity_objective(
    test_hdr: torch.Tensor, ref_hdr: torch.Tensor, weight: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> float:
    """Mean over ``mask`` of ``w * (Y_test - Y_ref)^2`` on nan-sanitized HDR; ``nan`` if empty.
    Sanity check only, never evidence (D3)."""
    err = weight.float() * (_hdr_luma(test_hdr) - _hdr_luma(ref_hdr)) ** 2
    if mask is not None:
        if not bool(mask.any()):
            return math.nan
        err = err[mask]
    return float(err.mean())


def ref_noise_flags(ref_var_mean: torch.Tensor, uniform_var_mean: torch.Tensor) -> torch.Tensor:
    """True where the reference's variance of the mean exceeds ``REF_NOISE_FLAG_RATIO`` times that of the
    pipeline-uniform run of the same seed and budget (identical flags for every allocator at that seed/budget)."""
    return ref_var_mean > REF_NOISE_FLAG_RATIO * uniform_var_mean


# ---------------------------------------------------------------------------------------------------------------------
# Per-image evaluation
# ---------------------------------------------------------------------------------------------------------------------


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return math.nan
    return float(values[mask].float().mean())


def evaluate_image(
    test_hdr: torch.Tensor,
    ref_hdr: torch.Tensor,
    ref_var_mean: torch.Tensor,
    ref_hi_mask: torch.Tensor,
    mesh_mask: torch.Tensor,
    flags: torch.Tensor,
    ecc_deg: torch.Tensor,
    weight: torch.Tensor,
    ppd: float,
    *,
    use_cvvdp: bool = True,
) -> Dict[str, float]:
    """All ``METRIC_KEYS`` for one test image against its reference."""
    test_ldr, ref_ldr = to_display_ldr(test_hdr), to_display_ldr(ref_hdr)
    mesh = mesh_mask.bool()
    flags = flags.bool()
    bands = band_masks(ecc_deg)

    regions: Dict[str, torch.Tensor] = {"all": torch.ones_like(mesh), "mesh": mesh, "mesh_clean": mesh & ~flags}
    regions.update({f"mesh_{b}": mesh & m for b, m in bands.items()})
    regions.update({f"all_{b}": m for b, m in bands.items()})

    out: Dict[str, float] = {}
    err_map = flip_error_map(test_ldr, ref_ldr, ppd)
    for name, region in regions.items():
        out[f"flip_{name}"] = _masked_mean(err_map, region)
        out[f"psnr_{name}"] = psnr(test_ldr, ref_ldr, region)

    cv_all = cv_mesh = None
    if use_cvvdp and cvvdp_available():
        cv_all = cvvdp_jod(test_ldr, ref_ldr, ppd)
        if bool(mesh.any()):
            cv_mesh = cvvdp_jod(torch.where(mesh[..., None], test_ldr, ref_ldr), ref_ldr, ppd)
    out["cvvdp_all"] = math.nan if cv_all is None else cv_all
    out["cvvdp_mesh"] = math.nan if cv_mesh is None else cv_mesh

    out["sanity_objective_all"] = sanity_objective(test_hdr, ref_hdr, weight)
    out["sanity_objective_mesh"] = sanity_objective(test_hdr, ref_hdr, weight, mesh)

    # Error domain: B's luma on nan-sanitized (unclamped) HDR, the same domain as the variance of the mean.
    sq_err = (_hdr_luma(test_hdr) - _hdr_luma(ref_hdr)) ** 2
    out["mse_lum_mesh"] = _masked_mean(sq_err, mesh)
    out["ref_noise_floor_mesh"] = _masked_mean(ref_var_mean, mesh)
    if bool(mesh.any()):
        denom = float(sq_err[mesh].double().sum())
        out["ref_noise_share_mesh"] = float(ref_var_mean[mesh].double().sum()) / denom if denom > 0.0 else math.nan
    else:
        out["ref_noise_share_mesh"] = math.nan
    out["ref_flagged_frac_mesh"] = _masked_mean(flags, mesh)
    out["mask_px_outside_ref_hi"] = float((mesh & ~ref_hi_mask.bool()).sum())
    out["nonfinite_px"] = float(count_nonfinite(test_hdr))
    return {k: out[k] for k in METRIC_KEYS}
