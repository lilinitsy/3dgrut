# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cheap headroom test: how far can the sampling rate on mesh pixels drop before the error is visible?

No adaptive allocation is involved. One pixel per ``k x k`` cell is kept on each object, and the rest are reconstructed
with a normalized Gaussian inside that object. Two arms:

- ``clean``: kept pixels take the reference value, so this measures the spatial bandwidth of the signal alone. It is an
  upper bound on headroom with respect to noise (as if temporal accumulation had removed it).
- ``budget``: kept pixels get ``b * k^2`` traced rays, so every ``k`` spends the same ``b`` rays per mesh pixel.
- ``filtered``: the fair baseline for ``budget``. Every pixel gets ``b`` rays and the same blur, so any gain of ``budget``
  over ``filtered`` comes from sparsity and not merely from filtering noise.

The signal is the whole radiance of mesh pixels, including any semi-transparent Gaussians in front of them, because the
kernel does not separate the two. The reconstruction is deliberately naive, so headroom is if anything understated.

Each 32 px tile gets a safe rate: the largest ``k`` whose clean FLIP stays within ``--flip-threshold``. The threshold is
a working choice, not a visibility threshold, so nearby values are reported too. The pre-stated rule: variable rate is
worth building if the safe rate (``1 / k^2``) varies at least 4x between the 10th and 90th percentile of tiles AND some
predictor reaches a Spearman ``|rho| >= 0.5`` with ``log2 k``.

With pycvvdp installed the rule is applied to CVVDP instead, using its raw heatmap (``1 - JOD / 10`` per pixel, so 0.1 is
one JOD lost locally). FLIP has no masking model, so a FLIP safe rate only measures signal bandwidth. The direct masking
test asks whether CVVDP finds the same display error less visible in high-contrast tiles: the Spearman correlation of
the masking weight (high in flat tiles) with visibility per unit of RMS error. Masking predicts a positive value.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from threedgrut_playground.adaptive.image_ops import normalized_blur2d

from .core_api import (
    ContrastMaskingTerm,
    SampleStats,
    WeightContext,
    center_view_dirs,
    luminance,
)
from .ldr import ensure_outside_repo, save_png, to_display_ldr
from .metrics import _cvvdp_metric, cvvdp_available, cvvdp_jod, flip_error_map, psnr
from .reference import reference_bundle
from .sampling import trace_only_renderer, traced_rounds, visible_mesh_hits
from .scenes import (
    GARDEN_GLOSSY,
    SphereSpec,
    assign_hits_to_objects,
    glossy_layout,
    scene_frame,
    setup_glossy_scene,
)
from .viewing import eccentricity_deg, image_center_dir, ppd_from_camera

Tensor = torch.Tensor

REPO_ROOT = Path(__file__).resolve().parents[3]
ROUGHNESS_LADDER = (0.05, 0.2, 0.4, 0.7)
FLOOR_ROUGHNESS = 0.5  # brushed_copper
TILE_SIZE = 32
MIN_TILE_COVERAGE = 0.5
BUDGET_SAMPLE_BASE = 0
FILL_DOUBLINGS = 5
RATE_SPREAD_THRESHOLD = 4.0
RHO_THRESHOLD = 0.5
LADDER_PRESET = replace(
    GARDEN_GLOSSY,
    spheres=tuple(
        SphereSpec(f"ladder_r{r}", a, 0.3, 0.2) for r, a in zip(ROUGHNESS_LADDER, (-0.66, -0.22, 0.22, 0.66))
    ),
)


def register_ladder_materials(engine) -> None:
    """Register one metal per ladder roughness, so that roughness is the only material difference between spheres."""
    materials = engine.primitives.registered_materials
    for roughness in ROUGHNESS_LADDER:
        name = f"ladder_r{roughness}"
        if name not in materials:
            materials[name] = replace(
                materials["rose_gold"], material_id=len(materials), metallic_factor=1.0, roughness_factor=roughness
            )


def procedural_sky(device, height: int = 256, width: int = 512) -> Tensor:
    """Bright ``[H, W, 4]`` envmap with soft blobs, symmetric in elevation so it needs no up-axis convention."""
    v, u = torch.meshgrid(
        (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height,
        (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width,
        indexing="ij",
    )
    tint = torch.tensor([0.75, 0.85, 1.0], device=device)
    level = 0.7 + 0.3 * torch.cos(2.0 * torch.pi * v) ** 2
    blobs = torch.zeros_like(u)
    for centre_u, centre_v in ((0.1, 0.3), (0.35, 0.7), (0.6, 0.3), (0.85, 0.7)):
        du = torch.remainder(u - centre_u + 0.5, 1.0) - 0.5
        blobs = blobs + 1.5 * torch.exp(-(du * du + (v - centre_v) ** 2) / (2.0 * 0.03**2))
    rgb = (level + blobs)[..., None] * tint
    return torch.cat([rgb, torch.ones_like(rgb[..., :1])], dim=-1)


def stratified_keep(height: int, width: int, k: int, seed: int) -> Tensor:
    """``[H, W]`` bool with one pseudo-random pixel kept per ``k x k`` cell (every pixel for ``k = 1``)."""
    generator = torch.Generator().manual_seed(seed)
    rows = height // k + 1
    cols = width // k + 1
    cell_y = torch.arange(rows)[:, None].expand(rows, cols)
    cell_x = torch.arange(cols)[None, :].expand(rows, cols)
    y = cell_y * k + torch.randint(0, k, (rows, cols), generator=generator)
    x = cell_x * k + torch.randint(0, k, (rows, cols), generator=generator)
    inside = (y < height) & (x < width)
    keep = torch.zeros(height, width, dtype=torch.bool)
    keep[y[inside], x[inside]] = True
    return keep


def reconstruct(values: Tensor, keep: Tensor, object_ids: Tensor, k: int) -> Tensor:
    """Fill every pixel of each object (``object_ids >= 0``) from the kept pixels of that object.

    Normalized Gaussian with ``sigma = 0.6 k``; pixels with no support retry at doubled sigma. ``k = 1`` returns
    ``values``. Pixels outside every object are returned unchanged.
    """
    if k <= 1:
        return values.clone()
    out = values.clone()
    for object_id in object_ids[object_ids >= 0].unique():
        region = object_ids == object_id
        kept = keep & region
        if not bool(kept.any()):  # an object smaller than a cell: fall back to its own pixels' mean
            out[region] = values[region].mean(dim=0)
            continue
        missing = region.clone()
        sigma = 0.6 * k
        for _ in range(FILL_DOUBLINGS):
            support = None
            for channel in range(3):
                filled, support = normalized_blur2d(values[..., channel], kept, sigma)
                out[..., channel] = torch.where(missing & (support > 1e-4), filled, out[..., channel])
            missing = missing & ~(support > 1e-4)
            if not bool(missing.any()):
                break
            sigma *= 2.0
    return out


def tile_means(values: Tensor, region: Tensor, tile: int = TILE_SIZE) -> tuple[Tensor, Tensor]:
    """Mean of ``values [H, W]`` over ``region`` per tile, and each tile's coverage of ``region``."""
    weights = region.to(torch.float64)
    sums = torch.nn.functional.avg_pool2d((values.to(torch.float64) * weights)[None, None], tile)[0, 0]
    coverage = torch.nn.functional.avg_pool2d(weights[None, None], tile)[0, 0]
    return sums / coverage.clamp_min(1e-9), coverage


def safe_rate(flip_by_k: Dict[int, Tensor], yardstick: Tensor) -> Tensor:
    """Per tile, the largest ``k`` in the ascending keys whose error never exceeded ``yardstick`` up to that ``k``."""
    ks = sorted(flip_by_k)
    safe = torch.full_like(yardstick, float(ks[0]))
    alive = torch.ones_like(yardstick, dtype=torch.bool)
    for k in ks:
        alive = alive & (flip_by_k[k] <= yardstick)
        safe = torch.where(alive, torch.full_like(safe, float(k)), safe)
    return safe


def decide(safe_k: Tensor, rhos: Dict[str, float]) -> Dict[str, object]:
    """Apply the pre-stated rule to the safe ``k`` of each tile."""
    rates = 1.0 / safe_k.to(torch.float64) ** 2
    p10, p90 = torch.quantile(rates, 0.1).item(), torch.quantile(rates, 0.9).item()
    spread = p90 / p10
    best = max(rhos, key=_abs_rho(rhos)) if rhos else None
    best_rho = rhos[best] if best else float("nan")
    return {
        "rate_p10": p10,
        "rate_p90": p90,
        "rate_spread": spread,
        "best_predictor": best,
        "best_rho": best_rho,
        "worth_building": bool(spread >= RATE_SPREAD_THRESHOLD and abs(best_rho) >= RHO_THRESHOLD),
    }


def _abs_rho(rhos: Dict[str, float]):
    def key(name: str) -> float:
        value = abs(rhos[name])
        return value if value == value else -1.0

    return key


_CVVDP_HEATMAP_METRICS: Dict[tuple, object] = {}


def cvvdp_heatmap(test_ldr: Tensor, ref_ldr: Tensor, ppd: float) -> Tensor:
    """CVVDP raw heatmap ``[H, W]`` (``1 - JOD / 10`` per pixel), on the input device."""
    import pycvvdp

    height, width = int(ref_ldr.shape[0]), int(ref_ldr.shape[1])
    key = (float(ppd), height, width)
    if key not in _CVVDP_HEATMAP_METRICS:
        base = _cvvdp_metric(ppd, height, width, ref_ldr.device)
        _CVVDP_HEATMAP_METRICS[key] = pycvvdp.cvvdp(
            display_name="adaptive_eval",
            display_photometry=base.display_photometry,
            display_geometry=base.display_geometry,
            device=ref_ldr.device,
            heatmap="raw",
        )
    _, stats = _CVVDP_HEATMAP_METRICS[key].predict(test_ldr.float(), ref_ldr.float(), dim_order="HWC")
    return stats["heatmap"][0, 0, 0].to(ref_ldr.device, torch.float32)


def threshold_decisions(by_k: Dict[int, Tensor], coverage: Tensor, usable: Tensor, predictors, threshold: float):
    """Safe ``k`` per tile at ``threshold``, its Spearman correlations and the decision, also at 0.5x and 1.5x."""
    result = {}
    for factor in (1.0, 0.5, 1.5):
        safe = safe_rate(by_k, torch.full_like(coverage, threshold * factor))[usable]
        rhos = {name: spearman(values[usable], torch.log2(safe)) for name, values in predictors.items()}
        ks = sorted(by_k)
        result[f"{threshold * factor:.3f}"] = {
            "safe_k_histogram": {str(k): int((safe == k).sum()) for k in ks},
            "spearman_with_log2_safe_k": rhos,
            **decide(safe, rhos),
        }
    return result


def object_id_map(renderer, camera, layout, device) -> Tensor:
    """``[H, W]`` object index of each pixel from one central-ish jittered ray per pixel: ``-1`` off every object."""
    height, width = int(camera.height), int(camera.width)
    counts = torch.ones(height * width, dtype=torch.int64, device=device)
    base = torch.full((height * width,), 2**31 - 2**21, dtype=torch.int64, device=device)
    ids = torch.full((height * width,), -1, dtype=torch.int64, device=device)
    for batch, out in traced_rounds(renderer, camera, counts, base, kind="object_ids"):
        visible = visible_mesh_hits(out).to(device)
        t = out["aov_first_hit_t"][:, 0].to(device)[visible]
        points = batch.origins.to(device)[visible] + t[:, None] * batch.directions.to(device)[visible]
        ids[batch.pixel_id.to(device).long()[visible]] = assign_hits_to_objects(points, layout).to(device)
    return ids.reshape(height, width)


def traced_means(renderer, camera, keep: Tensor, spp: int, device) -> Tensor:
    """Mean HDR ``[H, W, 3]`` of ``spp`` traced rays on the kept pixels (0 elsewhere)."""
    height, width = int(camera.height), int(camera.width)
    counts = keep.reshape(-1).to(device).to(torch.int64) * int(spp)
    base = torch.full((height * width,), BUDGET_SAMPLE_BASE, dtype=torch.int64, device=device)
    stats = SampleStats(height, width, device, track_halves=False)
    for batch, out in traced_rounds(renderer, camera, counts, base, kind="reflection_rate"):
        stats.add(batch.pixel_id.long(), out["rgb"], batch.sample_index.long())
    return stats.mean()


def composite(reconstructed: Tensor, reference: Tensor, region: Tensor) -> Tensor:
    """``reconstructed`` on ``region``, the reference elsewhere (Gaussian-only pixels are not under test)."""
    return torch.where(region[..., None], reconstructed, reference)


def spearman(a: Tensor, b: Tensor) -> float:
    """Spearman rank correlation of two 1D tensors (``nan`` if either is constant)."""
    from scipy.stats import spearmanr

    return float(spearmanr(a.cpu().numpy(), b.cpu().numpy()).statistic)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="Gaussian checkpoint")
    parser.add_argument("--out-dir", required=True, help="output folder, outside the repository")
    parser.add_argument("--resolution", type=int, default=512, help="square image size")
    parser.add_argument("--camera", default="close", choices=sorted(LADDER_PRESET.camera_offsets))
    parser.add_argument("--ks", default="1,2,4,8,16", help="cell sizes; the rate is 1/k^2")
    parser.add_argument("--budgets", default="1,4", help="rays per mesh pixel for the equal-budget arm")
    parser.add_argument("--flip-threshold", type=float, default=0.1, help="tile FLIP that counts as acceptable")
    parser.add_argument("--cvvdp-threshold", type=float, default=0.1, help="tile CVVDP heatmap; 0.1 is one JOD lost")
    parser.add_argument("--ref-spp", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def run(args: argparse.Namespace, engine=None) -> Dict[str, object]:
    out_dir = ensure_outside_repo(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ks = sorted(int(k) for k in args.ks.split(","))
    if ks[0] != 1:
        raise ValueError("--ks must start at 1")
    budgets = [int(b) for b in args.budgets.split(",")]

    if engine is None:
        from threedgrut_playground.engine import Engine3DGRUT

        engine = Engine3DGRUT(
            gs_object=str(args.ckpt),
            mesh_assets_folder=str(REPO_ROOT / "threedgrut_playground" / "assets"),
            default_config="apps/colmap_3dgrt.yaml",
        )
    device = torch.device(engine.device)
    register_ladder_materials(engine)
    engine.environment.envmap = procedural_sky(device)
    engine.environment.current_name = "procedural_sky"
    cameras = setup_glossy_scene(engine, LADDER_PRESET)
    layout = glossy_layout(scene_frame(engine.scene_mog.positions.detach().cpu()), LADDER_PRESET)
    camera = cameras[args.camera].to_camera(args.resolution, args.resolution)
    renderer = trace_only_renderer(engine)
    reference = reference_bundle(renderer, camera, args.ref_spp, out_dir, ckpt_path=str(args.ckpt))
    ppd = ppd_from_camera(camera)
    ref_hdr = reference.hdr
    ref_ldr = to_display_ldr(ref_hdr)
    save_png(ref_ldr, out_dir / "reference.png")

    object_ids = object_id_map(renderer, camera, layout, device)
    object_ids = torch.where(reference.mesh_mask, object_ids, torch.full_like(object_ids, -1))
    region = object_ids >= 0
    names = layout.object_names()
    height, width = int(camera.height), int(camera.width)
    keeps = {k: stratified_keep(height, width, k, args.seed + k).to(device) for k in ks}

    rows: List[Dict[str, object]] = []

    def score(arm: str, k: int, budget: float, image: Tensor) -> Tensor:
        ldr = to_display_ldr(image)
        flip = flip_error_map(ldr, ref_ldr, ppd)
        row: Dict[str, object] = {"arm": arm, "k": k, "rate": 1.0 / k**2, "budget": budget}
        row["flip_mesh"] = flip[region].mean().item()
        row["psnr_mesh"] = float(psnr(ldr, ref_ldr, region))
        for index, name in enumerate(names):
            selected = object_ids == index
            row[f"flip_{name}"] = flip[selected].mean().item() if bool(selected.any()) else float("nan")
        jod = cvvdp_jod(ldr, ref_ldr, ppd) if cvvdp_available() else None
        row["cvvdp"] = float("nan") if jod is None else float(jod)
        rows.append(row)
        return flip

    use_cvvdp = cvvdp_available()
    clean_flip: Dict[int, Tensor] = {}
    clean_vis: Dict[int, Tensor] = {}
    clean_err: Dict[int, Tensor] = {}
    for k in ks:
        image = composite(reconstruct(ref_hdr, keeps[k], object_ids, k), ref_hdr, region)
        clean_flip[k] = score("clean", k, 0.0, image)
        if use_cvvdp:
            ldr = to_display_ldr(image)
            clean_vis[k] = cvvdp_heatmap(ldr, ref_ldr, ppd)
            clean_err[k] = (luminance(ldr) - luminance(ref_ldr)) ** 2
            for index, name in enumerate(names):
                selected = object_ids == index
                rows[-1][f"vis_{name}"] = clean_vis[k][selected].mean().item() if bool(selected.any()) else float("nan")
        save_png(to_display_ldr(image), out_dir / f"clean_k{k}.png")

    for budget in budgets:
        for k in ks:
            means = traced_means(renderer, camera, keeps[k], budget * k * k, device)
            image = composite(reconstruct(means, keeps[k], object_ids, k), ref_hdr, region)
            score("budget", k, budget, image)
            save_png(to_display_ldr(image), out_dir / f"budget{budget}_k{k}.png")
            if k == ks[0]:
                dense_means = means
            else:
                filtered = reconstruct(dense_means, keeps[1], object_ids, k)
                score("filtered", k, budget, composite(filtered, ref_hdr, region))

    # Per-tile safe rate and its predictors.
    flip_by_k = {k: tile_means(clean_flip[k], region)[0] for k in ks}
    coverage = tile_means(clean_flip[ks[0]], region)[1]
    usable = coverage >= MIN_TILE_COVERAGE
    safe_k = safe_rate(flip_by_k, torch.full_like(coverage, args.flip_threshold))

    view_dirs = center_view_dirs(camera)
    masking = ContrastMaskingTerm()(
        WeightContext(
            ref_hdr, luminance(ref_hdr), reference.mesh_mask, view_dirs, camera, None, reference.display_var_mean
        )
    )
    roughness_by_id = torch.tensor((FLOOR_ROUGHNESS,) + ROUGHNESS_LADDER, device=device, dtype=torch.float32)
    roughness = roughness_by_id[object_ids.clamp_min(0)]
    eccentricity = eccentricity_deg(view_dirs, image_center_dir(camera))
    predictors = {
        "roughness": tile_means(roughness, region)[0],
        "masking_weight": tile_means(masking, region)[0],
        "eccentricity_deg": tile_means(eccentricity, region)[0],
        "luma": tile_means(luminance(ref_ldr), region)[0],
    }
    log_k = torch.log2(safe_k[usable])
    rhos = {name: spearman(values[usable], log_k) for name, values in predictors.items()}
    flip_decisions = threshold_decisions(flip_by_k, coverage, usable, predictors, args.flip_threshold)
    decision = flip_decisions[f"{args.flip_threshold:.3f}"]
    cvvdp_summary = None
    if use_cvvdp:
        vis_by_k = {k: tile_means(clean_vis[k], region)[0] for k in ks}
        cvvdp_decisions = threshold_decisions(vis_by_k, coverage, usable, predictors, args.cvvdp_threshold)
        decision = cvvdp_decisions[f"{args.cvvdp_threshold:.3f}"]
        masking_test = {}
        for k in ks[1:]:
            rms = tile_means(clean_err[k], region)[0].sqrt()
            per_error = vis_by_k[k][usable] / rms[usable].clamp_min(1e-6)
            masking_test[str(k)] = spearman(predictors["masking_weight"][usable], per_error)
        cvvdp_summary = {"decisions": cvvdp_decisions, "masking_test_rho_by_k": masking_test}

    with open(out_dir / "results.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), restval="")
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "tiles.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["safe_k_flip", *predictors])
        columns = [safe_k[usable].tolist(), *(values[usable].tolist() for values in predictors.values())]
        writer.writerows(zip(*columns))
    summary = {
        "ckpt": str(args.ckpt),
        "camera": args.camera,
        "ppd": ppd,
        "tiles": int(usable.sum()),
        "flip": flip_decisions,
        "cvvdp": cvvdp_summary,
        "decision": decision,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _print_table(rows, summary, names)
    return summary


def _print_table(rows: Sequence[Dict[str, object]], summary: Dict[str, object], names: Sequence[str]) -> None:
    print("arm        k   rate    budget  flip_mesh  psnr_mesh  " + "  ".join(f"{n:>8}" for n in names) + "  cvvdp")
    for row in rows:
        cells = "  ".join(f"{row[f'flip_{n}']:8.4f}" for n in names)
        print(
            f"{row['arm']:<9}{row['k']:3d}  {row['rate']:.4f}  {row['budget']:6.1f}  {row['flip_mesh']:9.4f}"
            f"  {row['psnr_mesh']:9.2f}  {cells}  {row['cvvdp']:.3f}"
        )
    if "vis_floor" in rows[0]:
        print("clean arm, mean CVVDP heatmap per object (0.1 = one JOD lost):")
        for row in rows:
            if row["arm"] == "clean":
                print(f"  k={row['k']:<3d}" + "  ".join(f"{n} {row[f'vis_{n}']:.3f}" for n in names))
    print("tiles:", summary["tiles"])
    print("flip:", json.dumps(summary["flip"], indent=1))
    print("cvvdp:", json.dumps(summary["cvvdp"], indent=1))
    print("decision:", summary["decision"]["worth_building"])


def main(argv=None) -> int:
    run(parse_args(argv))
    return 0
