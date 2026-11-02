# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Budget-sweep CLI and run loop (spec Goal 4, interface C).

For every camera of a glossy preset: a cached jittered reference, then for every budget and seed one render per
allocator at the same pilot-inclusive ray budget and sample offset, scored against the reference (``results.csv``),
then paired 95% intervals (``paired.csv``), the pre-stated decision rule (``decision.json``) and plots.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import platform
import re
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .analysis import (
    CAVEATS,
    CSV_COLUMNS,
    DECISION_RULE,
    PAIRED_COLUMNS,
    PLAIN_ALLOCATOR,
    TOPUP_SHARE_WARN,
    BudgetMismatchError,
    CsvLog,
    assemble_row,
    convergence_slope,
    decide,
    default_pairings,
    paired_differences,
    plot_curves,
)
from .core_api import (
    TOPUP_ROUNDS,
    AdaptiveRenderer,
    CostTelemetry,
    MapDumper,
    PerceptualVarianceAllocator,
    SimpleBudget,
    StochasticRounder,
    UniformAllocator,
    VarianceAllocator,
    VarianceOfMeanEstimator,
    center_view_dirs,
    make_weight_terms,
    telemetry_cost_fits,
)
from .ldr import ensure_outside_repo, save_png, to_display_ldr
from .metrics import (
    PRIMARY_METRIC,
    cvvdp_available,
    evaluate_image,
    ref_noise_flags,
    reference_weights,
)
from .reference import REF_MAX_SPP, reference_bundle
from .sampling import (
    PlumbingError,
    check_plumbing,
    plain_uniform_render,
    trace_only_renderer,
)
from .scenes import PRESETS, setup_glossy_scene
from .viewing import eccentricity_deg, image_center_dir, ppd_from_camera

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_ALLOCATORS = "uniform,variance_g1,variance_g0.5,perceptual_g1,perceptual_g0.5,plain_uniform"

ELIGIBILITY_NOTE = (
    "In plan B every allocator treats pixels with weight 0 as ineligible, and `DisplayClipTerm` is always part of the "
    "weight terms, so `uniform` and `variance_g*` also give no adaptive, defensive or top-up rays to pixels whose mean "
    "is clipped above display white; 'uniform' means uniform over the unclipped mesh region, not over all mesh pixels. "
    "`plain_uniform` has no such exclusion."
)

SEED_STRIDE = 2**24  # test renders use sample_offset = seed * 2^24 (seeds 0..62)
MAX_SEED = 62
WARMUP_SEED = 63  # discarded warm-up renders
REF_SPP_FACTOR = 8  # ref_spp >= 8 * n_cap
REF_NONMESH_FACTOR = 16  # ref_spp_nonmesh >= 16 * pilot_spp
CAPACITY_MARGIN = 0.02
RECOMMENDED_SEEDS = 5

_GAMMA_RE = re.compile(r"^(variance|perceptual)_g(\d+(?:\.\d+)?)$")
_ALLOCATOR_CLASSES = {"variance": VarianceAllocator, "perceptual": PerceptualVarianceAllocator}


# ---------------------------------------------------------------------------------------------------------------------
# allocator names, caps, capacity
# ---------------------------------------------------------------------------------------------------------------------


def parse_allocator(name: str) -> Tuple[str, Optional[float]]:
    """``"uniform"`` / ``"plain_uniform"`` -> ``(name, None)``; ``"variance_g<gamma>"`` / ``"perceptual_g<gamma>"`` with
    ``gamma`` in ``(0, 2]`` -> ``(class key, gamma)``; anything else raises ``ValueError``."""
    name = str(name).strip()
    if name in ("uniform", PLAIN_ALLOCATOR):
        return name, None
    m = _GAMMA_RE.match(name)
    if m is None:
        raise ValueError(
            f"unknown allocator {name!r}; expected 'uniform', 'plain_uniform', 'variance_g<gamma>' or "
            f"'perceptual_g<gamma>'"
        )
    gamma = float(m.group(2))
    if not 0.0 < gamma <= 2.0:
        raise ValueError(f"allocator {name!r}: gamma must be in (0, 2], got {gamma:g}")
    return m.group(1), gamma


def n_cap(pilot_spp: int, iterations: int, max_per_pixel: int) -> int:
    """Upper bound on any pixel's rays in one render: the pilot, every adaptive launch and both top-up rounds at the
    per-launch cap."""
    return int(pilot_spp) + int(iterations) * int(max_per_pixel) + TOPUP_ROUNDS * int(max_per_pixel)


def capacity_rays(
    n_pixels: int, n_unclipped: int, n_clipped: int, pilot_spp: int, iterations: int, max_per_pixel: int
) -> int:
    """Rays one render can absorb: every pixel its pilot, an unclipped region pixel ``(iterations + TOPUP_ROUNDS)``
    capped launches, a clipped region pixel one capped launch (top-up widening), non-region pixels nothing more."""
    return int(pilot_spp) * int(n_pixels) + int(max_per_pixel) * (
        (int(iterations) + TOPUP_ROUNDS) * int(n_unclipped) + int(n_clipped)
    )


def absorbable_budget(
    n_pixels: int,
    n_unclipped: int,
    n_clipped: int,
    pilot_spp: int,
    iterations: int,
    max_per_pixel: int,
    margin: float = CAPACITY_MARGIN,
) -> int:
    """:func:`capacity_rays` less a ``margin`` safety fraction (floored)."""
    cap = capacity_rays(n_pixels, n_unclipped, n_clipped, pilot_spp, iterations, max_per_pixel)
    return int(math.floor(cap * (1.0 - float(margin))))


def split_budgets(budgets: Sequence[int], n_pixels: int, absorbable: int) -> Tuple[List[int], List[Dict[str, int]]]:
    """``(kept, skipped)``: budgets whose ``budget_spp * n_pixels`` rays fit ``absorbable``, and a record per budget
    that does not."""
    kept: List[int] = []
    skipped: List[Dict[str, int]] = []
    for b in budgets:
        rays = int(b) * int(n_pixels)
        if rays <= int(absorbable):
            kept.append(int(b))
        else:
            skipped.append({"budget_spp": int(b), "budget_rays": rays, "absorbable_rays": int(absorbable)})
    return kept, skipped


def build_renderer(
    engine,
    allocator: str,
    seed: int,
    terms: list,
    max_per_pixel: int,
    telemetry=None,
    dumper=None,
) -> AdaptiveRenderer:
    """A pipeline renderer for ``allocator`` (``uniform``, ``variance_g<gamma>`` or ``perceptual_g<gamma>``).

    Every pipeline allocator shares ``VarianceOfMeanEstimator()``, ``terms``, ``StochasticRounder(seed=seed)`` and
    ``SimpleBudget(max_per_pixel, seed=seed)``; only the allocator and ``gamma`` differ.

    Eligibility (``ELIGIBILITY_NOTE``): In plan B every allocator treats pixels with weight 0 as ineligible, and
    ``DisplayClipTerm`` is always part of the weight terms, so ``uniform`` and ``variance_g*`` also give no adaptive,
    defensive or top-up rays to pixels whose mean is clipped above display white; 'uniform' means uniform over the
    unclipped mesh region, not over all mesh pixels. ``plain_uniform`` has no such exclusion.
    """
    key, gamma = parse_allocator(allocator)
    if key == "uniform":
        alloc = UniformAllocator()
    elif key in _ALLOCATOR_CLASSES:
        alloc = _ALLOCATOR_CLASSES[key](gamma=gamma)
    else:
        raise ValueError(f"{allocator!r} is not a pipeline allocator (plain_uniform uses trace_only_renderer)")
    return AdaptiveRenderer(
        engine,
        VarianceOfMeanEstimator(),
        terms,
        alloc,
        StochasticRounder(seed=int(seed)),
        SimpleBudget(int(max_per_pixel), seed=int(seed)),
        telemetry=telemetry,
        dumper=dumper,
    )


# ---------------------------------------------------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------------------------------------------------


def _int_list(text: str) -> List[int]:
    try:
        values = [int(v) for v in str(text).split(",") if v.strip() != ""]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {text!r}")
    if not values:
        raise argparse.ArgumentTypeError(f"expected at least one integer, got {text!r}")
    return values


def _name_list(text: str) -> List[str]:
    values = [v.strip() for v in str(text).split(",") if v.strip() != ""]
    if not values:
        raise argparse.ArgumentTypeError(f"expected at least one name, got {text!r}")
    return values


def _allocator_list(text: str) -> List[str]:
    names = _name_list(text)
    for name in names:
        try:
            parse_allocator(name)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc))
    return names


def parse_resolution(s: str) -> Tuple[int, int]:
    """``"WxH"`` -> ``(W, H)`` (positive integers)."""
    try:
        w, h = (int(v) for v in str(s).lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT such as 512x512, got {s!r}")
    if w < 1 or h < 1:
        raise argparse.ArgumentTypeError(f"resolution must be positive, got {s!r}")
    return w, h


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_budget_sweep.py",
        description=(
            "Adaptive-sampling budget sweep: equal-ray comparisons of allocators against a cached jittered reference, "
            "with paired 95%% intervals and a pre-stated decision rule. " + ELIGIBILITY_NOTE.replace("%", "%%")
        ),
    )
    p.add_argument("--ckpt", required=True, help="Gaussian checkpoint (must exist)")
    p.add_argument("--out-dir", required=True, help="output root, outside the repository (sweep_<time>/ inside)")
    p.add_argument(
        "--budgets",
        type=_int_list,
        default="8,16,32,64,128",
        help="average rays per image pixel including the pilot; integers > --pilot-spp (plain_uniform uses exactly "
        "b spp)",
    )
    p.add_argument(
        "--allocators",
        type=_allocator_list,
        default=DEFAULT_ALLOCATORS,
        help="uniform, plain_uniform, variance_g<gamma>, perceptual_g<gamma> (gamma in (0, 2]); must include uniform",
    )
    p.add_argument(
        "--weight-model",
        choices=sorted(PRIMARY_METRIC),
        default="cielab",
        help="cielab or cielab_nomask (masking ablation; primary metric FLIP), or tvi (primary metric CVVDP; requires "
        "pycvvdp)",
    )
    p.add_argument("--resolution", type=parse_resolution, default="512x512", help="WxH (ppd = H / 60 for the presets)")
    p.add_argument("--seeds", type=_int_list, default="0,1,2,3,4", help=f"seeds in [0, {MAX_SEED}]")
    p.add_argument("--pilot-spp", type=int, default=4, help="pilot rays per pixel (>= 4)")
    p.add_argument("--iterations", type=int, default=3, help="adaptive iterations (>= 1)")
    p.add_argument("--preset", default="garden_glossy", help=f"scene preset ({', '.join(sorted(PRESETS))})")
    p.add_argument("--cameras", type=_name_list, default="close", help="comma-separated camera names of the preset")
    p.add_argument("--max-per-pixel", type=int, default=192, help="SimpleBudget cap on new rays per pixel per launch")
    p.add_argument("--defensive-fraction", type=float, default=0.15, help="defensive uniform share in [0, 1)")
    p.add_argument(
        "--ref-spp",
        type=int,
        default=8192,
        help="reference rays per mesh pixel; must be >= 8 * n_cap (n_cap = pilot + (iterations + 2) * max-per-pixel)",
    )
    p.add_argument("--ref-spp-nonmesh", type=int, default=256, help="reference rays per non-mesh pixel (>= 16 * pilot)")
    p.add_argument(
        "--allow-weak-reference", action="store_true", help="allow --ref-spp below 8 * n_cap (with a warning)"
    )
    p.add_argument(
        "--budget-tol", type=float, default=0.01, help="allowed relative deviation of rays launched from the budget"
    )
    p.add_argument("--no-cvvdp", action="store_true", help="skip CVVDP even if pycvvdp is installed")
    p.add_argument(
        "--dump-maps",
        action="store_true",
        help="write every pipeline render's per-iteration maps (MapDumper) under sweep_*/dumps; the dumps are written "
        "inside the timed render, so wall_ms then includes disk I/O: never use it for timed comparisons",
    )
    return p


def validate_args(args, parser: argparse.ArgumentParser) -> None:
    """Reject invalid arguments with ``parser.error`` (``SystemExit``) before any engine import or file write."""
    err = parser.error
    if not Path(args.ckpt).is_file():
        err(f"--ckpt {args.ckpt} does not exist")
    try:
        ensure_outside_repo(args.out_dir, REPO_ROOT)
    except ValueError as exc:
        err(f"--out-dir: {exc}")
    if args.pilot_spp < 4:
        err(f"--pilot-spp must be >= 4, got {args.pilot_spp}")
    if args.iterations < 1:
        err(f"--iterations must be >= 1, got {args.iterations}")
    if args.max_per_pixel < 1:
        err(f"--max-per-pixel must be >= 1, got {args.max_per_pixel}")
    if not 0.0 <= args.defensive_fraction < 1.0:
        err(f"--defensive-fraction must be in [0, 1), got {args.defensive_fraction}")
    if not args.budget_tol > 0.0:
        err(f"--budget-tol must be > 0, got {args.budget_tol}")
    bad = [b for b in args.budgets if b <= args.pilot_spp]
    if bad:
        err(f"--budgets must all be > --pilot-spp={args.pilot_spp} (pilot-inclusive), got {bad}")
    if len(set(args.budgets)) != len(args.budgets):
        err(f"--budgets has duplicates: {args.budgets}")
    if "uniform" not in args.allocators:
        err("--allocators must include 'uniform' (reference-noise flags and the pairings need it)")
    if len(set(args.allocators)) != len(args.allocators):
        err(f"--allocators has duplicates: {args.allocators}")
    bad = [s for s in args.seeds if not 0 <= s <= MAX_SEED]
    if bad:
        err(f"--seeds must be in [0, {MAX_SEED}] (seed {WARMUP_SEED} is the warm-up), got {bad}")
    if len(set(args.seeds)) != len(args.seeds):
        err(f"--seeds has duplicates: {args.seeds}")
    if args.preset not in PRESETS:
        err(f"--preset {args.preset!r} unknown; expected one of {sorted(PRESETS)}")
    known = PRESETS[args.preset].camera_offsets
    bad = [c for c in args.cameras if c not in known]
    if bad:
        err(f"--cameras {bad} unknown for preset {args.preset!r}; expected some of {sorted(known)}")
    if args.weight_model == "tvi" and (args.no_cvvdp or not cvvdp_available()):
        err("--weight-model tvi needs CVVDP (its primary metric): install pycvvdp and do not pass --no-cvvdp")
    cap = n_cap(args.pilot_spp, args.iterations, args.max_per_pixel)
    if args.ref_spp < REF_SPP_FACTOR * cap:
        msg = (
            f"--ref-spp {args.ref_spp} < {REF_SPP_FACTOR} * n_cap = {REF_SPP_FACTOR * cap}: the reference's variance "
            f"of the mean is not <= 1/{REF_SPP_FACTOR} of every test pixel's"
        )
        if not args.allow_weak_reference:
            err(msg + " (pass --allow-weak-reference to run anyway)")
        warnings.warn(f"weak reference: {msg}", UserWarning, stacklevel=2)
    if args.ref_spp > REF_MAX_SPP:
        err(f"--ref-spp must be <= {REF_MAX_SPP} (reserved reference sample-index range), got {args.ref_spp}")
    if args.ref_spp_nonmesh < REF_NONMESH_FACTOR * args.pilot_spp:
        err(f"--ref-spp-nonmesh must be >= {REF_NONMESH_FACTOR} * pilot = {REF_NONMESH_FACTOR * args.pilot_spp}")
    if args.ref_spp_nonmesh > args.ref_spp:
        err(f"--ref-spp-nonmesh {args.ref_spp_nonmesh} must be <= --ref-spp {args.ref_spp}")
    if len(args.seeds) < RECOMMENDED_SEEDS:
        warnings.warn(
            f"only {len(args.seeds)} seeds: the paired 95% intervals are wide with fewer than {RECOMMENDED_SEEDS}",
            UserWarning,
            stacklevel=2,
        )


# ---------------------------------------------------------------------------------------------------------------------
# run loop helpers
# ---------------------------------------------------------------------------------------------------------------------


def _jsonable(obj: Any) -> Any:
    """Plain JSON (non-finite floats -> ``None``; tuples, paths and tensors converted)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        obj = obj.item()
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(_jsonable(obj), fh, indent=2)


def _git_head() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10, check=True
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _versions() -> Dict[str, Optional[str]]:
    from importlib import metadata

    import torch

    out: Dict[str, Optional[str]] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    for dist in ("flip-evaluator", "kaolin", "numpy", "cvvdp"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = None
    return out


def _sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _make_sweep_dir(out_dir: Path) -> Path:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    for k in range(100):
        d = out_dir / (f"sweep_{stamp}" if k == 0 else f"sweep_{stamp}_{k}")
        try:
            d.mkdir(parents=True, exist_ok=False)
            return d
        except FileExistsError:
            continue
    raise RuntimeError(f"could not create a fresh sweep directory under {out_dir}")


def _read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _save_counts_png(counts, path: Path) -> None:
    """Rays per pixel as a grayscale PNG normalized by the maximum count."""
    c = counts.float()
    save_png((c / c.max().clamp_min(1.0))[..., None].expand(*c.shape, 3), path)


def _verdict_table(verdicts: List[Dict[str, Any]], paired: List[Dict[str, Any]], better: str) -> str:
    """One line per pair: per-budget interval sign (``+`` win, ``-`` loss, ``0`` straddles zero / no interval), the
    counts, the verdict and the maximum top-up share (``*`` marks the top-up caveat)."""

    def sign(p) -> str:
        lo, hi = float(p["ci95_low"]), float(p["ci95_high"])
        if not (math.isfinite(lo) and math.isfinite(hi)):
            return "0"
        if (hi < 0 and better == "lower") or (lo > 0 and better == "higher"):
            return "+"
        if hi < 0 or lo > 0:
            return "-"
        return "0"

    lines = [
        f"{'camera':<9} {'pair':<34} {'per budget (+win -loss 0 none)':<44} {'W/L/n':<8} {'verdict':<13} topup_share"
    ]
    for v in verdicts:
        ps = sorted(
            (
                p
                for p in paired
                if p["camera"] == v["camera"]
                and p["weight_model"] == v["weight_model"]
                and p["allocator"] == v["allocator"]
                and p["baseline"] == v["baseline"]
                and p["metric"] == v["metric"]
            ),
            key=lambda p: float(p["budget_spp"]),
        )
        per = " ".join(f"{float(p['budget_spp']):g}:{sign(p)}" for p in ps)
        share = v["max_topup_share"]
        share_s = "nan" if not math.isfinite(share) else f"{share:.3f}"
        mark = " * topup_caveat" if v["topup_caveat"] else ""
        lines.append(
            f"{v['camera']:<9} {v['allocator'] + ' vs ' + v['baseline']:<34} {per:<44} "
            f"{v['wins']}/{v['losses']}/{v['n_budgets']:<4} {v['verdict']:<13} {share_s}{mark}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------------
# run loop
# ---------------------------------------------------------------------------------------------------------------------


def run_sweep(args, engine=None) -> Path:
    """Run the sweep described by validated ``args`` and return the sweep directory (see the module docstring)."""
    import torch

    out_dir = ensure_outside_repo(args.out_dir, REPO_ROOT)
    sweep_dir = _make_sweep_dir(out_dir)
    use_cvvdp = (not args.no_cvvdp) and cvvdp_available()
    primary, better = PRIMARY_METRIC[args.weight_model]
    width, height = args.resolution
    n_pixels = width * height
    pipeline_allocs = [a for a in args.allocators if a != PLAIN_ALLOCATOR]
    others = [a for a in pipeline_allocs if a != "uniform"]
    with_plain = PLAIN_ALLOCATOR in args.allocators

    config = {
        "args": vars(args),
        "n_cap": n_cap(args.pilot_spp, args.iterations, args.max_per_pixel),
        "git_head": _git_head(),
        "versions": _versions(),
        "cvvdp_available": cvvdp_available(),
        "use_cvvdp": use_cvvdp,
        "primary_metric": primary,
        "better": better,
        "notes": ELIGIBILITY_NOTE,
        "argv": sys.argv,
    }
    _write_json(sweep_dir / "config.json", config)
    decision: Dict[str, Any] = {
        "rule": DECISION_RULE,
        "caveats": list(CAVEATS),
        "eligibility_note": ELIGIBILITY_NOTE,
        "primary_metric": primary,
        "better": better,
        "weight_model": args.weight_model,
    }
    _write_json(sweep_dir / "decision.json", decision)
    print(f"[sweep] {sweep_dir}")
    print(f"[sweep] decision rule (stated before any render):\n{DECISION_RULE}")
    print(f"[sweep] {ELIGIBILITY_NOTE}")

    if engine is None:
        from threedgrut_playground.engine import Engine3DGRUT

        engine = Engine3DGRUT(
            gs_object=str(Path(args.ckpt)),
            mesh_assets_folder=str(REPO_ROOT / "threedgrut_playground" / "assets"),
            default_config="apps/colmap_3dgrt.yaml",
        )
    cams = setup_glossy_scene(engine, args.preset)
    terms = make_weight_terms(args.weight_model)
    csv_log = CsvLog(sweep_dir / "results.csv", CSV_COLUMNS)
    summary: Dict[str, Any] = {"notes": ELIGIBILITY_NOTE, "sweep_dir": str(sweep_dir), "cameras": {}}
    plumbing_errors: List[str] = []
    excluded: set = set()  # (camera, budget_spp) left out of the paired analysis
    t_sweep = time.perf_counter()
    smallest = min(args.budgets)
    ref_rays_limit = args.ref_spp / REF_SPP_FACTOR

    for cam_name in args.cameras:
        t_cam = time.perf_counter()
        camera = cams[cam_name].to_camera(width, height)
        ppd = ppd_from_camera(camera)
        cam_summary: Dict[str, Any] = {"ppd": ppd, "skipped_budgets": [], "budget_failures": [], "warnings": []}
        summary["cameras"][cam_name] = cam_summary

        # 1. reference (dedicated renderer, cached under out_dir/references)
        t0 = time.perf_counter()
        ref_renderer = build_renderer(engine, "uniform", 0, terms, args.max_per_pixel)
        ref = reference_bundle(
            ref_renderer,
            camera,
            args.ref_spp,
            out_dir,
            ckpt_path=str(args.ckpt),
            spp_nonmesh=args.ref_spp_nonmesh,
            pilot_spp=args.pilot_spp,
        )
        shutil.copyfile(ref.cache_dir / "ldr.png", sweep_dir / f"ref_{cam_name}.png")
        cam_summary["reference"] = {
            "key": ref.key,
            "cache_dir": str(ref.cache_dir),
            "load_or_render_s": time.perf_counter() - t0,
            "render_s": ref.meta.get("seconds"),
            "rays": ref.meta.get("rays"),
            "n_mesh_px": int(ref.mesh_mask.sum()),
            "mesh_fraction": float(ref.mesh_mask.float().mean()),
            "ref_noise_floor_mesh": ref.meta.get("ref_noise_floor_mesh"),
        }
        print(f"[sweep] {cam_name}: reference {ref.key[:12]} ({cam_summary['reference']['load_or_render_s']:.1f} s)")

        # 2. viewing model and weights on the reference
        view_dirs = center_view_dirs(camera)
        ecc = eccentricity_deg(view_dirs, image_center_dir(camera))
        w = reference_weights(terms, ref.hdr, ref.mesh_mask, view_dirs, camera, ref_noise_var=ref.display_var_mean)

        # 3. telemetry and discarded warm-ups
        telemetry = CostTelemetry()
        plain_tel = CostTelemetry()
        plain_renderer = trace_only_renderer(engine, telemetry=plain_tel)
        warm_offset = WARMUP_SEED * SEED_STRIDE
        warm = build_renderer(engine, "uniform", WARMUP_SEED, terms, args.max_per_pixel, telemetry=telemetry).render(
            camera,
            smallest * n_pixels,
            args.pilot_spp,
            args.iterations,
            region="mesh",
            sample_offset=warm_offset,
            defensive_fraction=args.defensive_fraction,
        )
        if with_plain:
            plain_uniform_render(plain_renderer, camera, spp=smallest, sample_offset=warm_offset)
        telemetry.reset()
        plain_tel.reset()

        # capacity pre-check (controller amendment 1)
        mesh = warm.mesh_mask.cpu().bool()
        clip_map = warm.maps[-1].get("weight_display_clip") if warm.maps else None
        unclipped = mesh & (clip_map > 0) if clip_map is not None else mesh
        n_unclipped, n_clipped = int(unclipped.sum()), int((mesh & ~unclipped).sum())
        absorb = absorbable_budget(
            n_pixels, n_unclipped, n_clipped, args.pilot_spp, args.iterations, args.max_per_pixel
        )
        budgets, skipped = split_budgets(args.budgets, n_pixels, absorb)
        cam_summary.update(
            capacity_rays=capacity_rays(
                n_pixels, n_unclipped, n_clipped, args.pilot_spp, args.iterations, args.max_per_pixel
            ),
            absorbable_rays=absorb,
            absorbable_spp=absorb / n_pixels,
            capacity_mesh_px_unclipped=n_unclipped,
            capacity_mesh_px_clipped=n_clipped,
            skipped_budgets=skipped,
        )
        del warm
        for s in skipped:
            msg = (
                f"{cam_name}: skipping budget {s['budget_spp']} spp ({s['budget_rays']} rays): above the absorbable "
                f"capacity {absorb} rays ({absorb / n_pixels:.1f} spp; {n_unclipped} unclipped + {n_clipped} clipped "
                f"mesh px, cap {args.max_per_pixel}, {args.iterations} iterations + {TOPUP_ROUNDS} top-up rounds)"
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
            cam_summary["warnings"].append(msg)
        if not budgets:
            cam_summary["warnings"].append(f"{cam_name}: no budget fits the capacity; camera skipped")
            print(f"[sweep] {cam_name}: no budget fits the absorbable capacity; skipping the camera")
            continue

        # 4. budgets x seeds x allocators
        n_ref_warn = 0
        images_dir = sweep_dir / "images"
        for budget in budgets:
            budget_rays = budget * n_pixels
            failed = None
            for seed in args.seeds:
                offset = seed * SEED_STRIDE
                base = {
                    "run_id": sweep_dir.name,
                    "preset": args.preset,
                    "camera": cam_name,
                    "width": width,
                    "height": height,
                    "ppd": ppd,
                    "weight_model": args.weight_model,
                    "seed": seed,
                    "sample_offset": offset,
                    "budget_spp": budget,
                    "budget_rays": budget_rays,
                    "pilot_spp": args.pilot_spp,
                    "iterations": args.iterations,
                    "max_per_pixel": args.max_per_pixel,
                    "defensive_fraction": args.defensive_fraction,
                    "ref_key": ref.key,
                    "ref_spp": args.ref_spp,
                    "ref_spp_nonmesh": args.ref_spp_nonmesh,
                }
                save_images = seed == args.seeds[0] and budget == budgets[-1]
                flags = uniform_mask = None
                try:
                    for alloc in ["uniform", *others]:
                        dumper = (
                            MapDumper(sweep_dir / "dumps" / cam_name / f"b{budget}" / f"{alloc}_s{seed}")
                            if args.dump_maps
                            else None
                        )
                        renderer = build_renderer(
                            engine, alloc, seed, terms, args.max_per_pixel, telemetry=telemetry, dumper=dumper
                        )
                        torch.manual_seed(seed)
                        _sync()
                        t0 = time.perf_counter()
                        res = renderer.render(
                            camera,
                            budget_rays,
                            args.pilot_spp,
                            args.iterations,
                            region="mesh",
                            sample_offset=offset,
                            defensive_fraction=args.defensive_fraction,
                        )
                        _sync()
                        wall_ms = (time.perf_counter() - t0) * 1000.0
                        if alloc == "uniform":
                            flags = ref_noise_flags(ref.var_mean, res.stats.variance_of_mean())
                            uniform_mask = res.mesh_mask
                        metrics = evaluate_image(
                            res.mean_hdr,
                            ref.hdr,
                            ref.var_mean,
                            ref.hi_mask,
                            res.mesh_mask,
                            flags,
                            ecc,
                            w,
                            ppd,
                            use_cvvdp=use_cvvdp,
                        )
                        cls, gamma = parse_allocator(alloc)
                        row = assemble_row(
                            {
                                **base,
                                "allocator": alloc,
                                "allocator_class": cls,
                                "gamma": "" if gamma is None else gamma,
                            },
                            res,
                            budget_rays,
                            args.budget_tol,
                            wall_ms,
                            metrics,
                        )
                        csv_log.write(row)
                        if row["max_px_count"] > ref_rays_limit:
                            n_ref_warn += 1
                            warnings.warn(
                                f"{cam_name} b{budget} s{seed} {alloc}: max_px_count {row['max_px_count']} > ref_spp/8 "
                                f"= {ref_rays_limit:g}",
                                UserWarning,
                                stacklevel=2,
                            )
                        if save_images:
                            save_png(
                                to_display_ldr(res.mean_hdr), images_dir / f"{cam_name}_b{budget}_s{seed}_{alloc}.png"
                            )
                            _save_counts_png(
                                res.counts, images_dir / f"{cam_name}_b{budget}_s{seed}_{alloc}_counts.png"
                            )
                        del res, renderer

                    if with_plain:
                        plain = plain_uniform_render(plain_renderer, camera, spp=budget, sample_offset=offset)
                        ns = SimpleNamespace(
                            counts=torch.full((height, width), budget, dtype=torch.int64, device=plain.mean_hdr.device),
                            mesh_mask=uniform_mask,
                            timings={
                                "rays_launched": plain.rays_launched,
                                "n_nonfinite": plain.n_nonfinite,
                                "rays_unspent": 0,
                                "gpu_ms": plain.gpu_ms,
                            },
                        )
                        metrics = evaluate_image(
                            plain.mean_hdr,
                            ref.hdr,
                            ref.var_mean,
                            ref.hi_mask,
                            uniform_mask,
                            flags,
                            ecc,
                            w,
                            ppd,
                            use_cvvdp=use_cvvdp,
                        )
                        row = assemble_row(
                            {**base, "allocator": PLAIN_ALLOCATOR, "allocator_class": PLAIN_ALLOCATOR, "gamma": ""},
                            ns,
                            budget_rays,
                            args.budget_tol,
                            plain.wall_ms,
                            metrics,
                        )
                        csv_log.write(row)
                        if save_images:
                            save_png(
                                to_display_ldr(plain.mean_hdr),
                                images_dir / f"{cam_name}_b{budget}_s{seed}_plain_uniform.png",
                            )
                        del plain
                except BudgetMismatchError as exc:
                    # Safety net behind the capacity pre-check: record, drop this budget from the paired analysis and
                    # keep sweeping the other budgets.
                    failed = f"{cam_name} b{budget} s{seed}: {exc}"
                    break
            if failed is not None:
                excluded.add((cam_name, budget))
                cam_summary["budget_failures"].append({"budget_spp": budget, "error": failed})
                warnings.warn(f"budget failure (excluded from the analysis): {failed}", UserWarning, stacklevel=2)
            print(
                f"[sweep] {cam_name}: budget {budget} done "
                f"({'FAILED' if failed else 'ok'}; {time.perf_counter() - t_cam:.0f} s into the camera)"
            )

        # 5. per-camera summary: cost fits, 1/n convergence of the uniform rows, ppd
        cam_summary["cost_fits"] = telemetry_cost_fits(telemetry)
        cam_summary["plain_cost_fits"] = telemetry_cost_fits(plain_tel)
        cam_summary["max_px_count_above_ref_spp_over_8"] = n_ref_warn
        uni = [
            r
            for r in _read_rows(sweep_dir / "results.csv")
            if r["camera"] == cam_name
            and r["allocator"] == "uniform"
            and (cam_name, int(r["budget_spp"])) not in excluded
        ]
        ns_, mses = [], []
        for r in uni:
            frac = float(r["mesh_fraction"])
            if frac > 0:
                ns_.append(float(r["rays_mesh"]) / (frac * n_pixels))
                mses.append(float(r["mse_lum_mesh"]))
        floor = float(uni[0]["ref_noise_floor_mesh"]) if uni else math.nan
        slope = convergence_slope(ns_, mses, floor) if uni else math.nan
        cam_summary["convergence"] = {"slope": slope, "n": ns_, "mse": mses, "floor": floor}
        if len({round(n, 6) for n in ns_}) < 2:
            cam_summary["convergence"]["note"] = "fewer than two distinct budgets: slope not checked"
        else:
            try:
                for msg in check_plumbing({"slope": slope}):
                    cam_summary["warnings"].append(msg)
                    warnings.warn(f"{cam_name}: {msg}", UserWarning, stacklevel=2)
            except PlumbingError as exc:
                cam_summary["plumbing_error"] = str(exc)
                plumbing_errors.append(f"{cam_name}: {exc}")
        cam_summary["seconds"] = time.perf_counter() - t_cam
        _write_json(sweep_dir / "summary.json", summary)
        print(f"[sweep] {cam_name}: done in {cam_summary['seconds']:.0f} s; convergence slope {slope:.3f}")

    # 6. paired differences, decision, plots
    csv_rows = [r for r in _read_rows(sweep_dir / "results.csv") if (r["camera"], int(r["budget_spp"])) not in excluded]
    pairings = default_pairings(args.allocators)
    metrics_paired = [primary, "psnr_mesh"]
    if args.weight_model in ("cielab", "cielab_nomask") and use_cvvdp:
        metrics_paired.append("cvvdp_mesh")
    elif args.weight_model == "tvi":
        metrics_paired.append("flip_mesh")
    metrics_paired = list(dict.fromkeys(metrics_paired))
    paired_log = CsvLog(sweep_dir / "paired.csv", PAIRED_COLUMNS)
    paired_all: List[Dict[str, Any]] = []
    for metric in metrics_paired:
        rows = paired_differences(csv_rows, metric, pairings)
        for p in rows:
            paired_log.write(p)
        paired_all += rows
    primary_rows = [p for p in paired_all if p["metric"] == primary]
    verdicts = decide(primary_rows, primary, better, rows=csv_rows)
    decision["verdicts"] = verdicts
    decision["topup_share_warn"] = TOPUP_SHARE_WARN
    decision["excluded_budgets"] = sorted([list(e) for e in excluded])
    _write_json(sweep_dir / "decision.json", decision)
    plots = []
    if csv_rows:
        for cam_name in dict.fromkeys(r["camera"] for r in csv_rows):
            plots += plot_curves(csv_rows, paired_all, sweep_dir / "plots", cam_name, primary)
    summary["plots"] = [str(p) for p in plots]
    summary["seconds"] = time.perf_counter() - t_sweep
    summary["n_rows"] = len(csv_rows)
    summary["plumbing_errors"] = plumbing_errors
    _write_json(sweep_dir / "summary.json", summary)

    print(f"[sweep] done in {summary['seconds']:.0f} s: {sweep_dir}")
    print(f"[sweep] verdicts on {primary} ({better} is better):")
    print(_verdict_table(verdicts, primary_rows, better))
    print(f"[sweep] note: {ELIGIBILITY_NOTE}")
    for c in CAVEATS:
        print(f"[sweep] caveat: {c}")
    if plumbing_errors:
        raise PlumbingError("; ".join(plumbing_errors))
    return sweep_dir


def main(argv: Optional[Sequence[str]] = None, engine=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    try:
        run_sweep(args, engine=engine)
    except PlumbingError as exc:
        print(f"[sweep] plumbing check failed (outputs written): {exc}", file=sys.stderr)
        return 1
    return 0
