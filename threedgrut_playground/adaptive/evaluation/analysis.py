# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Results CSV, budget check, paired 95% intervals, the pre-stated decision rule and plots (D3, D9, D14).

Everything here is CPU-only and works on plain row dicts (one per render), so it can be tested without an engine and
re-run on a finished ``results.csv``.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .metrics import METRIC_KEYS

NAN = float("nan")

# ---------------------------------------------------------------------------------------------------------------------
# columns
# ---------------------------------------------------------------------------------------------------------------------

IDENTITY_COLUMNS = (
    "run_id",
    "preset",
    "camera",
    "width",
    "height",
    "ppd",
    "weight_model",
    "allocator",
    "allocator_class",
    "gamma",
    "seed",
    "sample_offset",
    "budget_spp",
    "budget_rays",
    "pilot_spp",
    "iterations",
    "max_per_pixel",
    "defensive_fraction",
)
RAY_COLUMNS = (
    "rays_launched",
    "rays_mesh",
    "rays_nonmesh",
    "rays_unspent",
    "topup_rays",
    "topup_share",
    "n_nonfinite",
    "budget_rel_err",
    "mesh_fraction",
    "max_px_count",
)
TIME_COLUMNS = ("wall_ms", "gpu_ms", "n_launches")
REF_COLUMNS = ("ref_key", "ref_spp", "ref_spp_nonmesh")

CSV_COLUMNS = IDENTITY_COLUMNS + RAY_COLUMNS + TIME_COLUMNS + tuple(METRIC_KEYS) + REF_COLUMNS

PAIRED_COLUMNS = (
    "camera",
    "weight_model",
    "budget_spp",
    "allocator",
    "baseline",
    "metric",
    "n_seeds",
    "mean_diff",
    "std_diff",
    "ci95_low",
    "ci95_high",
    "mean_rays_launched",
    "mean_wall_ms",
)

DECISION_MIN_FRACTION = 0.6
TOPUP_SHARE_WARN = 0.10
PLAIN_ALLOCATOR = "plain_uniform"

DECISION_RULE = (
    "For weight model W with primary metric M (flip_mesh for cielab and cielab_nomask, lower is better; cvvdp_mesh for "
    "tvi, higher is "
    "better), allocator X beats baseline Y if the 95% t-interval of the per-seed paired difference X - Y in M lies "
    "entirely on the better side of 0 at >= 60% of the swept budgets (>= 3 of the default 5) and entirely on the worse "
    "side at none. The key comparison is perceptual_g<gamma> vs variance_g<gamma> for gamma in {1, 0.5}; uniform "
    "comparisons are secondary. The intervals cover seed-to-seed sampling noise only: every seed is scored against the "
    "same reference, so the reference's own error is common to all seeds; it largely cancels in the paired differences "
    "(identical additive term in expected squared error) but is not part of the CI. ref_noise_share_mesh and the "
    "_clean metrics bound that effect."
)

CAVEATS = (
    "Seeds are disjoint segments of one Owen-scrambled sequence per pixel (D2), so jitter error is weakly dependent "
    "across seeds while the confidence intervals assume independence.",
    "The default cielab weight (D9) failed its FLIP validation (the core's test_core_flip_validation is a strict "
    "xfail: FLIP's implied weight is flatter than L*), so a cielab-weighted run is not tuned to the primary metric.",
    "Clipped pixels (weight_display_clip == 0) are ineligible for every allocator and receive only pilot rays "
    "(see ELIGIBILITY_NOTE of the sweep).",
    "Rows with topup_share > 10% mix a uniform share into the adaptive allocator (the per-iteration cap pushes trimmed "
    "rays into the final uniform top-up), which dilutes adaptive allocators at high budgets.",
)


# ---------------------------------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------------------------------


def _f(value: Any) -> float:
    """Float of a CSV/row value; ``nan`` for missing or unparsable entries."""
    if value is None:
        return NAN
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return NAN


def _finite(x: float) -> bool:
    return math.isfinite(x)


def t95(n: int) -> float:
    """Two-sided 95% Student-t quantile for ``n`` samples (``n - 1`` degrees of freedom); ``nan`` for ``n < 2``."""
    if n < 2:
        return NAN
    from scipy import stats

    return float(stats.t.ppf(0.975, n - 1))


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return NAN, NAN
    mean = math.fsum(values) / n
    if n < 2:
        return mean, NAN
    return mean, math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (n - 1))


def _ci(values: Sequence[float]) -> Tuple[float, float, float, float]:
    """``(mean, std, ci95_low, ci95_high)``; the interval is ``nan`` for fewer than two values."""
    mean, std = _mean_std(values)
    n = len(values)
    if n < 2:
        return mean, std, NAN, NAN
    half = t95(n) * std / math.sqrt(n)
    return mean, std, mean - half, mean + half


def _nanmean(values: Sequence[float]) -> float:
    vals = [v for v in values if _finite(v)]
    return math.fsum(vals) / len(vals) if vals else NAN


# ---------------------------------------------------------------------------------------------------------------------
# budget check and CSV
# ---------------------------------------------------------------------------------------------------------------------


class BudgetMismatchError(RuntimeError):
    """Rays launched deviate from the ray budget by more than the tolerance."""


def check_budget(rays_launched: float, budget_rays: float, tol: float = 0.01) -> float:
    """Signed relative deviation ``(launched - budget) / budget``; raises if its magnitude exceeds ``tol``."""
    if not float(budget_rays) > 0:
        raise BudgetMismatchError(f"budget must be positive, got {budget_rays}")
    rel = (float(rays_launched) - float(budget_rays)) / float(budget_rays)
    if not abs(rel) <= tol:  # also raises on nan

        hint = (
            "the per-pixel cap (--max-per-pixel) left rays unspent; raise it or lower the budget"
            if rel < 0
            else "more rays were launched than budgeted"
        )
        raise BudgetMismatchError(
            f"rays launched {float(rays_launched):.0f} vs budget {float(budget_rays):.0f} ({rel:+.2%}, tolerance {tol:.2%}): {hint}"
        )
    return rel


class CsvLog:
    """Append-only CSV with a fixed header (written once); every write is flushed."""

    def __init__(self, path, columns: Sequence[str] = CSV_COLUMNS):
        self.path = Path(path)
        self.columns = tuple(columns)

    def _has_header(self) -> bool:
        """True if the file exists with content and its header equals the columns (else ``ValueError``)."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return False
        with open(self.path, newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh), [])
        if tuple(header) != self.columns:
            raise ValueError(f"{self.path} has a different header than the requested columns")
        return True

    def write(self, row: Mapping[str, Any]) -> None:
        if set(row) != set(self.columns):
            missing = sorted(set(self.columns) - set(row))
            extra = sorted(set(row) - set(self.columns))
            raise ValueError(f"row keys do not match the columns (missing {missing[:5]}, extra {extra[:5]})")
        has_header = self._has_header()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, lineterminator="\n")
            if not has_header:
                writer.writerow(self.columns)
            writer.writerow([row[c].item() if hasattr(row[c], "item") else row[c] for c in self.columns])
            fh.flush()


def assemble_row(
    base: Mapping[str, Any],
    result: Any,
    budget_rays: float,
    budget_tol: float,
    wall_ms: float,
    metrics: Mapping[str, float],
) -> Dict[str, Any]:
    """One results row from identity/reference fields (``base``), a render ``result`` and the metric values.

    ``result`` is an ``AdaptiveResult`` or a namespace with ``counts`` ``[H, W]``, ``mesh_mask`` ``[H, W]`` and
    ``timings``. Raises :class:`BudgetMismatchError` if the rays launched miss ``budget_rays`` by more than the
    tolerance. ``mesh_fraction`` is the fraction of pixels in the visible-mesh mask.
    """
    timings = result.timings
    counts, mask = result.counts, result.mesh_mask.bool()
    rays_launched = float(timings["rays_launched"])
    rel_err = check_budget(rays_launched, budget_rays, budget_tol)
    rays_total = int(counts.sum().item())
    rays_mesh = int(counts[mask].sum().item())
    topup = _f(timings.get("topup_rays", NAN))
    row: Dict[str, Any] = {k: base[k] for k in IDENTITY_COLUMNS + REF_COLUMNS}
    row.update(
        rays_launched=rays_launched,
        rays_mesh=rays_mesh,
        rays_nonmesh=rays_total - rays_mesh,
        rays_unspent=_f(timings.get("rays_unspent", NAN)),
        topup_rays=topup,
        topup_share=topup / rays_launched if _finite(topup) and rays_launched > 0 else NAN,
        n_nonfinite=_f(timings.get("n_nonfinite", NAN)),
        budget_rel_err=rel_err,
        mesh_fraction=float(mask.float().mean().item()),
        max_px_count=int(counts.max().item()),
        wall_ms=float(wall_ms),
        gpu_ms=_f(timings.get("gpu_ms", NAN)),
        n_launches=_f(timings.get("n_launches", NAN)),
    )
    row.update({k: metrics[k] for k in METRIC_KEYS})
    return row


# ---------------------------------------------------------------------------------------------------------------------
# aggregation and paired statistics
# ---------------------------------------------------------------------------------------------------------------------


def aggregate(
    rows: Iterable[Mapping[str, Any]], x_key: str, y_key: str
) -> Dict[str, List[Tuple[float, float, float, float]]]:
    """Per allocator and ``budget_spp``: ``(mean x, mean y, ci95_low, ci95_high)`` over seeds, ascending in budget.

    Rows with a non-finite ``x`` or ``y`` are dropped; a single seed gives the degenerate interval ``(y, y)``.
    """
    groups: Dict[str, Dict[float, List[Tuple[float, float]]]] = {}
    for r in rows:
        x, y, b = _f(r.get(x_key)), _f(r.get(y_key)), _f(r.get("budget_spp"))
        if _finite(x) and _finite(y):
            groups.setdefault(str(r["allocator"]), defaultdict(list))[b].append((x, y))
    out: Dict[str, List[Tuple[float, float, float, float]]] = {}
    for alloc, by_budget in groups.items():
        series = []
        for b in sorted(by_budget):
            xs, ys = zip(*by_budget[b])
            mean_y, _, lo, hi = _ci(ys)
            if len(ys) < 2:
                lo = hi = mean_y
            series.append((math.fsum(xs) / len(xs), mean_y, lo, hi))
        out[alloc] = series
    return out


def default_pairings(allocators: Sequence[str]) -> List[Tuple[str, str]]:
    """Every non-uniform pipeline allocator vs ``uniform`` plus ``perceptual_g<g>`` vs ``variance_g<g>``.

    ``plain_uniform`` is never paired.
    """
    present = [a for a in allocators if a != PLAIN_ALLOCATOR]
    pairs: List[Tuple[str, str]] = []
    if "uniform" in present:
        pairs += [(a, "uniform") for a in present if a != "uniform"]
    for a in present:
        if a.startswith("perceptual_g"):
            twin = "variance_g" + a[len("perceptual_g") :]
            if twin in present:
                pairs.append((a, twin))
    return pairs


def paired_differences(
    rows: Iterable[Mapping[str, Any]], metric: str, pairings: Sequence[Tuple[str, str]]
) -> List[Dict[str, Any]]:
    """Per (camera, weight model, budget, pair): per-seed ``metric(a) - metric(b)`` with a 95% t-interval.

    Seeds are matched by value (same seed gives the same ``sample_offset``); non-finite differences are skipped. A
    pair with matched seeds but no usable difference still yields a row (``n_seeds`` 0, ``nan`` statistics) so
    ``decide`` counts the budget.
    """
    index: Dict[Tuple, Mapping[str, Any]] = {}
    for r in rows:
        key = (str(r["camera"]), str(r["weight_model"]), _f(r["budget_spp"]), str(r["allocator"]), _f(r["seed"]))
        index[key] = r
    cells = sorted({k[:3] for k in index})
    out: List[Dict[str, Any]] = []
    for camera, wm, budget in cells:
        for a, b in pairings:
            seeds_a = {k[4] for k in index if k[:4] == (camera, wm, budget, a)}
            seeds_b = {k[4] for k in index if k[:4] == (camera, wm, budget, b)}
            seeds = sorted(seeds_a & seeds_b)
            if not seeds:
                continue
            diffs, rays, wall = [], [], []
            for s in seeds:
                ra, rb = index[(camera, wm, budget, a, s)], index[(camera, wm, budget, b, s)]
                d = _f(ra.get(metric)) - _f(rb.get(metric))
                if _finite(d):
                    diffs.append(d)
                    rays.append(_f(ra.get("rays_launched")))
                    wall.append(_f(ra.get("wall_ms")))
            mean, std, lo, hi = _ci(diffs)
            out.append(
                {
                    "camera": camera,
                    "weight_model": wm,
                    "budget_spp": budget,
                    "allocator": a,
                    "baseline": b,
                    "metric": metric,
                    "n_seeds": len(diffs),
                    "mean_diff": mean,
                    "std_diff": std,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "mean_rays_launched": _nanmean(rays),
                    "mean_wall_ms": _nanmean(wall),
                }
            )
    return out


def decide(
    paired_rows: Sequence[Mapping[str, Any]],
    metric: str,
    better: str,
    rows: Optional[Iterable[Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Apply the pre-stated rule per (camera, weight model, allocator, baseline).

    ``wins`` counts budgets whose interval lies entirely on the better side of 0 (``better`` is ``"lower"`` or
    ``"higher"``), ``losses`` those entirely on the worse side. Verdict ``"beats"`` needs
    ``wins >= ceil(DECISION_MIN_FRACTION * n_budgets)`` and no loss, ``"loses"`` is the mirror image, anything else
    (including non-finite intervals) is ``"inconclusive"``.

    ``max_topup_share`` is the maximum ``topup_share`` over the pair's rows: from ``max_topup_share`` keys of the
    paired rows if present, plus the raw ``rows`` (both allocators of the pair, same camera and weight model) if
    given; ``nan`` if unavailable. ``topup_caveat`` is True when it exceeds :data:`TOPUP_SHARE_WARN`; the verdict
    itself is unchanged.
    """
    if better not in ("lower", "higher"):
        raise ValueError(f"better must be 'lower' or 'higher', got {better!r}")
    raw = list(rows) if rows is not None else []
    groups: Dict[Tuple, List[Mapping[str, Any]]] = {}
    for p in paired_rows:
        if p.get("metric", metric) != metric:
            continue
        groups.setdefault((p["camera"], p["weight_model"], p["allocator"], p["baseline"]), []).append(p)
    out: List[Dict[str, Any]] = []
    for (camera, wm, alloc, base), ps in groups.items():
        wins = losses = 0
        for p in ps:
            lo, hi = _f(p["ci95_low"]), _f(p["ci95_high"])
            if not (_finite(lo) and _finite(hi)):
                continue
            below, above = hi < 0, lo > 0
            if (below and better == "lower") or (above and better == "higher"):
                wins += 1
            elif below or above:
                losses += 1
        n = len(ps)
        need = max(1, math.ceil(round(DECISION_MIN_FRACTION * n, 9)))
        if wins >= need and losses == 0:
            verdict = "beats"
        elif losses >= need and wins == 0:
            verdict = "loses"
        else:
            verdict = "inconclusive"
        shares = [_f(p.get("max_topup_share")) for p in ps]
        shares += [
            _f(r.get("topup_share"))
            for r in raw
            if str(r.get("camera")) == str(camera)
            and str(r.get("weight_model")) == str(wm)
            and str(r.get("allocator")) in (alloc, base)
        ]
        shares = [s for s in shares if _finite(s)]
        max_share = max(shares) if shares else NAN
        out.append(
            {
                "camera": camera,
                "weight_model": wm,
                "allocator": alloc,
                "baseline": base,
                "metric": metric,
                "better": better,
                "n_budgets": n,
                "wins": wins,
                "losses": losses,
                "verdict": verdict,
                "max_topup_share": max_share,
                "topup_caveat": bool(_finite(max_share) and max_share > TOPUP_SHARE_WARN),
            }
        )
    return out


def convergence_slope(n: Sequence[float], mse: Sequence[float], floor: float) -> float:
    """Least-squares slope of ``log(mse - floor)`` vs ``log n`` over the points with ``mse > floor``; ``nan`` if < 2."""
    import numpy as np

    pts = [
        (math.log(float(x)), math.log(float(y) - floor)) for x, y in zip(n, mse) if float(y) > floor and float(x) > 0
    ]
    if len(pts) < 2:
        return NAN
    xs, ys = np.array(pts, dtype=np.float64).T
    if np.ptp(xs) == 0.0:
        return NAN
    return float(np.polyfit(xs, ys, 1)[0])


# ---------------------------------------------------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------------------------------------------------


def _label_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Copies of ``rows`` whose ``allocator`` is suffixed with the weight model when several are present."""
    multi = len({str(r.get("weight_model")) for r in rows}) > 1
    out = []
    for r in rows:
        r = dict(r)
        if multi:
            r["allocator"] = f"{r['allocator']} [{r['weight_model']}]"
        out.append(r)
    return out


def _err(y: Sequence[float], lo: Sequence[float], hi: Sequence[float]) -> List[List[float]]:
    """Error-bar half-lengths; ``nan`` bounds draw no bar."""
    return [
        [a - b if _finite(b) else 0.0 for a, b in zip(y, lo)],
        [b - a if _finite(b) else 0.0 for a, b in zip(y, hi)],
    ]


def plot_curves(
    rows: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
    out_dir,
    camera: str,
    primary_metric: str,
) -> List[Path]:
    """Write the per-camera PNGs into ``out_dir`` and return their paths.

    ``<camera>_<metric>_vs_rays.png`` (pipeline allocators only, log2 x) and ``..._vs_wall_ms.png`` (all allocators,
    log x) for the primary metric and ``psnr_mesh``, mean with 95% CI bars; ``<camera>_paired_<primary>_vs_budget.png``
    with one line per pairing and a zero line. The sanity objective is not plotted.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cam_rows = [r for r in rows if str(r.get("camera")) == str(camera)]
    paths: List[Path] = []

    def save(fig, name: str) -> None:
        path = out_dir / name
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    for metric in dict.fromkeys((primary_metric, "psnr_mesh")):
        for x_key, suffix, log2, pipeline_only in (
            ("rays_launched", "vs_rays", True, True),
            ("wall_ms", "vs_wall_ms", False, False),
        ):
            sel = [r for r in cam_rows if not (pipeline_only and r.get("allocator") == PLAIN_ALLOCATOR)]
            fig, ax = plt.subplots(figsize=(6.4, 4.4))
            for alloc, series in aggregate(_label_rows(sel), x_key, metric).items():
                x, y, lo, hi = (list(c) for c in zip(*series))
                ax.errorbar(x, y, yerr=_err(y, lo, hi), marker="o", capsize=3, label=alloc)
            if log2:
                ax.set_xscale("log", base=2)
            else:
                ax.set_xscale("log")
            ax.set_xlabel(x_key)
            ax.set_ylabel(metric)
            ax.set_title(f"{camera}: {metric} vs {x_key}")
            ax.grid(True, alpha=0.3)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
            save(fig, f"{camera}_{metric}_{suffix}.png")

    cam_pairs = [
        p
        for p in paired_rows
        if str(p.get("camera")) == str(camera) and p.get("metric", primary_metric) == primary_metric
    ]
    multi = len({str(p["weight_model"]) for p in cam_pairs}) > 1
    lines: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for p in cam_pairs:
        lines[f"{p['allocator']} - {p['baseline']}" + (f" [{p['weight_model']}]" if multi else "")].append(p)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for name, ps in lines.items():
        ps = sorted(ps, key=lambda p: _f(p["budget_spp"]))
        y = [_f(p["mean_diff"]) for p in ps]
        lo, hi = [_f(p["ci95_low"]) for p in ps], [_f(p["ci95_high"]) for p in ps]
        ax.errorbar([_f(p["budget_spp"]) for p in ps], y, yerr=_err(y, lo, hi), marker="o", capsize=3, label=name)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("budget_spp")
    ax.set_ylabel(f"paired difference in {primary_metric}")
    ax.set_title(f"{camera}: paired {primary_metric} (mean, 95% CI)")
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)
    save(fig, f"{camera}_paired_{primary_metric}_vs_budget.png")
    return paths
