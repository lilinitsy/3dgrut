# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Perceptual oracle: does choosing cone vs MC per region, judged by CVVDP, beat choosing per material?

Reads the option cache that ``oracle_mix.py`` writes (``<out>/options.pt``: every option's two realizations A and B, the
per-pixel cost, the reference, the object map). No rendering.

Per-material rule (the baseline a reviewer would propose): one option per object (floor, each sphere, everything else).
Its oracle version picks each object's best option, so it is the best any per-material rule can do.
Finer rules pick per 32x32, 16x16 and 8x8 tile, and per pixel (error box-filtered 5x5).

Each rule picks on realization A, minimizing error + lambda * cost, and is scored on realization B. Error for picking is
the CVVDP per-pixel heatmap (1 - JOD/10); for contrast, the per-pixel rule also picks by FLIP. Scores: whole-image CVVDP
JOD (higher is better, 10 = identical to the reference) and mean FLIP on the glossy objects (lower is better).

Usage: python scripts/cone/oracle_perceptual.py --out C:/Users/ville/3dgrut_runs/oracle_mix
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from threedgrut_playground.adaptive.evaluation.ldr import save_png
from threedgrut_playground.adaptive.evaluation.metrics import cvvdp_jod, flip_error_map
from threedgrut_playground.adaptive.evaluation.reflection_rate import cvvdp_heatmap

BUDGET_SPP = (1, 2, 4, 8, 16, 32)
LAMBDAS = torch.logspace(-5, 3, 72).tolist()
SMOOTH = 5
FIGURE_BUDGET = 8


def log(text):
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def box(x, k):
    return F.avg_pool2d(x[None, None], k, stride=1, padding=k // 2, count_include_pad=False)[0, 0]


def tile_mean(x, k):
    return F.avg_pool2d(x[None, None], k)[0, 0].repeat_interleave(k, 0).repeat_interleave(k, 1)


def region_mean(x, ids):
    out = torch.zeros_like(x)
    for i in ids.unique().tolist():
        m = ids == i
        out[m] = x[m].mean()
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path("C:/Users/ville/3dgrut_runs/oracle_mix"))
    args = p.parse_args(argv)
    dev = torch.device("cuda")
    c = torch.load(args.out / "options.pt", map_location=dev)
    names, cost, A, B, ref = c["options"], c["cost"], c["A"], c["B"], c["reference"]
    ids, ppd, mc_ms = c["object_ids"], c["ppd"], c["mc_ms"]
    glossy = ids >= 0
    K, H, W = cost.shape
    out = args.out / "perceptual"
    out.mkdir(exist_ok=True)

    log("error maps on realization A")
    err = {
        "cvvdp": torch.stack([cvvdp_heatmap(A[k], ref, ppd) for k in range(K)]),
        "flip": torch.stack([flip_error_map(A[k], ref, ppd) for k in range(K)]),
    }

    def per_map(e, fn):
        return torch.stack([fn(e[k]) for k in range(K)])

    rules = {  # name -> (picking metric, error maps aggregated over the rule's regions)
        "per material": ("cvvdp", per_map(err["cvvdp"], lambda x: region_mean(x, ids))),
        "tile 32": ("cvvdp", per_map(err["cvvdp"], lambda x: tile_mean(x, 32))),
        "tile 16": ("cvvdp", per_map(err["cvvdp"], lambda x: tile_mean(x, 16))),
        "tile 8": ("cvvdp", per_map(err["cvvdp"], lambda x: tile_mean(x, 8))),
        "pixel": ("cvvdp", per_map(err["cvvdp"], lambda x: box(x, SMOOTH))),
        "pixel, picked by FLIP": ("flip", per_map(err["flip"], lambda x: box(x, SMOOTH))),
    }

    def compose(choice):
        img = torch.gather(B, 0, choice[None, :, :, None].expand(1, H, W, 3))[0]
        return img, torch.gather(cost, 0, choice[None])[0].sum().item()

    def score(img):
        return {"jod": cvvdp_jod(img, ref, ppd), "glossy_flip": flip_error_map(img, ref, ppd)[glossy].mean().item()}

    uniform = []
    for k in range(K):
        uniform.append({"name": names[k], "ms": cost[k].sum().item(), **score(B[k])})
        log(f"uniform {names[k]:<14} {uniform[-1]['ms']:8.1f} ms  JOD {uniform[-1]['jod']:.2f}")

    curves = {}
    # A per-material rule cannot know which of an object's pixels are cheap to trace: it sees the object's mean cost.
    # (Tiles are 32x32 or finer, inside the 32x32 cost tiles, so the finer rules see exactly the cost they would get.)
    material_cost = torch.stack([region_mean(cost[k], ids) for k in range(K)])
    rule_cost = {rule: (material_cost if rule == "per material" else cost) for rule in rules}
    for rule, (metric, e) in rules.items():
        pts, seen = [], {}
        for lam in LAMBDAS:
            choice = torch.argmin(e + lam * rule_cost[rule] * (H * W) / 1000.0, dim=0)
            key = hash(choice.cpu().numpy().tobytes())
            if key in seen:
                continue
            img, ms = compose(choice)
            seen[key] = True
            pts.append({"lambda": lam, "ms": ms, **score(img), "choice": choice})
        pts.sort(key=lambda q: q["ms"])
        curves[rule] = pts
        log(f"{rule}: {len(pts)} distinct allocations, {pts[0]['ms']:.0f}..{pts[-1]['ms']:.0f} ms")

    def at_budget(pts, budget):
        """The allocation closest in cost to the budget without going over it (2% slack)."""
        fit = [q for q in pts if q["ms"] <= budget * 1.02]
        return max(fit, key=lambda q: q["ms"]) if fit else None

    headline = []
    for n in BUDGET_SPP:
        budget = n * mc_ms
        ok = [u for u in uniform if u["ms"] <= budget * 1.02]
        best = max(ok, key=lambda u: u["jod"]) if ok else None
        row = {"budget_ms": budget, "budget_as_mc_spp": n, "best_uniform": best}
        line = f"budget MC {n:>2} spp ({budget:6.1f} ms): uniform {best['name'] if best else '-':<12} JOD "
        line += f"{best['jod']:.2f}" if best else "  - "
        for rule, pts in curves.items():
            q = at_budget(pts, budget)
            row[rule] = None if q is None else {k: v for k, v in q.items() if k != "choice"}
            line += f" | {rule} {'-' if q is None else round(q['jod'], 2)}"
        headline.append(row)
        log(line)

    # Figure: JOD against frame time
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = {
        "per material": "#229954",
        "tile 32": "#7d3c98",
        "tile 16": "#2471a3",
        "tile 8": "#17a589",
        "pixel": "#c0392b",
        "pixel, picked by FLIP": "#e67e22",
    }
    for ax, key, label in (
        (axes[0], "jod", "CVVDP JOD, whole image (higher is better)"),
        (axes[1], "glossy_flip", "mean FLIP on glossy objects (lower is better)"),
    ):
        mc = [u for u in uniform if u["name"].startswith("MC")]
        cones = [u for u in uniform if u["name"].startswith("cone")]
        ax.plot([u["ms"] for u in mc], [u[key] for u in mc], "o-", color="#555", label="uniform MC")
        ax.plot([u["ms"] for u in cones], [u[key] for u in cones], "s--", color="#999", label="uniform cone (caps)")
        for rule, pts in curves.items():
            ax.plot(
                [q["ms"] for q in pts],
                [q[key] for q in pts],
                "-",
                color=colors[rule],
                label=f"oracle {rule}",
                ls=":" if "FLIP" in rule else "-",
            )
        ax.set_xscale("log")
        ax.set_xlabel("frame time (ms, sum of per-pixel costs)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)
    fig.suptitle("Room ladder 512 px: oracle allocations picked by CVVDP, at increasing granularity")
    fig.tight_layout()
    fig.savefig(out / "jod_curves.png", dpi=130)

    # Images at one budget: the per-material rule against the finer rules
    budget = FIGURE_BUDGET * mc_ms
    n_mc = sum(1 for nm in names if nm.startswith("MC"))
    n_cone = K - n_mc
    palette = torch.tensor(
        [[0.15 + 0.85 * i / max(n_mc - 1, 1)] * 3 for i in range(n_mc)]
        + [[1.0, 0.85 - 0.6 * j / max(n_cone - 1, 1), 0.1] for j in range(n_cone)],
        device=dev,
    )
    for rule in ("per material", "tile 16", "pixel", "pixel, picked by FLIP"):
        q = at_budget(curves[rule], budget)
        if q is None:
            continue
        tag = rule.replace(", picked by ", "_").replace(" ", "_")
        save_png(compose(q["choice"])[0], out / f"image_{tag}_mc{FIGURE_BUDGET}budget.png")
        save_png(palette[q["choice"]], out / f"choice_{tag}_mc{FIGURE_BUDGET}budget.png")
    save_png(ref, out / "reference.png")

    result = {
        "uniform": uniform,
        "headline": headline,
        "curves": {r: [{k: v for k, v in q.items() if k != "choice"} for q in pts] for r, pts in curves.items()},
    }
    (out / "results.json").write_text(json.dumps(result, indent=2, default=float))
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
