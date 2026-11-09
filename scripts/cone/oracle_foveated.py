# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Foveated oracle: with a gaze point, does a lookup on what is known at the hit (material, eccentricity) match a
per-pixel oracle, and does knowing the gaze beat the gaze-blind per-material rule?

Reads ``<out>/options.pt`` from ``oracle_mix.py`` (two realizations A, B of every option, per-pixel cost, reference,
object map). No rendering. Metric: FovVideoVDP (foveated, gaze = fixation point), display geometry matching the camera
(512 px over a 60 degree horizontal FOV). Picking uses its per-pixel heatmap on realization A; scores are its JOD on
realization B (higher is better, 10 = identical).

Rules, each swept over lambda in  error + lambda * cost:

- ``material, gaze-blind``: one option per object picked by the non-foveated CVVDP heatmap. Today's kind of rule.
- ``material, per gaze``: one option per object picked by the foveated heatmap of this gaze (an oracle).
- ``table, held-out gaze``: one option per (object, eccentricity band), fitted on the OTHER gazes and applied to this
  one. The practical lookup: everything it uses is known at the hit before the reflection is traced.
- ``table, per gaze``: the same table fitted on this gaze (its upper bound).
- ``tile 16`` and ``pixel``: per-region oracles (upper bounds for anything).

Objects have one roughness each here, so "object" stands for "roughness".

Usage: python scripts/cone/oracle_foveated.py --out C:/Users/ville/3dgrut_runs/oracle_mix
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from oracle_perceptual import box, region_mean, tile_mean
from threedgrut_playground.adaptive.evaluation.core_api import L_BLACK, L_PEAK
from threedgrut_playground.adaptive.evaluation.ldr import save_png
from threedgrut_playground.adaptive.evaluation.reflection_rate import cvvdp_heatmap

FOV_DEG = 60.0
ECC_BANDS = (0.0, 5.0, 10.0, 20.0, 30.0, 1e9)
BUDGET_SPP = (2, 4, 8, 16, 32)
LAMBDAS = torch.logspace(-5, 3, 72).tolist()
SMOOTH = 5
FIGURE_BUDGET = 4


def log(text):
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


class FovMetric:
    def __init__(self, width, height, device):
        import pyfvvdp

        photo = pyfvvdp.fvvdp_display_photo_eotf(
            Y_peak=float(L_PEAK), contrast=float(L_PEAK) / float(L_BLACK), EOTF="sRGB"
        )
        geom = pyfvvdp.fvvdp_display_geometry((width, height), fov_horizontal=FOV_DEG)
        kw = dict(display_photometry=photo, display_geometry=geom, foveated=True, quiet=True, device=device)
        self.score_metric = pyfvvdp.fvvdp(**kw)
        self.map_metric = pyfvvdp.fvvdp(heatmap="raw", **kw)

    def jod(self, img, ref, gaze):
        q, _ = self.score_metric.predict(img.float(), ref.float(), dim_order="HWC", fixation_point=torch.tensor(gaze))
        return float(q)

    def heatmap(self, img, ref, gaze):
        _, st = self.map_metric.predict(img.float(), ref.float(), dim_order="HWC", fixation_point=torch.tensor(gaze))
        return st["heatmap"][0, 0, 0].to(ref.device, torch.float32)


def eccentricity(width, height, gaze, device):
    """``[H, W]`` degrees between each pixel's view direction and the gaze direction (pinhole, horizontal FOV)."""
    f = (width / 2) / math.tan(math.radians(FOV_DEG / 2))
    ys, xs = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    d = torch.stack([xs + 0.5 - width / 2, ys + 0.5 - height / 2, torch.full_like(xs, f, dtype=torch.float32)], -1)
    g = torch.tensor([gaze[0] + 0.5 - width / 2, gaze[1] + 0.5 - height / 2, f], device=device, dtype=torch.float32)
    cos = (d.float() @ g) / (d.float().norm(dim=-1) * g.norm())
    return torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0)))


def band_of(ecc):
    edges = torch.tensor(ECC_BANDS[1:-1], device=ecc.device)
    return torch.bucketize(ecc, edges)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path("C:/Users/ville/3dgrut_runs/oracle_mix"))
    args = p.parse_args(argv)
    dev = torch.device("cuda")
    c = torch.load(args.out / "options.pt", map_location=dev)
    names, cost, A, B, ref = c["options"], c["cost"], c["A"], c["B"], c["reference"]
    ids, obj_names, ppd, mc_ms = c["object_ids"], c["object_names"], c["ppd"], c["mc_ms"]
    K, H, W = cost.shape
    out = args.out / "foveated"
    out.mkdir(exist_ok=True)
    n_obj = len(obj_names) + 1  # objects 0..n-1, plus background (-1) as the last cell
    obj_idx = torch.where(ids >= 0, ids, torch.full_like(ids, n_obj - 1))
    n_band = len(ECC_BANDS) - 1

    gazes = {}
    for i, name in enumerate(obj_names):
        ys, xs = torch.nonzero(ids == i, as_tuple=True)
        gazes[f"on {name}"] = (int(xs.float().median()), int(ys.float().median()))
    gazes["on background"] = (W // 2, H // 8)
    log(f"gazes: {gazes}")

    log("gaze-blind CVVDP heatmaps")
    blind = torch.stack([cvvdp_heatmap(A[k], ref, ppd) for k in range(K)])
    blind_material = torch.stack([region_mean(blind[k], obj_idx) for k in range(K)])
    material_cost = torch.stack([region_mean(cost[k], obj_idx) for k in range(K)])  # a material rule sees mean cost
    # pycvvdp and pyfvvdp both import top-level helper modules by bare name (``interp``, ``utils``...): drop
    # pycvvdp's from sys.modules so pyfvvdp loads its own. The CVVDP maps above are already computed.
    import pycvvdp

    cv_dir = Path(pycvvdp.__file__).parent
    for mod_name, mod in list(sys.modules.items()):
        mod_file = getattr(mod, "__file__", None)
        if "." not in mod_name and mod_file and Path(mod_file).parent == cv_dir:
            del sys.modules[mod_name]
    metric = FovMetric(W, H, dev)

    per_gaze = {}
    for gname, gaze in gazes.items():
        hm = torch.stack([metric.heatmap(A[k], ref, gaze) for k in range(K)])
        cell = obj_idx * n_band + band_of(eccentricity(W, H, gaze, dev))
        per_gaze[gname] = {"heat": hm, "cell": cell}
    log("foveated heatmaps done")

    def compose(choice):
        img = torch.gather(B, 0, choice[None, :, :, None].expand(1, H, W, 3))[0]
        return img, torch.gather(cost, 0, choice[None])[0].sum().item()

    lam_scale = (H * W) / 1000.0

    def table_choice(train, test_cell, lam):
        """Fit one option per cell on the training gazes, apply it to ``test_cell``."""
        n_cells = n_obj * n_band
        err = torch.zeros(K, n_cells, device=dev)
        cst = torch.zeros(K, n_cells, device=dev)
        cnt = torch.zeros(n_cells, device=dev)
        for g in train:
            cell = per_gaze[g]["cell"].flatten()
            err.index_add_(1, cell, per_gaze[g]["heat"].flatten(1))
            cst.index_add_(1, cell, cost.flatten(1))
            cnt.index_add_(0, cell, torch.ones_like(cell, dtype=torch.float32))
        pick = torch.argmin(err + lam * cst * lam_scale, dim=0)
        # cells never seen in training: the nearest band of the same object that was seen
        seen = cnt > 0
        for cidx in torch.nonzero(~seen).flatten().tolist():
            o, b = divmod(cidx, n_band)
            near = [
                o * n_band + bb for bb in sorted(range(n_band), key=lambda bb: abs(bb - b)) if seen[o * n_band + bb]
            ]
            if near:
                pick[cidx] = pick[near[0]]
        return pick[test_cell]

    rules = (
        "material, gaze-blind",
        "material, per gaze",
        "table, held-out gaze",
        "table, per gaze",
        "tile 16",
        "pixel",
    )
    curves = {g: {} for g in gazes}
    for gname, gaze in gazes.items():
        hm, cell = per_gaze[gname]["heat"], per_gaze[gname]["cell"]
        sel = {
            "material, per gaze": torch.stack([region_mean(hm[k], obj_idx) for k in range(K)]),
            "tile 16": torch.stack([tile_mean(hm[k], 16) for k in range(K)]),
            "pixel": torch.stack([box(hm[k], SMOOTH) for k in range(K)]),
            "material, gaze-blind": blind_material,
        }
        others = [g for g in gazes if g != gname]
        for rule in rules:
            pts, seen = [], set()
            for lam in LAMBDAS:
                if rule == "table, held-out gaze":
                    choice = table_choice(others, cell, lam)
                elif rule == "table, per gaze":
                    choice = table_choice([gname], cell, lam)
                else:
                    rc = material_cost if rule.startswith("material") else cost
                    choice = torch.argmin(sel[rule] + lam * rc * lam_scale, dim=0)
                key = hash(choice.cpu().numpy().tobytes())
                if key in seen:
                    continue
                seen.add(key)
                img, ms = compose(choice)
                pts.append({"lambda": lam, "ms": ms, "jod": metric.jod(img, ref, gaze), "choice": choice})
            pts.sort(key=lambda q: q["ms"])
            curves[gname][rule] = pts
        log(f"{gname}: curves done")

    uniform = {
        g: [{"name": names[k], "ms": cost[k].sum().item(), "jod": metric.jod(B[k], ref, gz)} for k in range(K)]
        for g, gz in gazes.items()
    }

    def at_budget(pts, budget):
        fit = [q for q in pts if q["ms"] <= budget * 1.02]
        return max(fit, key=lambda q: q["ms"]) if fit else None

    headline = []
    for n in BUDGET_SPP:
        budget = n * mc_ms
        row = {"budget_as_mc_spp": n, "budget_ms": budget, "per_gaze": {}}
        means = {r: [] for r in rules + ("best uniform",)}
        for g in gazes:
            ok = [u for u in uniform[g] if u["ms"] <= budget * 1.02]
            bu = max(ok, key=lambda u: u["jod"]) if ok else None
            entry = {"best uniform": None if bu is None else (bu["name"], bu["jod"])}
            if bu is not None:
                means["best uniform"].append(bu["jod"])
            for r in rules:
                q = at_budget(curves[g][r], budget)
                entry[r] = None if q is None else q["jod"]
                if q is not None:
                    means[r].append(q["jod"])
            row["per_gaze"][g] = entry
        row["mean_over_gazes"] = {r: (sum(v) / len(v) if len(v) == len(gazes) else None) for r, v in means.items()}
        headline.append(row)
        log(
            f"budget MC {n:>2} spp ({budget:6.1f} ms) mean JOD: "
            + " | ".join(f"{r} {'-' if v is None else round(v, 2)}" for r, v in row["mean_over_gazes"].items())
        )

    # Figure: mean JOD over gazes against frame time, rules interpolated on a common time grid
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = torch.logspace(math.log10(mc_ms * 1.05), math.log10(mc_ms * 64), 40).tolist()

    def mean_curve(get_pts):
        ys = []
        for t in grid:
            vals = [at_budget(get_pts(g), t) for g in gazes]
            ys.append(None if any(v is None for v in vals) else sum(v["jod"] for v in vals) / len(vals))
        return ys

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = {
        "material, gaze-blind": "#888",
        "material, per gaze": "#229954",
        "table, held-out gaze": "#7d3c98",
        "table, per gaze": "#2471a3",
        "tile 16": "#17a589",
        "pixel": "#c0392b",
    }
    for r in rules:
        ys = mean_curve(lambda g, r=r: curves[g][r])
        xs = [t for t, y in zip(grid, ys) if y is not None]
        ax.plot(
            xs,
            [y for y in ys if y is not None],
            "-",
            color=colors[r],
            label=f"oracle {r}" if "held-out" not in r else "lookup (material x eccentricity), fitted on other gazes",
            lw=2.2 if "held-out" in r else 1.4,
        )
    for kind, style in (("MC", "o-"), ("cone", "s--")):
        pts = [
            (u["ms"], sum(uniform[g][k]["jod"] for g in gazes) / len(gazes))
            for k, u in enumerate(uniform[next(iter(gazes))])
            if u["name"].startswith(kind)
        ]
        ax.plot(*zip(*pts), style, color="#333" if kind == "MC" else "#aaa", label=f"uniform {kind}", ms=4)
    ax.set_xscale("log")
    ax.set_xlabel("frame time (ms, sum of per-pixel costs)")
    ax.set_ylabel("FovVideoVDP JOD, mean over 6 gazes (higher is better)")
    ax.set_title("Room ladder 512 px, foveated: lookup on (material, eccentricity) vs oracles")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "foveated_curves.png", dpi=130)

    # Choice maps at one budget for one gaze
    n_mc = sum(1 for nm in names if nm.startswith("MC"))
    palette = torch.tensor(
        [[0.15 + 0.85 * i / max(n_mc - 1, 1)] * 3 for i in range(n_mc)]
        + [[1.0, 0.85 - 0.6 * j / max(K - n_mc - 1, 1), 0.1] for j in range(K - n_mc)],
        device=dev,
    )
    for g in list(gazes)[:2]:
        for r in ("table, held-out gaze", "pixel", "material, gaze-blind"):
            q = at_budget(curves[g][r], FIGURE_BUDGET * mc_ms)
            if q is None:
                continue
            tag = f"{g}_{r}".replace(", ", "_").replace(" ", "_")
            m = palette[q["choice"]].clone()
            gx, gy = gazes[g]
            m[max(gy - 4, 0) : gy + 5, max(gx - 4, 0) : gx + 5] = torch.tensor([0.0, 0.6, 1.0], device=dev)
            save_png(m, out / f"choice_{tag}_mc{FIGURE_BUDGET}budget.png")

    result = {
        "gazes": gazes,
        "ecc_bands_deg": ECC_BANDS,
        "headline": headline,
        "curves": {
            g: {r: [{k: v for k, v in q.items() if k != "choice"} for q in pts] for r, pts in rs.items()}
            for g, rs in curves.items()
        },
        "uniform": uniform,
    }
    (out / "results.json").write_text(json.dumps(result, indent=2, default=float))
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
