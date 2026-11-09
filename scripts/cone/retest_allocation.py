# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fair retest of the cone/MC allocation claim, after the 2026-11-08 review (C:/Users/ville/3dgrut_runs/review).

Changes against oracle_mix.py / oracle_perceptual.py, one per review finding:

- MC picks its lobe by weight (kernel fix in ``sampled_microfacet_brdf``): no samples wasted on a metal's diffuse lobe.
- MC reuses the activated particle buffer between frames (``Tracer._particle_buffers``): no per-frame rebuild overhead.
- Cone frames trace camera rays as thin rays (``engine.cone_camera_rays = False``): cones only on reflections, so a
  "uniform cone" frame is the glossy-only cone baseline, and background pixels cost and look like MC.
- Caps avoid band edges (the cost sawtooth peaks at s_k = 0.005 * 2^(k-1)).
- The per-material rule is solved exactly (multiple-choice knapsack by dynamic programming), not by a lambda sweep.
- Everything is scored at three viewing conditions: the camera's own 8.53 ppd, 30 ppd and 60 ppd.
- The per-tile cost model is checked against real mixed frames: an oracle's choice map is rendered as compacted launches
  (one per cone cap, one per MC pass over the pixels that still need samples) and timed, and the image it produces is
  scored too.

Usage (from the repository root, .venv/Scripts on PATH, PYTHONUTF8=1):
    python scripts/cone/retest_allocation.py --out C:/Users/ville/3dgrut_runs/retest
    python scripts/cone/retest_allocation.py --out C:/Users/ville/3dgrut_runs/retest --reuse   # analysis only
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from oracle_mix import MC_SPP, RES, Bench, box, median_ms, tile_mean
from oracle_perceptual import region_mean
from threedgrut_playground.adaptive.evaluation.cone_equal_time import ConeSetting, configure, to_display
from threedgrut_playground.adaptive.evaluation.ldr import save_png
from threedgrut_playground.adaptive.evaluation.metrics import cvvdp_jod
from threedgrut_playground.adaptive.evaluation.reflection_rate import cvvdp_heatmap

CAPS = (0.015, 0.03, 0.06, 0.1, 0.2, 0.35)  # none on a band edge (0.005 * 2^k)
SEED_B = 1 << 20
SEED_MIXED = 1 << 22
PPDS = (None, 30.0, 60.0)  # None: the camera's own
BUDGET_SPP = (1, 2, 4, 8, 16, 32)
VALIDATE_SPP = (2, 4, 8, 16)
LAMBDAS = torch.logspace(-5, 3, 72).tolist()
DP_BIN_MS = 0.25


def log(text):
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def set_mode(b, cap):
    configure(b.engine, None if cap is None else ConeSetting(cap, 1.0))
    b.engine.use_gaussian_lod = cap is not None
    b.engine.cone_camera_rays = False


def render_options(b, out):
    options, timing = [], {}
    set_mode(b, None)
    mc_ms = median_ms(lambda: b.frame())
    mc_cost, info = b.tile_cost(mc_ms)
    timing["mc_1spp"] = {"full_ms": mc_ms, **info}
    log(f"MC 1 spp {mc_ms:.2f} ms {info}")
    a, bb = b.means(set(MC_SPP), 0), b.means(set(MC_SPP), SEED_B)
    for n in MC_SPP:
        options.append(
            {"name": f"MC {n}", "kind": "mc", "spp": n, "cap": 0.0, "cost": mc_cost * n, "A": a[n], "B": bb[n]}
        )
    for cap in CAPS:
        set_mode(b, cap)
        b.frame()  # builds the LOD slots for this cap
        ms = median_ms(lambda: b.frame(), frames=5 if cap >= 0.1 else 15)
        cost, info = b.tile_cost(ms)
        timing[f"cone_{cap:g}"] = {"full_ms": ms, **info}
        log(f"cone {cap:g}: {ms:.1f} ms {info}")
        a, bb = b.means({1}, 0), b.means({1}, SEED_B)
        options.append(
            {"name": f"cone {cap:g}", "kind": "cone", "spp": 1, "cap": cap, "cost": cost, "A": a[1], "B": bb[1]}
        )
    set_mode(b, None)
    data = {
        "options": [o["name"] for o in options],
        "kind": [o["kind"] for o in options],
        "spp": [o["spp"] for o in options],
        "cap": [o["cap"] for o in options],
        "cost": torch.stack([o["cost"] for o in options]).cpu(),
        "A": torch.stack([o["A"] for o in options]).cpu(),
        "B": torch.stack([o["B"] for o in options]).cpu(),
        "reference": b.reference.cpu(),
        "object_ids": b.object_ids.cpu(),
        "object_names": b.names,
        "ppd": b.ppd,
        "mc_ms": mc_ms,
        "timing": timing,
    }
    torch.save(data, out / "options.pt")
    return data


def exact_region_choice(err, cost, region, budget):
    """Exact multiple-choice knapsack: one option per region minimizing summed error with summed cost <= budget.

    err, cost: [K, H, W]; region: [H, W] ints. Costs are binned to DP_BIN_MS (rounded up, so the result never exceeds
    the budget). Returns a [H, W] choice, or None if even the cheapest assignment is over budget.
    """
    ids = region.unique().tolist()
    K = err.shape[0]
    e = torch.stack([err[:, region == r].sum(dim=1) for r in ids], dim=0).double().cpu()  # [R, K]
    c = torch.stack([cost[:, region == r].sum(dim=1) for r in ids], dim=0).double().cpu()
    cb = torch.ceil(c / DP_BIN_MS).long()
    nb = int(budget / DP_BIN_MS) + 1
    inf = float("inf")
    dp = torch.full((nb,), inf, dtype=torch.float64)
    dp[0] = 0.0
    picks = []
    for r in range(len(ids)):
        new = torch.full((nb,), inf, dtype=torch.float64)
        arg = torch.full((nb,), -1, dtype=torch.long)
        for k in range(K):
            s = int(cb[r, k])
            if s >= nb:
                continue
            cand = torch.full((nb,), inf, dtype=torch.float64)
            cand[s:] = dp[: nb - s] + e[r, k]
            better = cand < new
            new[better], arg[better] = cand[better], k
        dp = new
        picks.append(arg)
    if not torch.isfinite(dp).any():
        return None
    bin_ = int(torch.argmin(dp))
    choice_r = [0] * len(ids)
    for r in reversed(range(len(ids))):
        k = int(picks[r][bin_])
        choice_r[r] = k
        bin_ -= int(cb[r, k])
    choice = torch.zeros_like(region)
    for r, k in zip(ids, choice_r):
        choice[region == r] = k
    return choice


def analyse(data, out):
    dev = torch.device("cuda")
    names, cost, A, B, ref = (
        data["options"],
        data["cost"].to(dev),
        data["A"].to(dev),
        data["B"].to(dev),
        data["reference"].to(dev),
    )
    ids, mc_ms = data["object_ids"].to(dev), data["mc_ms"]
    K, H, W = cost.shape
    lam_scale = H * W / 1000.0

    def compose(choice):
        return (
            torch.gather(B, 0, choice[None, :, :, None].expand(1, H, W, 3))[0],
            torch.gather(cost, 0, choice[None])[0].sum().item(),
        )

    results, kept = {}, {}
    for ppd in PPDS:
        p = data["ppd"] if ppd is None else ppd
        tag = f"{p:.2f}ppd"
        heat = torch.stack([cvvdp_heatmap(A[k], ref, p) for k in range(K)])
        jod = lambda img: cvvdp_jod(img, ref, p)
        uniform = [{"name": names[k], "ms": cost[k].sum().item(), "jod": jod(B[k])} for k in range(K)]
        sel = {
            "tile 32": torch.stack([tile_mean(h, 32) for h in heat]),
            "pixel": torch.stack([box(h, 5) for h in heat]),
        }
        curves = {}
        for rule, e in sel.items():
            pts, seen = [], set()
            for lam in LAMBDAS:
                choice = torch.argmin(e + lam * cost * lam_scale, dim=0)
                key = hash(choice.cpu().numpy().tobytes())
                if key in seen:
                    continue
                seen.add(key)
                img, ms = compose(choice)
                pts.append({"ms": ms, "jod": jod(img), "choice": choice})
            curves[rule] = sorted(pts, key=lambda q: q["ms"])
        rows = []
        for n in BUDGET_SPP:
            budget = n * mc_ms
            ok = [u for u in uniform if u["ms"] <= budget * 1.02]
            best = max(ok, key=lambda u: u["jod"])
            row = {"budget_spp": n, "budget_ms": budget, "best_uniform": best["name"], "best_uniform_jod": best["jod"]}
            mat = exact_region_choice(heat, cost, ids, budget * 1.02)
            if mat is not None:
                img, ms = compose(mat)
                row["per_material"] = {"ms": ms, "jod": jod(img)}
            for rule, pts in curves.items():
                fit = [q for q in pts if q["ms"] <= budget * 1.02]
                if fit:
                    q = max(fit, key=lambda q: q["ms"])
                    row[rule] = {"ms": q["ms"], "jod": q["jod"]}
                    if ppd is None and n in VALIDATE_SPP:
                        kept[(rule, n)] = q["choice"]
            rows.append(row)
            log(
                f"{tag} budget MC {n:>2} ({budget:6.1f} ms): uniform {best['name']:<9} {best['jod']:.2f} | per material "
                f"{row.get('per_material', {}).get('jod', float('nan')):.2f} | tile 32 "
                f"{row.get('tile 32', {}).get('jod', float('nan')):.2f} | pixel {row.get('pixel', {}).get('jod', float('nan')):.2f}"
            )
        results[tag] = {"uniform": uniform, "budgets": rows}
    return results, kept


def time_mixed(b, data, choice, reps=3):
    """Render ``choice`` ([H, W] option index) as compacted launches; return (measured ms, image, per-group ms)."""
    dev = b.device
    kind, spp, cap = data["kind"], data["spp"], data["cap"]
    flat = choice.flatten().to(dev)
    o, d = b.rays_o.reshape(-1, 3), b.rays_d.reshape(-1, 3)
    img = torch.zeros(RES * RES, 3, device=dev)
    groups = []  # (label, cap or None, pixel indices, frame id)
    for k in sorted(set(flat.tolist())):
        if kind[k] == "cone":
            groups.append((f"cone {cap[k]:g}", cap[k], torch.nonzero(flat == k).flatten(), SEED_MIXED))
    n_pix = torch.tensor([spp[k] if kind[k] == "mc" else 0 for k in range(len(kind))], device=dev)[flat]
    for p in range(int(n_pix.max().item())):
        groups.append((f"mc pass {p}", None, torch.nonzero(n_pix > p).flatten(), SEED_MIXED + 1 + p))

    def launch(group):
        _, gcap, idx, fid = group
        return b.frame(o[idx][None, None].contiguous(), d[idx][None, None].contiguous(), frame_id=fid)[0]

    per_group = {}
    for g in groups:
        set_mode(b, g[1])
        out = launch(g)  # warm-up; also builds the LOD slots for this cap outside the timing
        if g[1] is None:
            img[g[2]] += out / n_pix[g[2]][:, None]
        else:
            img[g[2]] = out
        per_group[g[0]] = median_ms(lambda: launch(g), warmups=1, frames=reps)
    set_mode(b, None)
    image = to_display(b.engine, img.reshape(RES, RES, 3))
    return sum(per_group.values()), image, per_group


def validate(b, data, kept, out):
    ref, ppd = b.reference, b.ppd
    names = data["options"]
    rows = []
    for n in VALIDATE_SPP:  # uniform MC through the same harness: n full-frame passes
        choice = torch.full((RES, RES), names.index(f"MC {n}"), dtype=torch.long)
        ms, img, _ = time_mixed(b, data, choice)
        rows.append(
            {
                "allocation": f"uniform MC {n}",
                "predicted_ms": n * data["mc_ms"],
                "measured_ms": ms,
                "jod": cvvdp_jod(img, ref, ppd),
            }
        )
        log(
            f"validate uniform MC {n}: predicted {n * data['mc_ms']:.1f} ms, measured {ms:.1f} ms, JOD {rows[-1]['jod']:.2f}"
        )
    for k, nm in enumerate(names):
        if data["kind"][k] == "cone":
            choice = torch.full((RES, RES), k, dtype=torch.long)
            ms, img, _ = time_mixed(b, data, choice)
            pred = data["cost"][k].sum().item()
            rows.append(
                {
                    "allocation": f"uniform {nm}",
                    "predicted_ms": pred,
                    "measured_ms": ms,
                    "jod": cvvdp_jod(img, ref, ppd),
                }
            )
            log(f"validate uniform {nm}: predicted {pred:.1f} ms, measured {ms:.1f} ms, JOD {rows[-1]['jod']:.2f}")
    for (rule, n), choice in sorted(kept.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        pred = torch.gather(data["cost"], 0, choice.cpu()[None])[0].sum().item()
        ms, img, groups = time_mixed(b, data, choice)
        jod = cvvdp_jod(img, ref, ppd)
        rows.append(
            {
                "allocation": f"oracle {rule} @ MC {n} budget",
                "predicted_ms": pred,
                "measured_ms": ms,
                "jod": jod,
                "groups_ms": groups,
            }
        )
        save_png(img, out / f"mixed_{rule.replace(' ', '')}_mc{n}.png")
        log(
            f"validate oracle {rule} @ MC {n}: predicted {pred:.1f} ms, measured {ms:.1f} ms ({len(groups)} launches), JOD {jod:.2f}"
        )
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path("C:/Users/ville/3dgrut_runs/retest"))
    p.add_argument("--reuse", action="store_true", help="skip rendering and load <out>/options.pt")
    p.add_argument("--no-validate", action="store_true")
    args = p.parse_args(argv)
    from types import SimpleNamespace

    bench_args = SimpleNamespace(
        ckpt=REPO_ROOT / "runs/room_3dgrt/room-0211_151617/ckpt_last.pt",
        ref=Path("C:/Users/ville/3dgrut_runs/cone_equal_time/reference_unclamped_512px_16384spp.pt"),
        out=args.out,
    )
    b = Bench(bench_args)
    out = b.out
    data = torch.load(out / "options.pt") if args.reuse else render_options(b, out)
    results, kept = analyse(data, out)
    summary = {"mc_ms": data["mc_ms"], "timing": data["timing"], "ppd_native": data["ppd"], "results": results}
    if not args.no_validate:
        summary["validation"] = validate(b, data, kept, out)
    (out / "results.json").write_text(json.dumps(summary, indent=2, default=float))
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
