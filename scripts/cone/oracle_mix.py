# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Oracle go/no-go for the per-pixel choice "wider cone or more samples" (room roughness ladder, 512 px, close camera).

Question: if an oracle picks, per pixel, the option with the lowest error, does the mix beat the best UNIFORM option
(one setting for the whole frame) at equal time? If not even the oracle wins, there is no allocator paper.

Options: MC at n spp (cone off, plain BVH) and LOD cone frames at several caps (every PBR hit is a cone, max roughness
1). Every option is rendered twice with different random numbers: realization A picks, realization B
is scored, so the oracle cannot win by picking lucky noise.

Cost per pixel: each base option (MC 1 spp, cone at cap c) is timed on every 32x32 tile, launched as 8x8 copies of
the tile's rays so the GPU is full; the tile times are rescaled to sum to the full-frame time. A pixel costs its tile's
time / pixels, times the frame count. This assumes a mixed frame costs the sum of its pixels' costs (an idealized
implementation; an upper bound on what a real allocator gets).

Choice: per pixel (error maps box-filtered 5x5 before choosing) and per 16x16 tile, minimizing error + lambda * cost,
swept over lambda. Error is FLIP against the 16k spp reference; the headline is mean FLIP over the glossy objects
(floor + spheres). Frames are unclamped (``configure``) and clamped once for display.

Usage: python scripts/cone/oracle_mix.py --out C:/Users/ville/3dgrut_runs/oracle_mix
"""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from threedgrut_playground.adaptive.evaluation.cone_equal_time import (
    ConeSetting,
    configure,
    load_ladder,
    object_roughness,
    render_frame,
    to_display,
)
from threedgrut_playground.adaptive.evaluation.ldr import ensure_outside_repo, save_png
from threedgrut_playground.adaptive.evaluation.metrics import cvvdp_jod, flip_error_map
from threedgrut_playground.adaptive.evaluation.reflection_rate import LADDER_PRESET, object_id_map
from threedgrut_playground.adaptive.evaluation.sampling import trace_only_renderer
from threedgrut_playground.adaptive.evaluation.scenes import glossy_layout, scene_frame
from threedgrut_playground.adaptive.evaluation.viewing import ppd_from_camera

RES = 512
TILE = 32
REPLICATE = 8  # each tile's rays are launched 8x8 times so the launch fills the GPU (throughput, not latency)
CHOICE_TILE = 16
SMOOTH = 5
MC_SPP = (1, 2, 4, 8, 16, 32, 64, 128)
CONE_CAPS = (0.02, 0.05, 0.1, 0.2, 0.35)
CONE_FRAMES = (1,)  # cone frames are deterministic on the metal ladder; x4 equals x1
SEED_B = 1 << 20  # realization B's first frame id
BUDGET_SPP = (2, 4, 8, 16, 32, 64)  # headline budgets: the time of uniform MC at this many spp
WARMUPS, FRAMES, TILE_FRAMES = 3, 15, 5


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", type=Path, default=REPO_ROOT / "runs/room_3dgrt/room-0211_151617/ckpt_last.pt")
    p.add_argument(
        "--ref",
        type=Path,
        default=Path("C:/Users/ville/3dgrut_runs/cone_equal_time/reference_unclamped_512px_16384spp.pt"),
    )
    p.add_argument("--out", type=Path, default=Path("C:/Users/ville/3dgrut_runs/oracle_mix"))
    p.add_argument("--caps", default=",".join(f"{c:g}" for c in CONE_CAPS))
    return p.parse_args(argv)


def log(text):
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def median_ms(step, warmups=WARMUPS, frames=FRAMES):
    for _ in range(warmups):
        step()
    torch.cuda.synchronize()
    times = []
    for _ in range(frames):
        start = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times)


class Bench:
    def __init__(self, args):
        from threedgrut_playground.engine import Engine3DGRUT

        self.out = ensure_outside_repo(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.engine = e = Engine3DGRUT(
            gs_object=str(args.ckpt),
            mesh_assets_folder=str(REPO_ROOT / "threedgrut_playground" / "assets"),
            default_config="apps/colmap_3dgrt.yaml",
        )
        self.device = torch.device(e.device)
        self.preset = load_ladder(e)
        self.camera = self.preset.to_camera(RES, RES, device=self.device)
        rays = e.raygen(self.camera, use_spp=False)
        self.rays_o, self.rays_d = rays.rays_ori.contiguous(), rays.rays_dir.contiguous()
        self.ppd = ppd_from_camera(self.camera)
        layout = glossy_layout(scene_frame(e.scene_mog.positions.detach().cpu()), LADDER_PRESET)
        self.names = layout.object_names()
        self.object_ids = object_id_map(trace_only_renderer(e), self.camera, layout, self.device)
        self.roughness = object_roughness(self.names)
        self.glossy = self.object_ids >= 0
        cached = torch.load(args.ref, map_location=self.device)
        self.reference = to_display(e, (cached["even"] + cached["odd"]) / 2.0)
        half = flip_error_map(to_display(e, cached["even"]), to_display(e, cached["odd"]), self.ppd)
        self.reference_noise = half[self.glossy].mean().item()

    def set_mode(self, cap):
        configure(self.engine, None if cap is None else ConeSetting(cap, 1.0))
        self.engine.use_gaussian_lod = cap is not None

    def frame(self, o=None, d=None, frame_id=0, fov_scale=1.0):
        self.engine.camera_fov = self.preset.fov_deg * fov_scale
        self.engine.frame_id = frame_id
        return render_frame(self.engine, self.rays_o if o is None else o, self.rays_d if d is None else d)

    def means(self, counts, first):
        """Unclamped means of the first n frames for each n in ``counts``, as displayed images."""
        out, acc = {}, None
        for i in range(max(counts)):
            f = self.frame(frame_id=first + i)
            acc = f if acc is None else acc + f
            if i + 1 in counts:
                out[i + 1] = to_display(self.engine, acc / (i + 1))
        return out

    def tile_cost(self, full_ms):
        """``[RES, RES]`` ms per pixel for one frame of the current mode."""
        n = RES // TILE
        side = TILE * REPLICATE
        times = torch.zeros(n, n)
        for ty in range(n):
            for tx in range(n):
                sl = (slice(None), slice(ty * TILE, (ty + 1) * TILE), slice(tx * TILE, (tx + 1) * TILE))
                o = self.rays_o[sl].repeat(1, REPLICATE, REPLICATE, 1).contiguous()
                d = self.rays_d[sl].repeat(1, REPLICATE, REPLICATE, 1).contiguous()
                times[ty, tx] = median_ms(lambda: self.frame(o, d, fov_scale=side / RES), warmups=1, frames=2)
        raw_sum = times.sum().item() / (REPLICATE * REPLICATE)
        times *= full_ms / times.sum().item()
        per_pixel = times.repeat_interleave(TILE, 0).repeat_interleave(TILE, 1) / (TILE * TILE)
        return per_pixel.to(self.device), {
            "tile_sum_ms": raw_sum,
            "tile_max_over_mean": (times.max() / times.mean()).item(),
        }


def box(x, k):
    return F.avg_pool2d(x[None, None], k, stride=1, padding=k // 2, count_include_pad=False)[0, 0]


def tile_mean(x, k):
    return F.avg_pool2d(x[None, None], k)[0, 0].repeat_interleave(k, 0).repeat_interleave(k, 1)


def main(argv=None):
    args = parse_args(argv)
    caps = tuple(float(c) for c in args.caps.split(","))
    b = Bench(args)
    log(f"ppd {b.ppd:.2f}, glossy pixels {int(b.glossy.sum())}, reference noise FLIP {b.reference_noise:.4f}")

    options = []  # dicts: name, kind, cap, frames, cost [H,W] ms, A, B images
    timing = {}
    b.set_mode(None)
    mc_ms = median_ms(lambda: b.frame())
    mc_cost, info = b.tile_cost(mc_ms)
    timing["mc_1spp"] = {"full_ms": mc_ms, **info}
    log(f"MC 1 spp {mc_ms:.2f} ms ({info})")
    imgs_a, imgs_b = b.means(set(MC_SPP), 0), b.means(set(MC_SPP), SEED_B)
    for n in MC_SPP:
        options.append(
            {"name": f"MC {n}", "kind": "mc", "frames": n, "cost": mc_cost * n, "A": imgs_a[n], "B": imgs_b[n]}
        )
    for cap in caps:
        b.set_mode(cap)
        b.frame()  # builds the LOD slots for this cap
        ms = median_ms(lambda: b.frame(), frames=5 if cap >= 0.1 else FRAMES)
        cost, info = b.tile_cost(ms)
        timing[f"cone_{cap:g}"] = {"full_ms": ms, **info}
        log(f"cone {cap:g}: {ms:.1f} ms ({info})")
        imgs_a, imgs_b = b.means(set(CONE_FRAMES), 0), b.means(set(CONE_FRAMES), SEED_B)
        for n in CONE_FRAMES:
            options.append(
                {
                    "name": f"cone {cap:g} x{n}",
                    "kind": "cone",
                    "cap": cap,
                    "frames": n,
                    "cost": cost * n,
                    "A": imgs_a[n],
                    "B": imgs_b[n],
                }
            )
    b.set_mode(None)
    torch.save(  # everything the analysis needs, so other selection rules can be tried without rendering again
        {
            "options": [o["name"] for o in options],
            "cost": torch.stack([o["cost"] for o in options]).cpu(),
            "A": torch.stack([o["A"] for o in options]).cpu(),
            "B": torch.stack([o["B"] for o in options]).cpu(),
            "reference": b.reference.cpu(),
            "object_ids": b.object_ids.cpu(),
            "object_names": b.names,
            "ppd": b.ppd,
            "mc_ms": mc_ms,
            "timing": timing,
        },
        b.out / "options.pt",
    )

    ref, ppd, glossy = b.reference, b.ppd, b.glossy
    err_a = torch.stack([flip_error_map(o["A"], ref, ppd) for o in options])  # [K,H,W]
    cost = torch.stack([o["cost"] for o in options])
    imgs_b = torch.stack([o["B"] for o in options])  # [K,H,W,3]
    total_ms = lambda c: c.sum().item()

    def score(img):
        flip = flip_error_map(img, ref, ppd)
        per_obj = {
            n: flip[b.object_ids == i].mean().item() for i, n in enumerate(b.names) if bool((b.object_ids == i).any())
        }
        return flip[glossy].mean().item(), per_obj

    uniform = []
    for k, o in enumerate(options):
        g, per_obj = score(o["B"])
        uniform.append({"name": o["name"], "ms": total_ms(o["cost"]), "glossy_flip": g, "per_object": per_obj})
        log(f"uniform {o['name']:<16} {uniform[-1]['ms']:9.1f} ms  glossy FLIP {g:.4f}")

    def compose(choice):
        img = torch.gather(imgs_b, 0, choice[None, :, :, None].expand(1, RES, RES, 3))[0]
        c = torch.gather(cost, 0, choice[None])[0]
        return img, c

    selectors = {
        "pixel": torch.stack([box(e, SMOOTH) for e in err_a]),
        "tile16": torch.stack([tile_mean(e, CHOICE_TILE) for e in err_a]),
    }
    obj_err = torch.zeros_like(err_a)  # per-object oracle: one option per object (and per background)
    for i in range(-1, len(b.names)):
        m = b.object_ids == i
        if bool(m.any()):
            obj_err[:, m] = err_a[:, m].mean(dim=1, keepdim=True)
    selectors["object"] = obj_err
    cost_sel = {"pixel": cost, "tile16": cost, "object": cost}

    lambdas = torch.logspace(-6, 2, 120).tolist()
    curves = {}
    for sel_name, sel in selectors.items():
        pts = []
        for lam in lambdas:
            choice = torch.argmin(sel + lam * cost_sel[sel_name] * (RES * RES), dim=0)
            img, c = compose(choice)
            pts.append((lam, total_ms(c), choice))
        curves[sel_name] = pts

    def best_uniform(budget):
        ok = [u for u in uniform if u["ms"] <= budget * 1.02]
        return min(ok, key=lambda u: u["glossy_flip"]) if ok else None

    headline = []
    choice_maps = {}
    for n in BUDGET_SPP:
        budget = n * mc_ms
        row = {"budget_ms": budget, "budget_as_mc_spp": n}
        bu = best_uniform(budget)
        row["best_uniform"] = (
            None if bu is None else {"name": bu["name"], "ms": bu["ms"], "glossy_flip": bu["glossy_flip"]}
        )
        row["mc_uniform_flip"] = next(u["glossy_flip"] for u in uniform if u["name"] == f"MC {n}")
        for sel_name, pts in curves.items():
            fit = [p for p in pts if p[1] <= budget * 1.02]
            if not fit:
                row[sel_name] = None
                continue
            lam, ms, choice = min(fit, key=lambda p: abs(p[1] - budget))
            img, _ = compose(choice)
            g, per_obj = score(img)
            shares = torch.bincount(choice[glossy].flatten(), minlength=len(options)).float() / int(glossy.sum())
            row[sel_name] = {
                "ms": ms,
                "glossy_flip": g,
                "per_object": per_obj,
                "glossy_pixel_share": {
                    options[k]["name"]: round(s, 4) for k, s in enumerate(shares.tolist()) if s > 0.005
                },
            }
            if sel_name == "pixel":
                choice_maps[n] = (choice, img)
                jod = cvvdp_jod(img, ref, ppd)
                row["pixel"]["cvvdp_jod"] = jod
                if bu is not None:
                    bu_img = next(o["B"] for o in options if o["name"] == bu["name"])
                    row["best_uniform_cvvdp_jod"] = cvvdp_jod(bu_img, ref, ppd)
        headline.append(row)
        px = row.get("pixel")
        log(
            f"budget = MC {n} spp ({budget:.1f} ms): best uniform "
            f"{'-' if bu is None else bu['name']} {'' if bu is None else round(bu['glossy_flip'], 4)}; oracle pixel "
            f"{'-' if px is None else round(px['glossy_flip'], 4)}, tile16 "
            f"{'-' if row['tile16'] is None else round(row['tile16']['glossy_flip'], 4)}, object "
            f"{'-' if row['object'] is None else round(row['object']['glossy_flip'], 4)}"
        )

    # Figures: cost/error curves and the oracle's choice maps
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    mc_pts = [(u["ms"], u["glossy_flip"]) for u in uniform if u["name"].startswith("MC")]
    ax.plot(*zip(*mc_pts), "o-", color="#555", label="uniform MC (1..128 spp)")
    for n in CONE_FRAMES:
        pts = [
            (u["ms"], u["glossy_flip"]) for u in uniform if u["name"].startswith("cone") and u["name"].endswith(f"x{n}")
        ]
        ax.plot(*zip(*pts), "s--", label=f"uniform cone x{n} (caps {','.join(f'{c:g}' for c in caps)})")
    styles = {
        "pixel": ("#c0392b", "oracle per pixel"),
        "tile16": ("#2471a3", "oracle per 16x16 tile"),
        "object": ("#229954", "oracle per object"),
    }
    for sel_name, pts in curves.items():
        xs, ys, seen = [], [], set()
        for lam, ms, choice in pts:
            if round(ms, 1) in seen:
                continue
            seen.add(round(ms, 1))
            xs.append(ms)
            ys.append(score(compose(choice)[0])[0])
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        ax.plot(
            [xs[i] for i in order], [ys[i] for i in order], "-", color=styles[sel_name][0], label=styles[sel_name][1]
        )
    ax.axhline(b.reference_noise, color="#999", ls=":", lw=1, label="reference half-vs-half noise")
    ax.set_xscale("log")
    ax.set_xlabel("frame time (ms, sum of per-pixel costs)")
    ax.set_ylabel("mean FLIP on glossy objects (lower is better)")
    ax.set_title("Room ladder 512 px: oracle mix of cones and MC vs uniform settings")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(b.out / "cost_error_curves.png", dpi=130)

    palette = torch.tensor(
        [[0.15 + 0.85 * i / (len(MC_SPP) - 1)] * 3 for i in range(len(MC_SPP))]
        + [[1.0, 0.85 - 0.6 * j / max(len(caps) - 1, 1), 0.1] for j in range(len(caps)) for _ in CONE_FRAMES],
        device=b.device,
    )
    for n, (choice, img) in choice_maps.items():
        save_png(palette[choice], b.out / f"choice_map_mc{n}budget.png")
        save_png(img, b.out / f"oracle_pixel_mc{n}budget.png")
    save_png(ref, b.out / "reference.png")

    result = {
        "timing": timing,
        "reference_noise_glossy_flip": b.reference_noise,
        "options": [o["name"] for o in options],
        "uniform": uniform,
        "headline": headline,
        "choice_map_palette": "grey = MC (dark 1 spp .. white 128 spp); yellow..orange = cones (narrow .. wide cap)",
    }
    (b.out / "results.json").write_text(json.dumps(result, indent=2, default=float))
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
