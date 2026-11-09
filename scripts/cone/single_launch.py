# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Times oracle allocations as ONE launch with a per-pixel mode (``Engine3DGRUT.pixel_mode``), against the per-tile
cost model they were chosen with and against uniform settings timed the same way.

Reads ``<retest>/options.pt`` from ``retest_allocation.py`` (fair MC, cones on reflections only, caps off band edges).
For each budget (the time of uniform MC n spp under the model) it re-derives the oracles at the camera's ppd: tile 32
and pixel (lambda sweep on the CVVDP heatmap of realization A) and per material (exact knapsack), turns each choice
map into a per-pixel mode (MC n -> n, cone cap c -> -c), and renders it in one 512 px launch. MC samples use frames
SEED_B + i, so the frame reproduces realization B. Uniform MC n and uniform cones are timed through the same path.

Usage: python scripts/cone/single_launch.py --retest C:/Users/ville/3dgrut_runs/retest
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from oracle_mix import Bench, box, tile_mean
from retest_allocation import LAMBDAS, SEED_B, exact_region_choice
from threedgrut_playground.adaptive.evaluation.cone_equal_time import ConeSetting, configure, render_frame, to_display
from threedgrut_playground.adaptive.evaluation.ldr import save_png
from threedgrut_playground.adaptive.evaluation.metrics import cvvdp_jod
from threedgrut_playground.adaptive.evaluation.reflection_rate import cvvdp_heatmap

BUDGET_SPP = (1, 2, 4, 8, 16)
WARMUPS, FRAMES = 2, 7


def log(text):
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def mode_of(choice, data, device):
    """Option index map [H, W] -> per-pixel mode [H, W]: MC n spp -> n, cone cap c -> -c."""
    values = torch.tensor(
        [float(s) if k == "mc" else -float(c) for k, s, c in zip(data["kind"], data["spp"], data["cap"])], device=device
    )
    return values[choice.to(device)].contiguous()


def timed(b, mode):
    """Median ms of one full-frame launch with ``mode``, and the frame (realization B's random numbers)."""
    e = b.engine
    e.pixel_mode = mode

    def step():
        e.frame_id = SEED_B
        return render_frame(e, b.rays_o, b.rays_d)

    for _ in range(WARMUPS):
        step()
    torch.cuda.synchronize()
    times = []
    for _ in range(FRAMES):
        t0 = time.perf_counter()
        img = step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    e.pixel_mode = None
    return statistics.median(times), to_display(e, img)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--retest", type=Path, default=Path("C:/Users/ville/3dgrut_runs/retest"))
    args = p.parse_args(argv)
    out = args.retest / "single_launch"
    b = Bench(
        SimpleNamespace(
            ckpt=REPO_ROOT / "runs/room_3dgrt/room-0211_151617/ckpt_last.pt",
            ref=Path("C:/Users/ville/3dgrut_runs/cone_equal_time/reference_unclamped_512px_16384spp.pt"),
            out=out,
        )
    )
    e, dev = b.engine, b.device
    configure(e, ConeSetting(max(c for c in torch.load(args.retest / "options.pt")["cap"]), 1.0))
    e.use_gaussian_lod, e.cone_camera_rays = True, False

    data = torch.load(args.retest / "options.pt")
    names, cost, A, B = data["options"], data["cost"].to(dev), data["A"].to(dev), data["B"].to(dev)
    ref, ppd, ids, mc_ms = b.reference, data["ppd"], data["object_ids"].to(dev), data["mc_ms"]
    K, H, W = cost.shape
    heat = torch.stack([cvvdp_heatmap(A[k], ref, ppd) for k in range(K)])
    sel = {"tile 32": torch.stack([tile_mean(h, 32) for h in heat]), "pixel": torch.stack([box(h, 5) for h in heat])}
    lam_scale = H * W / 1000.0

    def model_ms(choice):
        return torch.gather(cost, 0, choice[None])[0].sum().item()

    def composed(choice):
        return torch.gather(B, 0, choice[None, :, :, None].expand(1, H, W, 3))[0]

    rows = []
    for k, nm in enumerate(names):  # uniform options through the same one-launch path
        choice = torch.full((H, W), k, dtype=torch.long, device=dev)
        ms, img = timed(b, mode_of(choice, data, dev))
        jod = cvvdp_jod(img, ref, ppd)
        rows.append(
            {"allocation": f"uniform {nm}", "kind": "uniform", "model_ms": model_ms(choice), "ms": ms, "jod": jod}
        )
        log(f"uniform {nm:<11} model {rows[-1]['model_ms']:7.1f} ms, one launch {ms:7.1f} ms, JOD {jod:.2f}")

    for n in BUDGET_SPP:
        budget = n * mc_ms * 1.02
        picks = {}
        mat = exact_region_choice(heat, cost, ids, budget)
        if mat is not None:
            picks["per material"] = mat.to(dev)
        for rule, err in sel.items():
            best = None
            for lam in LAMBDAS:
                choice = torch.argmin(err + lam * cost * lam_scale, dim=0)
                ms = model_ms(choice)
                if ms <= budget and (best is None or ms > best[0]):
                    best = (ms, choice)
            if best is not None:
                picks[rule] = best[1]
        for rule, choice in picks.items():
            ms, img = timed(b, mode_of(choice, data, dev))
            jod = cvvdp_jod(img, ref, ppd)
            drift = float((img - composed(choice)).abs().max())
            rows.append(
                {
                    "allocation": f"oracle {rule} @ MC {n}",
                    "kind": "oracle",
                    "rule": rule,
                    "budget_spp": n,
                    "model_ms": model_ms(choice),
                    "ms": ms,
                    "jod": jod,
                    "max_abs_vs_composed_B": drift,
                }
            )
            save_png(img, out / f"{rule.replace(' ', '')}_mc{n}.png")
            log(
                f"oracle {rule:<12} @ MC {n:>2}: model {rows[-1]['model_ms']:7.1f} ms, one launch {ms:7.1f} ms "
                f"({ms / rows[-1]['model_ms']:.2f}x), JOD {jod:.2f} (max |diff| vs composed B {drift:.1e})"
            )

    # At measured cost: the best uniform setting that costs no more than each oracle frame
    uni = [r for r in rows if r["kind"] == "uniform"]
    for r in rows:
        if r["kind"] != "oracle":
            continue
        ok = [u for u in uni if u["ms"] <= r["ms"] * 1.02]
        best = max(ok, key=lambda u: u["jod"]) if ok else None
        r["best_uniform_at_measured_cost"] = (
            None if best is None else {"name": best["allocation"], "ms": best["ms"], "jod": best["jod"]}
        )
        if best is not None:
            log(
                f"{r['allocation']:<28} {r['ms']:7.1f} ms JOD {r['jod']:.2f}  vs  {best['allocation']:<18} "
                f"{best['ms']:7.1f} ms JOD {best['jod']:.2f}  ->  {r['jod'] - best['jod']:+.2f}"
            )
    (out / "results.json").write_text(json.dumps({"mc_ms_model": mc_ms, "rows": rows}, indent=2, default=float))
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
