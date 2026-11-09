# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frame time of cone tracing against one Monte Carlo sample on the cone test ramp.

Places the ramp with ``place_cone_ramp``, then times five configurations (3 warm-ups, median of 20 frames, CUDA
synchronized): MC at 1 spp, cone at max_sigma 0, 0.002, 0.005, and cone at 0.005 with split 0.5. Also reports the env
fraction of the placed ramp at 256x256 and the placement time.

Usage: python scripts/cone/time_cone.py --ckpt <ckpt.pt> --resolution 512 --out <folder outside the repo>
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from threedgrut_playground.adaptive.evaluation.cone_scene import apply_plan, choose_cone_ramp, env_fraction
from threedgrut_playground.adaptive.evaluation.ldr import ensure_outside_repo

WARMUPS = 3
FRAMES = 20
CONFIGS = [
    ("mc_1spp", dict(use_cone_tracing=False, cone_max_sigma=0.005, cone_split=0.0)),
    ("cone_sigma_0", dict(use_cone_tracing=True, cone_max_sigma=0.0, cone_split=0.0)),
    ("cone_sigma_0.002", dict(use_cone_tracing=True, cone_max_sigma=0.002, cone_split=0.0)),
    ("cone_sigma_0.005", dict(use_cone_tracing=True, cone_max_sigma=0.005, cone_split=0.0)),
    ("cone_sigma_0.005_split_0.5", dict(use_cone_tracing=True, cone_max_sigma=0.005, cone_split=0.5)),
]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--out", required=True, type=Path, help="output folder, outside the repo")
    return parser.parse_args(argv)


def frame_ms(engine, camera) -> float:
    rays = engine.raygen(camera, use_spp=False)
    torch.cuda.synchronize()
    start = time.perf_counter()
    engine._render_playground_hybrid(rays.rays_ori.clone(), rays.rays_dir.clone())
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0


def main(argv=None) -> int:
    args = parse_args(argv)
    out = ensure_outside_repo(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from threedgrut_playground.engine import Engine3DGRUT

    engine = Engine3DGRUT(
        gs_object=str(args.ckpt),
        mesh_assets_folder=str(REPO_ROOT / "threedgrut_playground" / "assets"),
        default_config="apps/colmap_3dgrt.yaml",
    )
    torch.cuda.synchronize()
    start = time.perf_counter()
    plan, _ = choose_cone_ramp(engine)  # place_cone_ramp = choose_cone_ramp + apply_plan; the plan is needed below
    apply_plan(engine, plan)
    torch.cuda.synchronize()
    placement_s = time.perf_counter() - start
    preset = plan.camera
    camera = preset.to_camera(args.resolution, args.resolution, device=engine.device)
    engine.camera_fov = preset.fov_deg
    print(f"placement: {placement_s:.1f} s")

    results = {}
    env = None
    for name, values in CONFIGS:
        for key, value in values.items():
            setattr(engine, key, value)
        for _ in range(WARMUPS):  # also rebuilds the BVH after a max_sigma change
            frame_ms(engine, camera)
        if name == "cone_sigma_0.005":
            env = env_fraction(engine, plan, 256)
            print(f"env fraction (256x256, cone sigma 0.005): {env:.4f} (planned bound < 0.1)")
        times = [frame_ms(engine, camera) for _ in range(FRAMES)]
        results[name] = statistics.median(times)
        print(f"{name}: {results[name]:.2f} ms (min {min(times):.2f}, max {max(times):.2f})")

    ratio = results["cone_sigma_0.005"] / results["mc_1spp"]
    print(f"cone 0.005 / MC 1 spp: {ratio:.2f}x (within 2x: {ratio <= 2.0})")
    payload = {
        "ckpt": str(args.ckpt),
        "resolution": args.resolution,
        "warmups": WARMUPS,
        "frames": FRAMES,
        "median_ms": results,
        "cone_005_over_mc": ratio,
        "env_fraction_256": env,
        "placement_seconds": placement_s,
        "camera": {"eye": preset.eye, "at": preset.at, "fov_deg": preset.fov_deg},
    }
    (out / "timing.json").write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
