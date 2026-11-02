# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI for the adaptive-sampling oracle study: engine -> four-sphere scene -> collect -> evaluate -> summary.

Run from the repository root::

    python scripts/adaptive/run_oracle_study.py --ckpt runs/.../ckpt_last.pt --out-dir C:/Users/me/3dgrut_runs/oracle_v1
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from threedgrut_playground.adaptive.oracle import (
    OracleConfig,
    collect_oracle_data,
    ensure_outside_repo,
    evaluate_oracle,
    oracle_camera,
    setup_oracle_scene,
)

MIN_MESH_COVERAGE = 0.01


def _resolution(text: str) -> tuple[int, int]:
    try:
        w, h = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT such as 512x512, got {text!r}")
    if w < 1 or h < 1:
        raise argparse.ArgumentTypeError(f"resolution must be positive, got {text!r}")
    return w, h


def _float_list(text: str) -> tuple[float, ...]:
    try:
        return tuple(float(v) for v in text.split(",") if v.strip() != "")
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a comma-separated list of numbers, got {text!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Adaptive-sampling oracle study (headroom of sigma-based allocation).")
    p.add_argument("--ckpt", required=True, help="3DGRT checkpoint (.pt)")
    p.add_argument("--out-dir", required=True, help="output directory; must be outside the repository")
    p.add_argument("--resolution", type=_resolution, default=(512, 512), metavar="WxH")
    p.add_argument("--oracle-spp", type=int, default=256)
    p.add_argument("--ref-spp", type=int, default=2048)
    p.add_argument("--pilot-spp", type=int, default=4)
    p.add_argument("--budgets", type=_float_list, default=(8.0, 16.0, 32.0, 64.0))
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--gammas", type=_float_list, default=(1.0, 0.5))
    p.add_argument("--defensive-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-pool-gib", type=float, default=2.0)
    return p


def _has_interface_a() -> bool:
    try:
        from threedgrut_playground.tracer import Tracer
    except Exception:
        return False
    return "ray_pixel_id" in inspect.signature(Tracer.render_playground).parameters


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # the summary contains sigma, gamma, +-, em dash
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)

    # Validation first: nothing heavy is imported or loaded before these two checks pass.
    try:
        out = ensure_outside_repo(args.out_dir)
        cfg = OracleConfig(
            oracle_spp=args.oracle_spp,
            ref_spp=args.ref_spp,
            pilot_spp=args.pilot_spp,
            budgets=tuple(args.budgets),
            gammas=tuple(args.gammas),
            repeats=args.repeats,
            defensive_fraction=args.defensive_fraction,
            seed=args.seed,
            max_pool_gib=args.max_pool_gib,
        )
        cfg.validate()
    except ValueError as exc:
        print(f"error: {exc}")
        return 1
    if not _has_interface_a():
        print("error: interface A (plan A) not implemented")
        return 1

    os.chdir(REPO_ROOT)  # hydra configs resolve relative to the repo root
    from threedgrut_playground.engine import Engine3DGRUT

    width, height = args.resolution
    try:
        engine = Engine3DGRUT(
            gs_object=str(Path(args.ckpt)),
            mesh_assets_folder=str(REPO_ROOT / "threedgrut_playground" / "assets"),
            default_config="apps/colmap_3dgrt.yaml",
        )
        engine.camera_type = "Pinhole"
        camera, focus = oracle_camera(engine, width, height)
        names = setup_oracle_scene(engine, camera, focus)
        print(f"scene: {len(names)} spheres, focus distance {focus:.4f}; collecting ...", flush=True)
        data = collect_oracle_data(engine, camera, cfg)
    except ValueError as exc:
        print(f"error: {exc}")
        return 1

    # collect_oracle_data stores "at" as the point 1 unit along the view axis (the Camera keeps no look-at);
    # record the true O10 look-at and the focus distance so that summary.json is not misleading.
    cam_meta = data.meta["camera"]
    eye = cam_meta["eye"]
    unit_pt = cam_meta["at"]
    cam_meta["at_unit_axis_point"] = unit_pt
    cam_meta["at"] = [e + focus * (u - e) for e, u in zip(eye, unit_pt)]
    cam_meta["focus_distance"] = focus
    data.meta["spheres"] = names

    P = data.height * data.width
    M = int(data.mesh_mask.sum())
    coverage = M / P
    print(
        f"mesh coverage {100.0 * coverage:.2f}% ({M}/{P}); n_nonfinite={data.n_nonfinite}; "
        f"timings={ {k: round(v, 1) for k, v in data.timings.items()} }",
        flush=True,
    )
    if coverage < MIN_MESH_COVERAGE:
        print(f"error: oracle spheres are not visible (mesh coverage {100.0 * coverage:.2f}%)")
        return 1

    report = evaluate_oracle(data, cfg, out, device=engine.device)
    print(report.summary_text)
    print(f"\noutput directory: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
