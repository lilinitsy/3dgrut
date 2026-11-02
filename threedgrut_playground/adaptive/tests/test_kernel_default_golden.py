# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default-path bit-identity of the playground tracer (D7), plus an informational timing test.

Capture mode (``THREEDGRUT_CAPTURE_GOLDEN=1``) writes the goldens on unmodified code; compare mode checks them
bitwise. Goldens live in ``$THREEDGRUT_TEST_GOLDEN_DIR/kernel_default_v1/``; see the plan's Golden log for the
recapture rule.
"""

import csv
import gc
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import pytest
import torch

from threedgrut_playground.adaptive.tests.kernel_helpers import (
    GOLDEN_CASES,
    GOLDEN_SEED,
    MIN_VISIBLE_MESH_PIXELS,
    REPO_ROOT,
    bit_equal,
    build_engine,
    configure_sphere,
    default_view,
    has_gaussians,
    has_gpu_and_ckpt,
    make_pinhole_rays,
    render_golden_case,
    requires_gpu_ckpt,
    resolve_ckpt,
    restore_engine,
    snapshot_engine,
    visible_mesh_pixel_count,
)


def _golden_dir() -> Path:  # requires_gpu_ckpt already passed, so a missing env var is a configuration error
    root = os.environ.get("THREEDGRUT_TEST_GOLDEN_DIR")
    if not root:
        pytest.fail("THREEDGRUT_TEST_GOLDEN_DIR must be set for GPU test runs")
    root = Path(root).resolve()
    assert root != REPO_ROOT and REPO_ROOT not in root.parents
    return root / "kernel_default_v1"


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load_golden_or_fail(case) -> dict:
    golden_dir = _golden_dir()
    marker = golden_dir / "CAPTURED.json"
    if not marker.exists():  # the only skip path: nothing has been captured yet
        pytest.skip("golden not captured; run Task 1 capture on unmodified code")
    captured = json.loads(marker.read_text(encoding="utf-8"))
    if case.name not in captured.get("cases", []):
        pytest.fail(f"golden case {case.name} is missing from {marker}")
    path = golden_dir / f"{case.name}.pt"
    if not path.exists():
        pytest.fail(f"golden file {path} is missing, but {marker} lists it as captured")
    golden = torch.load(path, map_location="cpu", weights_only=True)
    assert golden["meta"]["ckpt"] == str(resolve_ckpt()), "golden was captured with a different checkpoint"
    return golden


def _record_capture(golden_dir: Path, case_name: str) -> None:
    marker = golden_dir / "CAPTURED.json"
    old = json.loads(marker.read_text(encoding="utf-8")).get("cases", []) if marker.exists() else []
    marker.write_text(
        json.dumps(
            {
                "cases": sorted(set(old) | {case_name}),
                "git_head": _git_head(),
                "date": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


@pytest.fixture(scope="module")
def golden_engine():
    """Module-private engine: the goldens never share state with other test modules."""
    if not has_gpu_and_ckpt():
        pytest.skip("needs CUDA and THREEDGRUT_TEST_CKPT")
    engine = build_engine()
    yield engine
    # Best effort only: the engine holds reference cycles (e.g. VideoRecorder(renderer=self)) and OptiX allocations.
    del engine
    gc.collect()
    torch.cuda.empty_cache()


SHARED_ENGINE_CASES = ["pbr_blend_smooth", "glass_nogauss", "engine_pbr"]


# Defined FIRST in the module: it runs before golden_engine exists, so the session engine is alive first,
# and the private engine is created afterwards (two-engine coexistence, checked before any kernel edit).
@requires_gpu_ckpt
@pytest.mark.parametrize("name", SHARED_ENGINE_CASES)
def test_shared_engine_reproduces_golden(playground_engine, name):
    if os.environ.get("THREEDGRUT_CAPTURE_GOLDEN") == "1":
        pytest.skip("compare-only test")
    case = next(c for c in GOLDEN_CASES if c.name == name)
    golden = _load_golden_or_fail(case)
    out = render_golden_case(playground_engine, case)
    for key, ref in golden["tensors"].items():
        assert bit_equal(out[key], ref), f"shared engine {name}/{key} differs"


@requires_gpu_ckpt
@pytest.mark.parametrize("case", GOLDEN_CASES, ids=lambda c: c.name)
def test_default_path_matches_golden(golden_engine, case):
    if has_gaussians(case):  # guards against a golden that passes with a buried sphere
        visible = visible_mesh_pixel_count(golden_engine, case)
        print(f"{case.name}: {visible} visible sphere pixels")
        assert visible >= MIN_VISIBLE_MESH_PIXELS
    out = render_golden_case(golden_engine, case)
    path = _golden_dir() / f"{case.name}.pt"
    if os.environ.get("THREEDGRUT_CAPTURE_GOLDEN") == "1":
        if path.exists():
            pytest.fail(f"{path} exists; recapture only per the Golden log rule (delete by hand after diagnosis)")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"meta": {"ckpt": str(resolve_ckpt()), "case": case.name}, "tensors": out}, path)
        _record_capture(path.parent, case.name)
        return
    golden = _load_golden_or_fail(case)
    for key, ref in golden["tensors"].items():
        assert bit_equal(out[key], ref), f"{case.name}/{key}: {int((out[key] != ref).sum())} elements differ"


@requires_gpu_ckpt
def test_info_default_path_timing(golden_engine):
    """Informational only, never asserts: median/p10/p90 of the engine's interactive path at 1080p."""
    if os.environ.get("THREEDGRUT_TEST_TIMING") != "1":
        pytest.skip("timing is informational; set THREEDGRUT_TEST_TIMING=1 to run it")
    from threedgrut_playground.engine import OptixPrimitiveTypes

    height, width = 1080, 1920
    engine = golden_engine
    snap = snapshot_engine(engine)
    times_ms = []
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(GOLDEN_SEED)
            configure_sphere(engine, OptixPrimitiveTypes.PBR, "brushed_copper")
            o, d = make_pinhole_rays(*default_view(engine), height=height, width=width)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with torch.no_grad():
                for i in range(25):  # 5 warm-up + 20 timed
                    o_call, d_call = o.clone(), d.clone()  # clone outside the timed region
                    torch.cuda.synchronize()
                    start.record()
                    engine._render_playground_hybrid(o_call, d_call)
                    end.record()
                    torch.cuda.synchronize()
                    if i >= 5:
                        times_ms.append(start.elapsed_time(end))
    finally:
        restore_engine(engine, snap)

    t = torch.tensor(times_ms, dtype=torch.float64)
    median, p10, p90 = (float(torch.quantile(t, q)) for q in (0.5, 0.1, 0.9))
    print(f"default path {height}x{width}: median {median:.3f} ms, p10 {p10:.3f} ms, p90 {p90:.3f} ms")

    csv_path = _golden_dir().parent / "timing.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["label", "date", "git_head", "height", "width", "median_ms", "p10_ms", "p90_ms"])
        writer.writerow(
            [
                os.environ.get("THREEDGRUT_TIMING_LABEL", ""),
                datetime.now().isoformat(timespec="seconds"),
                _git_head(),
                height,
                width,
                f"{median:.4f}",
                f"{p10:.4f}",
                f"{p90:.4f}",
            ]
        )
