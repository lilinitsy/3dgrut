# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CSV, budget check, paired intervals, decision rule and plots of ``evaluation/analysis.py`` (CPU only)."""

import csv
import math
from types import SimpleNamespace

import pytest
import torch

from threedgrut_playground.adaptive.evaluation import analysis as A
from threedgrut_playground.adaptive.evaluation.metrics import METRIC_KEYS


def _row(alloc, budget, seed, rays, flip):
    r = dict.fromkeys(A.CSV_COLUMNS, math.nan)
    r.update(
        run_id=f"close_{alloc}_b{budget:g}_s{seed}",
        camera="close",
        weight_model="cielab",
        allocator=alloc,
        budget_spp=budget,
        seed=seed,
        sample_offset=seed * 2**24,
        rays_launched=rays,
        wall_ms=rays / 900.0,
        flip_mesh=flip,
        psnr_mesh=1 / flip,
    )
    return r


def _pr(b, lo, hi, **extra):
    return {
        "camera": "close",
        "weight_model": "cielab",
        "budget_spp": b,
        "allocator": "perceptual_g1",
        "baseline": "variance_g1",
        "metric": "flip_mesh",
        "ci95_low": lo,
        "ci95_high": hi,
        **extra,
    }


def test_columns_contract():
    assert isinstance(A.CSV_COLUMNS, tuple)
    # 69 columns of the brief plus topup_share (controller amendment 1)
    assert len(A.CSV_COLUMNS) == len(set(A.CSV_COLUMNS)) == 70 and set(METRIC_KEYS) <= set(A.CSV_COLUMNS)
    assert "topup_share" in A.CSV_COLUMNS


def test_check_budget():
    assert A.check_budget(1000, 1000) == 0.0
    with pytest.raises(A.BudgetMismatchError, match="1100"):
        A.check_budget(1100, 1000)
    with pytest.raises(A.BudgetMismatchError, match="max-per-pixel"):
        A.check_budget(500, 1000)


def test_csv_log(tmp_path):
    A.CsvLog(tmp_path / "r.csv").write(_row("uniform", 8, 0, 100, 0.2))
    A.CsvLog(tmp_path / "r.csv").write(_row("variance_g1", 8, 0, 100, 0.1))
    with open(tmp_path / "r.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["allocator"] for r in rows] == ["uniform", "variance_g1"] and rows[0]["cvvdp_all"] == "nan"
    with pytest.raises(ValueError):
        A.CsvLog(tmp_path / "r.csv").write({"run_id": "x"})
    with pytest.raises(ValueError):  # a different header must not be appended to
        A.CsvLog(tmp_path / "r.csv", columns=("a", "b")).write({"a": 1, "b": 2})


def test_t95_and_aggregate_ci():
    assert A.t95(5) == pytest.approx(2.776, abs=1e-3) and math.isnan(A.t95(1))
    rows = [_row("uniform", 8, s, 100, 0.2 + 0.1 * s) for s in range(3)] + [_row("uniform", 16, 0, 200, 0.1)]
    (x, y, lo, hi), single = A.aggregate(rows, "rays_launched", "flip_mesh")["uniform"]
    assert (x, y) == (100.0, pytest.approx(0.3)) and hi - y == pytest.approx(A.t95(3) * 0.1 / math.sqrt(3))
    assert single == (200.0, 0.1, 0.1, 0.1)


def test_aggregate_drops_nonfinite():
    rows = [_row("uniform", 8, s, 100, f) for s, f in enumerate((0.2, math.nan, 0.4))]
    ((x, y, lo, hi),) = A.aggregate(rows, "rays_launched", "flip_mesh")["uniform"]
    assert y == pytest.approx(0.3) and lo < y < hi


def test_default_pairings():
    allocs = ["uniform", "variance_g1", "variance_g0.5", "perceptual_g1", "perceptual_g0.5", "plain_uniform"]
    p = A.default_pairings(allocs)
    assert ("perceptual_g1", "variance_g1") in p and ("perceptual_g0.5", "variance_g0.5") in p
    assert ("variance_g1", "uniform") in p and not any("plain_uniform" in pair for pair in p)


def test_paired_differences_ci():
    rows = [_row("variance_g1", 8, s, 100, 0.20 + 0.05 * s) for s in range(5)]
    rows += [_row("perceptual_g1", 8, s, 100, 0.18 + 0.05 * s + 0.002 * (s % 2)) for s in range(5)]
    (p,) = A.paired_differences(rows, "flip_mesh", [("perceptual_g1", "variance_g1")])
    d = torch.tensor([-0.02, -0.018, -0.02, -0.018, -0.02], dtype=torch.float64)
    assert p["n_seeds"] == 5 and p["mean_diff"] == pytest.approx(d.mean().item())
    half = A.t95(5) * d.std().item() / math.sqrt(5)
    assert (p["ci95_low"], p["ci95_high"]) == (
        pytest.approx(d.mean().item() - half),
        pytest.approx(d.mean().item() + half),
    )
    assert set(p) == set(A.PAIRED_COLUMNS)


def test_decide_rule():
    wins3 = [_pr(8, -2, -1), _pr(16, -2, -1), _pr(32, -2, -1), _pr(64, -1, 1), _pr(128, -1, 1)]
    assert A.decide(wins3, "flip_mesh", "lower")[0]["verdict"] == "beats"
    assert A.decide(wins3[:2] + wins3[3:], "flip_mesh", "lower")[0]["verdict"] == "inconclusive"  # 2 of 4 < 60%
    assert A.decide(wins3[:4] + [_pr(128, 1, 2)], "flip_mesh", "lower")[0]["verdict"] == "inconclusive"  # one loss


def test_decide_higher_loses_and_nan():
    wins = [_pr(b, 1, 2) for b in (8, 16, 32)] + [_pr(64, -1, 1), _pr(128, -1, 1)]
    assert A.decide(wins, "flip_mesh", "higher")[0]["verdict"] == "beats"
    (d,) = A.decide(wins, "flip_mesh", "lower")
    assert d["verdict"] == "loses" and (d["wins"], d["losses"]) == (0, 3)
    nan = math.nan  # all-nan intervals (e.g. cvvdp metric without pycvvdp) are inconclusive, not an error
    (d,) = A.decide([_pr(b, nan, nan, metric="cvvdp_mesh") for b in (8, 16, 32)], "cvvdp_mesh", "higher")
    assert d["verdict"] == "inconclusive" and (d["wins"], d["losses"], d["n_budgets"]) == (0, 0, 3)


def test_decide_topup_caveat_flag():
    ps = [_pr(8, -2, -1, max_topup_share=0.02), _pr(16, -2, -1, max_topup_share=0.25), _pr(32, -2, -1)]
    (d,) = A.decide(ps, "flip_mesh", "lower")
    assert d["verdict"] == "beats" and d["max_topup_share"] == 0.25 and d["topup_caveat"] is True
    (d,) = A.decide([_pr(8, -2, -1, max_topup_share=0.05)], "flip_mesh", "lower")
    assert d["topup_caveat"] is False
    (d,) = A.decide([_pr(8, -2, -1)], "flip_mesh", "lower")  # rows do not carry it
    assert math.isnan(d["max_topup_share"]) and d["topup_caveat"] is False
    # raw rows of the pair supply the share when the paired rows lack it
    raw = [_row("perceptual_g1", 8, 0, 100, 0.1), _row("variance_g1", 8, 0, 100, 0.2), _row("uniform", 8, 0, 100, 0.3)]
    raw[0]["topup_share"], raw[2]["topup_share"] = 0.31, 0.9
    (d,) = A.decide([_pr(8, -2, -1)], "flip_mesh", "lower", rows=raw)
    assert d["max_topup_share"] == pytest.approx(0.31) and d["topup_caveat"] is True and A.TOPUP_SHARE_WARN == 0.10


def test_caveats_are_nonempty_strings():
    assert isinstance(A.CAVEATS, tuple) and len(A.CAVEATS) >= 4
    assert all(isinstance(c, str) and c.strip() for c in A.CAVEATS)
    text = " ".join(A.CAVEATS)
    assert "Owen" in text and "xfail" in text and "ELIGIBILITY_NOTE" in text and "topup_share" in text
    assert "60%" in A.DECISION_RULE and "flip_mesh" in A.DECISION_RULE


def test_convergence_slope():
    n = [8.0, 32.0, 128.0]
    assert A.convergence_slope(n, [2.0 / x + 1e-4 for x in n], 1e-4) == pytest.approx(-1.0)
    assert math.isnan(A.convergence_slope([8.0], [1.0], 0.0))


def test_assemble_row_uses_rays_launched():
    counts = torch.tensor([[3, 1], [1, 3]])
    mask = torch.tensor([[True, False], [False, True]])
    res = SimpleNamespace(
        counts=counts,
        mesh_mask=mask,
        timings={"rays_launched": 8, "rays_unspent": 0, "n_nonfinite": 0, "gpu_ms": 1.0},
    )
    base = dict.fromkeys(A.CSV_COLUMNS[:18] + ("ref_key", "ref_spp", "ref_spp_nonmesh"), 0)
    row = A.assemble_row(base, res, 8, 0.01, 2.0, dict.fromkeys(METRIC_KEYS, 0.0))
    assert set(row) == set(A.CSV_COLUMNS) and math.isnan(row["topup_rays"]) and math.isnan(row["topup_share"])
    assert (row["rays_launched"], row["rays_mesh"], row["rays_nonmesh"], row["max_px_count"]) == (8, 6, 2, 3)
    with pytest.raises(A.BudgetMismatchError):
        A.assemble_row(base, res, 16, 0.01, 2.0, dict.fromkeys(METRIC_KEYS, 0.0))


def test_assemble_row_topup_share():
    res = SimpleNamespace(
        counts=torch.full((2, 2), 2),
        mesh_mask=torch.ones(2, 2, dtype=torch.bool),
        timings={"rays_launched": 8, "topup_rays": 2, "n_launches": 3, "gpu_ms": 1.0},
    )
    base = dict.fromkeys(A.CSV_COLUMNS[:18] + ("ref_key", "ref_spp", "ref_spp_nonmesh"), 0)
    row = A.assemble_row(base, res, 8, 0.01, 2.0, dict.fromkeys(METRIC_KEYS, 0.0))
    assert row["topup_share"] == pytest.approx(0.25) and row["topup_rays"] == 2 and row["n_launches"] == 3
    assert row["mesh_fraction"] == 1.0 and row["budget_rel_err"] == 0.0


def test_plot_curves(tmp_path):
    pytest.importorskip("matplotlib")
    rows = [
        _row(a, b, s, 100 * b, 0.3 / b + 0.01 * s)
        for a in ("uniform", "variance_g1", "perceptual_g1")
        for b in (8, 16)
        for s in (0, 1)
    ]
    paired = A.paired_differences(rows, "flip_mesh", A.default_pairings(["uniform", "variance_g1", "perceptual_g1"]))
    paths = A.plot_curves(rows, paired, tmp_path, "close", "flip_mesh")
    names = {p.name for p in paths}
    assert {
        "close_flip_mesh_vs_rays.png",
        "close_flip_mesh_vs_wall_ms.png",
        "close_paired_flip_mesh_vs_budget.png",
    } <= names
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_check_budget_within_tolerance_nan_and_large_counts():
    assert A.check_budget(1005, 1000) == pytest.approx(0.005)
    with pytest.raises(A.BudgetMismatchError):
        A.check_budget(float("nan"), 1000)
    with pytest.raises(A.BudgetMismatchError):
        A.check_budget(10, 0)
    with pytest.raises(A.BudgetMismatchError, match="137710400"):
        A.check_budget(137_710_400, 132_710_400)
