# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests of the budget-sweep CLI: allocator names, ray caps, argument validation, capacity pre-check."""

from pathlib import Path

import pytest

from threedgrut_playground.adaptive.evaluation import sweep as S


def _ok(tmp_path, *extra):
    ckpt = tmp_path / "c.pt"
    ckpt.write_bytes(b"")
    return ["--ckpt", str(ckpt), "--out-dir", str(tmp_path / "o"), *extra]


def test_parse_allocator():
    assert S.parse_allocator("uniform") == ("uniform", None)
    assert S.parse_allocator("plain_uniform") == ("plain_uniform", None)
    assert S.parse_allocator("perceptual_g0.5") == ("perceptual", 0.5)
    assert S.parse_allocator("variance_g1") == ("variance", 1.0)
    assert S.parse_allocator("variance_g2") == ("variance", 2.0)
    for bad in ("perceptual", "variance_g0", "foo_g1", "variance_g2.5", "perceptual_g-1", "uniform_g1"):
        with pytest.raises(ValueError):
            S.parse_allocator(bad)


def test_n_cap_includes_topup():
    assert S.n_cap(4, 3, 192) == 4 + 3 * 192 + 2 * 192 == 964


def test_parser_defaults():
    a = S.build_parser().parse_args(["--ckpt", "c.pt", "--out-dir", "o"])
    assert a.allocators == [
        "uniform",
        "variance_g1",
        "variance_g0.5",
        "perceptual_g1",
        "perceptual_g0.5",
        "plain_uniform",
    ]
    assert (
        a.budgets,
        a.seeds,
        a.pilot_spp,
        a.iterations,
        a.max_per_pixel,
        a.ref_spp,
        a.weight_model,
        a.resolution,
    ) == ([8, 16, 32, 64, 128], [0, 1, 2, 3, 4], 4, 3, 192, 8192, "cielab", (512, 512))


@pytest.mark.parametrize(
    "extra",
    [
        ["--budgets", "4,8"],
        ["--budgets", "8.5"],
        ["--pilot-spp", "2"],
        ["--iterations", "0"],
        ["--allocators", "variance_g1,perceptual_g1"],
        ["--allocators", "uniform,bogus"],
        ["--cameras", "nope"],
        ["--preset", "nope"],
        ["--seeds", "0,63"],
        ["--ref-spp", "4096"],
        ["--defensive-fraction", "1.0"],
        ["--weight-model", "lab"],
    ],
)
def test_main_rejects_bad_args_before_gpu(tmp_path, extra):
    with pytest.raises(SystemExit):
        S.main(_ok(tmp_path, *extra))


def test_tvi_requires_cvvdp(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "cvvdp_available", lambda: False)
    with pytest.raises(SystemExit):
        S.main(_ok(tmp_path, "--weight-model", "tvi"))


def test_weak_reference_can_be_allowed(tmp_path):
    args = S.build_parser().parse_args(_ok(tmp_path, "--ref-spp", "4096", "--allow-weak-reference"))
    with pytest.warns(UserWarning, match="reference"):
        S.validate_args(args, S.build_parser())


def test_main_rejects_out_dir_inside_repo(tmp_path):
    repo = Path(S.__file__).resolve().parents[3]
    with pytest.raises((SystemExit, ValueError)):
        S.main(_ok(tmp_path)[:2] + ["--out-dir", str(repo / "sweep_out")])
    assert not (repo / "sweep_out").exists()


def test_capacity_and_budget_skip_decision():
    """Controller amendment 1: an unclipped region pixel absorbs at most (iterations + 2) * cap new rays, a clipped
    one the cap (top-up widening), every pixel the pilot; budgets above the capacity less a 2% margin are skipped."""
    P, unclipped, clipped = 1000, 80, 20
    cap = S.capacity_rays(P, unclipped, clipped, pilot_spp=4, iterations=3, max_per_pixel=10)
    assert cap == 4 * 1000 + 10 * (5 * 80 + 20) == 8200
    assert S.absorbable_budget(P, unclipped, clipped, 4, 3, 10) == int(8200 * 0.98) == 8036
    kept, skipped = S.split_budgets([5, 8, 9], P, 8036)
    assert kept == [5, 8] and [s["budget_spp"] for s in skipped] == [9]
    assert skipped[0] == {"budget_spp": 9, "budget_rays": 9000, "absorbable_rays": 8036}
    assert S.split_budgets([5, 6], P, 8036) == ([5, 6], [])
    assert S.split_budgets([9], P, 8036)[0] == []
    # garden overview at 512 px with the defaults (~8.7% mesh): 64 spp fits, 128 spp does not
    P = 512 * 512
    absorb = S.absorbable_budget(P, int(0.087 * P), 0, 4, 3, 192)
    assert S.split_budgets([8, 16, 32, 64, 128], P, absorb)[0] == [8, 16, 32, 64]
