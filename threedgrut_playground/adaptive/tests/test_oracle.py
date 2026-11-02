# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import inspect
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from kaolin.render.camera import (
    Camera,
    generate_centered_pixel_coords,
    generate_pinhole_rays,
)

from threedgrut_playground.adaptive.oracle import (
    CIELAB_KAPPA,
    HEADROOM_COLUMNS,
    KNOWN_LIMITATIONS,
    ORACLE_SAMPLE_OFFSET,
    REF_SAMPLE_OFFSET,
    REPO_ROOT,
    RULES,
    RUN_COLUMNS,
    WINSOR_DELTA,
    OracleConfig,
    OracleData,
    allocate,
    analytic_mse,
    bootstrap_means,
    class_blur2d,
    collect_oracle_data,
    display_clip_weight,
    ensure_outside_repo,
    evaluate_oracle,
    flip_map,
    gaussian_blur2d,
    lstar_slope,
    luminance,
    luminance_adaptation_weight,
    oracle_camera,
    oracle_rays,
    pack_rows,
    perceptual_weight,
    pilot_stats,
    pixel_jitter,
    psnr_from_mse,
    setup_oracle_scene,
    smooth_sigma_like_b,
    srgb_eotf,
    trace_samples,
)


def test_out_dir_inside_repo_is_refused(tmp_path, monkeypatch):
    bad_paths = [REPO_ROOT, REPO_ROOT / "runs" / "oracle"]
    if os.name == "nt":
        bad_paths.append(Path(str(REPO_ROOT).upper()) / "x")
    for bad in bad_paths:
        with pytest.raises(ValueError, match="inside the repository"):
            ensure_outside_repo(bad)
    monkeypatch.chdir(REPO_ROOT)
    with pytest.raises(ValueError, match="inside the repository"):
        ensure_outside_repo("oracle_out")  # relative -> resolved against CWD
    assert ensure_outside_repo(tmp_path / "o") == (tmp_path / "o").resolve()


def test_psnr_from_mse():
    assert psnr_from_mse(0.01) == pytest.approx(20.0) and psnr_from_mse(0.0) == pytest.approx(120.0)


def test_flip_map_identity_and_shape():
    g = torch.Generator().manual_seed(0)
    ref = torch.rand(32, 48, 3, generator=g)
    m0 = flip_map(ref, ref, ppd=512 / 60)
    assert m0.shape == (32, 48) and m0.dtype == torch.float32 and m0.abs().max().item() < 1e-6
    noisy = (ref + 0.05 * torch.randn(32, 48, 3, generator=g)).clamp(0, 1)
    lo, hi = flip_map(noisy, ref, ppd=8.0).mean().item(), flip_map(noisy, ref, ppd=67.0).mean().item()
    assert lo > 0.01 and lo != pytest.approx(hi, rel=1e-3)  # ppd actually reaches FLIP


def _grey(v, shape=(16, 16)):
    return torch.full((*shape, 3), float(v), dtype=torch.float64)


def test_blur_and_class_blur_match_plan_b():
    x = torch.full((7, 5), 3.0, dtype=torch.float64)
    torch.testing.assert_close(gaussian_blur2d(x, 1.5), x)
    z = torch.zeros(21, 21, dtype=torch.float64)
    z[10, 10] = 1.0
    b = gaussian_blur2d(z, 1.0)
    assert b[10, 13] > 0 and b[10, 14] == 0 and abs(b.sum().item() - 1.0) < 1e-9
    cls = torch.zeros(8, 8, dtype=torch.bool)
    cls[:, :4] = True
    x = torch.where(cls, 5.0, 1.0).double()
    torch.testing.assert_close(class_blur2d(x, cls, 2.0)[0], x)


def test_srgb_eotf_and_lstar_slope():
    v = torch.tensor([0.0, 0.04045, 0.5, 1.0, -0.2, 1.7], dtype=torch.float64)
    torch.testing.assert_close(
        srgb_eotf(v), torch.tensor([0.0, 0.04045 / 12.92, 0.21404114, 1.0, 0.0, 1.0], dtype=torch.float64)
    )
    assert lstar_slope(torch.tensor(0.0, dtype=torch.float64)).item() == pytest.approx(CIELAB_KAPPA / 12.92)


# Y' = 0, 0.1, ..., 1 (plan B's pinned values)
LSTAR_W = [0.4886, 1.6657, 1.3649, 1.1957, 1.0827, 1.0, 0.9359, 0.8842, 0.8413, 0.8049, 0.7734]


def test_luminance_weight_is_the_lstar_slope():
    assert [luminance_adaptation_weight(_grey(y / 10))[0, 0].item() for y in range(11)] == pytest.approx(
        LSTAR_W, abs=1e-3
    )
    ys = torch.linspace(0, 1, 2001, dtype=torch.float64)
    w = luminance_adaptation_weight(ys[None, :, None].expand(1, 2001, 3))[0]
    assert w.min().item() == pytest.approx(0.4886, abs=1e-3) and w.max().item() == pytest.approx(1.7003, abs=2e-3)
    assert abs(ys[w.argmax()].item() - 0.0922) < 1e-3


def test_luminance_weight_is_pointwise():  # D9 term has no spatial pooling
    rgb = _grey(0.9, (64, 64))
    rgb[28:36, 28:36] = 0.2
    w = luminance_adaptation_weight(rgb)
    assert w[32, 32].item() == pytest.approx(1.3649, abs=1e-3) and w[0, 0].item() == pytest.approx(0.8049, abs=1e-3)


def test_display_clip_weight_curve():
    vals = {m: display_clip_weight(_grey(m), None)[0, 0].item() for m in (0.5, 1.0, 1.025, 1.05, 1.5)}
    assert vals == pytest.approx({0.5: 1.0, 1.0: 1.0, 1.025: 0.5, 1.05: 0.0, 1.5: 0.0}, abs=1e-6)
    partly = _grey(1.5).clone()
    partly[..., 2] = 0.3
    assert display_clip_weight(partly, None)[0, 0].item() == 1.0
    assert display_clip_weight(_grey(1.1), torch.full((16, 16), 0.01))[0, 0].item() == pytest.approx(0.5, abs=1e-6)


def test_weights_sanitize_nonfinite_inputs():
    for bad in (float("nan"), -1.0):
        w = perceptual_weight(_grey(bad), None)
        assert torch.isfinite(w).all() and w[0, 0].item() == pytest.approx(0.4886, rel=1e-3)
    assert (perceptual_weight(_grey(1.5), None) == 0).all()


B4 = lambda m: torch.full((m,), 4, dtype=torch.int64)
C = lambda m, c=1000: torch.full((m,), c, dtype=torch.int64)
D = lambda *v: torch.tensor(v, dtype=torch.float64)
ONES = lambda m: torch.ones(m, dtype=torch.float64)


def test_allocate_uniform_equal_counts():
    n, u = allocate("uniform", torch.rand(10, dtype=torch.float64), ONES(10), ONES(10), B4(10), C(10), 160, 0.15)
    assert n.tolist() == [16] * 10 and u == 0


@pytest.mark.parametrize("rule", RULES)
def test_allocate_conserves_total_and_bounds(rule):
    s = torch.linspace(0.01, 1, 50, dtype=torch.float64)
    w = torch.linspace(2, 0.5, 50, dtype=torch.float64)
    n, u = allocate(rule, s, w, ONES(50), B4(50), C(50, 40), 800, 0.15)
    assert n.sum() == 800 and u == 0 and (n >= 4).all() and (n <= 40).all() and n.dtype == torch.int64


def test_allocate_defensive_share():  # extra 160, d = 24 -> +2.4 each; Neyman 136 all to pixel 9
    n, _ = allocate("sigma", D(*([0.0] * 9 + [1.0])), ONES(10), ONES(10), B4(10), C(10), 200, 0.15)
    assert set(n[:9].tolist()) <= {6, 7} and n[9].item() in (142, 143) and n.sum() == 200


def test_allocate_neyman_ratio_and_gamma():
    big = C(2, 10**6)
    assert allocate("sigma", D(1, 3), D(1, 1), ONES(2), B4(2), big, 4000, 0.0)[0].tolist() == [1000, 3000]
    assert allocate("sigma_sqrt_w", D(1, 1), D(1, 9), ONES(2), B4(2), big, 4000, 0.0)[0].tolist() == [1000, 3000]
    assert allocate("sigma", D(1, 9), D(1, 1), ONES(2), B4(2), big, 4000, 0.0, gamma=0.5)[0].tolist() == [1000, 3000]
    n = allocate("sigma_sqrt_w", D(1, 1), D(1, 81), ONES(2), B4(2), big, 4000, 0.0, gamma=0.5)[0]
    assert n.tolist() == [1000, 3000]


def test_allocate_caps_redistribute():  # pixel 0 capped at 20; the excess flows through the 2-round top-up
    cap = torch.tensor([20, 1000, 1000, 1000])
    n, u = allocate("sigma", D(10, 1, 1, 1), ONES(4), ONES(4), B4(4), cap, 400, 0.0)
    assert n[0] == 20 and n.sum() == 400 and u == 0 and all(125 <= v <= 128 for v in n[1:].tolist())


def test_allocate_all_capped_reports_unspent():
    n, u = allocate("sigma", D(1, 2, 3, 4), ONES(4), ONES(4), B4(4), C(4, 10), 50, 0.15)
    assert n.tolist() == [10] * 4 and u == 10


def test_allocate_zero_sigma_falls_back_to_uniform():
    for rule in ("sigma", "sigma_sqrt_w"):
        n = allocate(rule, torch.zeros(10, dtype=torch.float64), ONES(10), ONES(10), B4(10), C(10), 160, 0.15)[0]
        assert n.tolist() == [16] * 10


def test_allocate_clipped_pixels_get_no_defensive_share():
    clipped = D(1, 1, 0, 0)  # pixels 2, 3 are above display white
    w = D(1, 1, 0, 0)
    n_unclipped = allocate("sigma_sqrt_w", ONES(4), w, ONES(4), B4(4), C(4), 64, 0.15)[0]
    assert n_unclipped.tolist() == [26, 26, 6, 6]
    n_clipped = allocate("sigma_sqrt_w", ONES(4), w, clipped, B4(4), C(4), 64, 0.15)[0]
    assert n_clipped.tolist() == [28, 28, 4, 4]
    for rule in ("uniform", "sigma"):  # w = 1 keeps them Neyman-eligible, but no defensive share
        assert allocate(rule, ONES(4), ONES(4), clipped, B4(4), C(4), 64, 0.15)[0].tolist() == [18, 18, 14, 14]
    z = torch.zeros(4, dtype=torch.float64)
    n, u = allocate("sigma_sqrt_w", ONES(4), z, z, B4(4), C(4), 64, 0.15)
    assert n.tolist() == [16] * 4 and u == 0  # nothing eligible or unclipped: top-up fallback spends it


def test_allocate_rejects_bad_inputs():
    with pytest.raises(ValueError, match="pilot"):
        allocate("uniform", D(1), D(1), D(1), B4(1), C(1), 3, 0.15)
    with pytest.raises(ValueError, match="unknown rule"):
        allocate("foo", D(1), D(1), D(1), B4(1), C(1), 8, 0.15)
    with pytest.raises(ValueError, match="gamma"):
        allocate("sigma", D(1), D(1), D(1), B4(1), C(1), 8, 0.15, gamma=-1.0)


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def test_pilot_stats_ignores_nonfinite():
    s = torch.tensor([[1.0, float("nan"), 3.0, 5.0]]).reshape(1, 4, 1).expand(1, 4, 3).clone()
    mean, sig, nv = pilot_stats(s)
    assert nv.tolist() == [3] and mean[0].tolist() == pytest.approx([3.0] * 3) and sig[0].item() == pytest.approx(2.0)
    mean_w, sig_w, _ = pilot_stats(s, winsor_delta=WINSOR_DELTA)  # samples -> 1, 1.05, 1.05
    assert mean_w[0].tolist() == pytest.approx([3.0] * 3) and sig_w[0].item() == pytest.approx(0.0288675, rel=1e-5)
    assert mean.dtype == sig.dtype == torch.float64 and nv.dtype == torch.int64


def test_pilot_stats_winsorizes_both_ends():
    s = torch.tensor([0.2, 3.0, 0.4, -1.0]).reshape(1, 4, 1).expand(1, 4, 3).clone()
    assert pilot_stats(s)[1][0].item() == pytest.approx(1.6842407, rel=1e-6)
    assert pilot_stats(s, winsor_delta=0.05)[1][0].item() == pytest.approx(0.4552929, rel=1e-6)  # [0.2, 1.05, 0.4, 0]


def test_pilot_stats_small_counts():
    s = torch.ones(3, 4, 3)
    s[0] = float("nan")  # no valid sample
    s[1, 1:] = float("inf")  # one valid sample
    mean, sig, nv = pilot_stats(s)
    assert nv.tolist() == [0, 1, 4]
    assert mean[0].tolist() == [0.0] * 3 and mean[1].tolist() == pytest.approx([1.0] * 3)
    assert sig.tolist() == [0.0, 0.0, 0.0]


def test_smooth_sigma_like_b_is_class_aware_and_fills_gaps():
    mask = torch.zeros(8, 8, dtype=torch.bool)
    mask[:, :4] = True
    sig = torch.where(mask, 1.0, 0.1).double()
    n = torch.full((8, 8), 4)
    n[3, 1] = 1
    sig[3, 1] = 0.0
    out = smooth_sigma_like_b(sig, n, mask)
    torch.testing.assert_close(
        out, torch.where(mask, 1.0, 0.1).double(), atol=1e-12, rtol=0
    )  # gap filled, no leak, no c4
    n2 = torch.where(mask, 1, 4)  # no valid mesh pixel -> median of all valid
    assert torch.allclose(smooth_sigma_like_b(sig, n2, mask)[mask], torch.tensor(0.1, dtype=torch.float64))


def test_smooth_sigma_like_b_median_of_same_class_and_zero_fallback():
    mask = torch.zeros(9, 24, dtype=torch.bool)
    mask[:, :3] = True
    sig = torch.zeros(9, 24, dtype=torch.float64)
    n = torch.ones(9, 24, dtype=torch.int64)
    sig[:, :3], n[:, :3] = 0.5, 4  # valid mesh columns
    sig[:, 20], n[:, 20] = 0.2, 4  # valid off-mesh columns, far from the isolated pixel
    sig[:, 21], n[:, 21] = 0.4, 4
    sig[:, 22], n[:, 22] = 0.9, 4
    sig[4, 10], n[4, 10] = 0.0, 1  # off-mesh pixel with zero support at sigma_px = 0.5 (taps reach 2 px)
    out = smooth_sigma_like_b(sig, n, mask, sigma_px=0.5)
    assert out[4, 10].item() == pytest.approx(0.4)  # median of valid off-mesh sigma {0.2, 0.4, 0.9}, not the mesh 0.5
    mask2 = mask.clone()
    mask2[4, 10] = True  # mesh pixel, valid mesh pixels exist but none within reach -> same-class median 0.5
    assert smooth_sigma_like_b(sig, n, mask2, sigma_px=0.5)[4, 10].item() == pytest.approx(0.5)
    assert (smooth_sigma_like_b(sig, torch.ones(9, 24, dtype=torch.int64), mask) == 0).all()  # nothing valid -> 0


def test_analytic_mse_known_value():
    assert psnr_from_mse(
        analytic_mse(torch.full((5, 3), 0.04, dtype=torch.float64), torch.full((5,), 4))
    ) == pytest.approx(20.0)


def test_bootstrap_constant_pool_is_exact():
    pool = torch.full((3, 16, 3), 0.7)
    pilot = torch.full((3, 4, 3), 0.7)
    est = bootstrap_means(pool, pilot, torch.tensor([[4, 10, 20], [20, 20, 4]]), _gen())
    torch.testing.assert_close(est, torch.full((2, 3, 3), 0.7))


def test_bootstrap_full_draw_equals_full_mean():
    g = _gen(1)
    pool = torch.rand(5, 16, 3, generator=g)
    pilot = torch.rand(5, 4, 3, generator=g)
    est = bootstrap_means(pool, pilot, torch.full((1, 5), 20), _gen(7))[0]
    torch.testing.assert_close(est, (pool.sum(1) + pilot.sum(1)) / 20, atol=1e-5, rtol=0)


def test_bootstrap_mse_matches_sigma2_over_n():  # independent truth (mean 0, sigma 1): E err^2 = 1/n
    g = _gen(2)
    M, S = 4000, 64
    pool = torch.randn(M, S, 3, generator=g)
    pilot = torch.randn(M, 4, 3, generator=g)
    n = torch.where(torch.arange(M) % 2 == 0, 8, 32)
    est = bootstrap_means(pool, pilot, n[None], _gen(3))[0]
    for nn in (8, 32):
        assert (est[n == nn] ** 2).mean().item() == pytest.approx(1 / nn, rel=0.08)


def test_bootstrap_skips_nonfinite_samples():
    pool = torch.ones(2, 8, 3)
    pool[0, [0, 5]] = float("nan")
    pool[1, 3, 1] = float("inf")
    pilot = torch.ones(2, 4, 3)
    est = bootstrap_means(pool, pilot, torch.tensor([[10, 11]]), _gen())
    assert torch.isfinite(est).all() and torch.allclose(est, torch.ones_like(est))


def test_bootstrap_rejects_overdraw():
    pool = torch.ones(2, 8, 3)
    pool[0, 0] = float("nan")
    with pytest.raises(ValueError, match="pool"):
        bootstrap_means(pool, torch.ones(2, 4, 3), torch.tensor([[12, 12]]), _gen())  # pixel 0 has only 7 valid
    with pytest.raises(ValueError, match="pool"):
        bootstrap_means(pool, torch.ones(2, 4, 3), torch.tensor([[3, 4]]), _gen())  # below the pilot count


def test_bootstrap_reproducible_by_seed():
    g = _gen(4)
    pool = torch.rand(50, 32, 3, generator=g)
    pilot = torch.rand(50, 4, 3, generator=g)
    n = torch.full((1, 50), 12)
    a, b, c = (bootstrap_means(pool, pilot, n, _gen(s)) for s in (5, 5, 6))
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_bootstrap_paired_across_rules_and_without_replacement():
    pool = torch.arange(16, dtype=torch.float32).reshape(1, 16, 1).expand(1, 16, 3).clone()  # distinct samples
    pilot = torch.zeros(1, 4, 3)
    counts = torch.tensor([[5], [9], [20]])  # k=0 uses 1 pool draw, k=1 uses 5, k=2 uses all 16
    est = bootstrap_means(pool, pilot, counts, _gen(11))
    assert est[2, 0, 0].item() == pytest.approx(120.0 / 20)  # all 16 distinct samples exactly once
    # nested prefixes of one permutation: sums are recoverable and the smaller draw is contained in the larger
    s1, s5 = est[0, 0, 0].item() * 5, est[1, 0, 0].item() * 9
    assert (
        0 <= s1 <= 15
        and s1 == pytest.approx(round(s1), abs=1e-4)
        and s5 >= s1 - 1e-4
        and s5 == pytest.approx(round(s5), abs=1e-4)
    )


# ---------------------------------------------------------------------------------------------------------------------
# evaluate_oracle (Task 5)
# ---------------------------------------------------------------------------------------------------------------------


def _synthetic_data(
    h=32, w=32, sigmas=(0.02, 0.3), mean=0.4, bg=0.5, pool_spp=64, pilot_spp=4, seed=0, mesh=True, pilot_swap=False
):
    """Mesh = columns < w//2; rows < h//2 use sigmas[0], the rest sigmas[1]. Reference moments are empirical (4096 draws).

    With ``pilot_swap`` the pilot samples use the swapped sigma blocks (large sigma where the pool/reference truth is
    small and vice versa), so pilot-sigma allocation is deliberately wrong while the oracle one is right.
    """
    g = torch.Generator().manual_seed(seed)
    P, ref_draws = h * w, 4096
    mask = torch.zeros(h, w, dtype=torch.bool)
    if mesh:
        mask[:, : w // 2] = True
    idx = torch.nonzero(mask.flatten()).flatten()
    M = idx.numel()
    sig = torch.where(idx // w < h // 2, sigmas[0], sigmas[1]).float()[:, None, None]

    psig = torch.where(idx // w < h // 2, sigmas[1], sigmas[0]).float()[:, None, None] if pilot_swap else sig

    def draw(n, s=sig):
        return mean + s * torch.randn(M, n, 3, generator=g)

    pilot = torch.full((P, pilot_spp, 3), bg)
    pilot[idx] = draw(pilot_spp, psig)
    pool = draw(pool_spp)
    s, ss = torch.zeros(M, 3, dtype=torch.float64), torch.zeros(M, 3, dtype=torch.float64)
    ls, lss, ws, wss = (torch.zeros(M, dtype=torch.float64) for _ in range(4))
    for c0 in range(0, M, 128):
        x = (mean + sig[c0 : c0 + 128] * torch.randn(min(128, M - c0), ref_draws, 3, generator=g)).double()
        s[c0 : c0 + 128], ss[c0 : c0 + 128] = x.sum(1), (x * x).sum(1)
        lum, wl = luminance(x), luminance(x.clamp(0.0, 1.0 + WINSOR_DELTA))
        ls[c0 : c0 + 128], lss[c0 : c0 + 128] = lum.sum(1), (lum * lum).sum(1)
        ws[c0 : c0 + 128], wss[c0 : c0 + 128] = wl.sum(1), (wl * wl).sum(1)
    return OracleData(
        height=h, width=w, ppd=512 / 60, mesh_mask=mask, pilot=pilot, pool=pool,
        ref_sum=s, ref_sumsq=ss, ref_luma_sum=ls, ref_luma_sumsq=lss, ref_wluma_sum=ws, ref_wluma_sumsq=wss,
        ref_count=torch.full((M,), ref_draws, dtype=torch.int64),
        n_nonfinite={"pilot": 0, "pool": 0, "ref": 0}, timings={}, meta={},
    )  # fmt: skip


CFG = OracleConfig(oracle_spp=64, ref_spp=1024, budgets=(8, 16), repeats=3)  # gammas default (1.0, 0.5)
KEY = lambda r: (r["rule"], r["sigma_source"], r["gamma"], r["budget_spp_mesh"])


def _row(rep, rule, src, b, g=1.0):
    return next(r for r in rep.headroom if KEY(r) == (rule, src, g, b))


def test_oracle_data_accessors():
    d = _synthetic_data()
    idx = d.mesh_index()
    assert idx.dtype == torch.int64 and idx.numel() == 512 and idx[0] == 0 and idx[16] == 32  # row-major
    assert d.ref_mean().shape == (512, 3) and d.ref_var_rgb().shape == (512, 3)
    top = d.ref_sigma_luma()[:256]  # rows < 16 use sigma 0.02; luma of 3 iid channels has std sigma * |w|_2
    assert top.mean().item() == pytest.approx(0.02 * (0.2126**2 + 0.7152**2 + 0.0722**2) ** 0.5, rel=0.03)
    assert d.ref_sigma_luma(winsorized=False).shape == (512,)


def test_evaluate_outputs_and_schema(tmp_path):
    rep = evaluate_oracle(_synthetic_data(), CFG, tmp_path)
    with open(tmp_path / "oracle_runs.csv", newline="") as f:
        rows = list(csv.DictReader(f))
        assert tuple(rows[0]) == RUN_COLUMNS and len(rows) == 9 * 2 * 3
    assert {float(r["gamma"]) for r in rows} == {0.0, 1.0, 0.5}
    with open(tmp_path / "oracle_headroom.csv", newline="") as f:
        assert tuple(next(csv.DictReader(f))) == HEADROOM_COLUMNS
    for p in (
        "summary.json",
        "reference_ldr.png",
        "mesh_mask.png",
        "plots/psnr_mesh_vs_budget.png",
        "plots/flip_mesh_vs_budget.png",
        "plots/dpsnr_mesh_vs_budget.png",
        "plots/dflip_mesh_vs_budget.png",
    ):
        assert (tmp_path / p).is_file()
    assert all(int(r["rays_mesh"]) == round(float(r["budget_spp_mesh"]) * 512) for r in rows)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["known_limitations"] == list(KNOWN_LIMITATIONS) and len(KNOWN_LIMITATIONS) == 7
    assert summary["ppd"] == pytest.approx(512 / 60)
    t = rep.summary_text
    assert "contrast_masking excluded" in t and t.index("Known limitations") < t.index("Numbers to look at")
    assert all(s in t for s in KNOWN_LIMITATIONS) and "upper bound" not in t and "achievable" not in t
    raw = (tmp_path / "summary.json").read_text()
    assert "upper bound" not in raw and "achievable" not in raw
    assert {Path(p) for p in rep.files} >= {tmp_path / "oracle_runs.csv", tmp_path / "summary.json"}


def test_evaluate_finds_headroom_on_heteroscedastic_scene(tmp_path):
    rep = evaluate_oracle(_synthetic_data(), CFG, tmp_path)
    for rule in ("sigma", "sigma_sqrt_w"):
        r = _row(rep, rule, "oracle", 16)
        assert r["dpsnr_analytic_mesh"] > 1.5 and r["dpsnr_mesh_mean"] > 1.0 and r["dflip_mesh_mean"] < 0
        assert 1.0 < r["ray_saving_analytic_image"] < r["ray_saving_analytic"]  # pilot rays dilute image savings
        assert (
            0 < _row(rep, rule, "oracle", 16, g=0.5)["dpsnr_analytic_mesh"] < r["dpsnr_analytic_mesh"]
        )  # gamma=1 is MSE-optimal
        assert math.isfinite(_row(rep, rule, "pilot", 16)["estimation_loss_db"])
        assert math.isnan(_row(rep, rule, "oracle", 16)["estimation_loss_db"])
    u = _row(rep, "uniform", "none", 16, g=0.0)
    assert u["dpsnr_mesh_mean"] == 0.0 and u["dflip_mesh_mean"] == 0.0 and u["dpsnr_analytic_mesh"] == 0.0


def test_evaluate_empirical_matches_analytic(tmp_path):
    rep = evaluate_oracle(_synthetic_data(), CFG, tmp_path)
    for key in (("uniform", "none", 0.0, 16), ("sigma", "oracle", 1.0, 16), ("sigma_sqrt_w", "oracle", 0.5, 16)):
        runs = [r for r in rep.runs if KEY(r) == key]
        emp = sum(r["psnr_mesh"] for r in runs) / len(runs)
        assert abs(emp - runs[0]["psnr_analytic_mesh"]) < 0.4
    r8 = _row(rep, "sigma", "oracle", 8)  # empirical and analytic ray savings agree where both are defined
    assert r8["ray_saving_empirical"] == pytest.approx(r8["ray_saving_analytic"], rel=0.10)


def test_evaluate_pilot_path_uses_pilot_sigma_and_oracle_error(tmp_path):
    # pilot sees the swapped sigma blocks: pilot-sigma allocation is wrong, oracle-sigma allocation is right, and every
    # row's analytic error is scored against the oracle var_rgb (not the pilot variance). Caveat: the swapped pilot
    # samples also enter the empirical estimates (4 samples with the wrong variance), so empirical dPSNR is polluted
    # here; only the analytic columns and |estimation loss| are asserted.
    rep = evaluate_oracle(_synthetic_data(pilot_swap=True), CFG, tmp_path)
    for rule in ("sigma", "sigma_sqrt_w"):
        o, p = _row(rep, rule, "oracle", 16), _row(rep, rule, "pilot", 16)
        assert o["dpsnr_analytic_mesh"] > 1.5  # observed 2.04
        assert p["dpsnr_analytic_mesh"] < -1.0  # observed -4.13: pilot allocation is worse than uniform
        assert abs(p["estimation_loss_db"]) > 1.0  # observed -3.84; 0.0 when the pilot path uses oracle sigma
        assert p["estimation_loss_db"] == pytest.approx(o["dpsnr_mesh_mean"] - p["dpsnr_mesh_mean"])


def test_evaluate_homoscedastic_has_no_oracle_headroom(tmp_path):
    rep = evaluate_oracle(_synthetic_data(sigmas=(0.1, 0.1), bg=0.4), CFG, tmp_path)  # uniform field -> uniform w
    for rule in ("sigma", "sigma_sqrt_w"):
        for g in (1.0, 0.5):
            assert abs(_row(rep, rule, "oracle", 16, g)["dpsnr_analytic_mesh"]) < 0.05
            assert abs(_row(rep, rule, "oracle", 16, g)["dpsnr_mesh_mean"]) < 0.3


def test_evaluate_rejects_empty_mask_and_bad_config(tmp_path):
    with pytest.raises(ValueError, match="no visible mesh"):
        evaluate_oracle(_synthetic_data(mesh=False), CFG, tmp_path)
    for bad in (
        dict(pilot_spp=3),
        dict(budgets=(80,)),
        dict(defensive_fraction=1.0),
        dict(repeats=0),
        dict(gammas=(0.0,)),
    ):
        with pytest.raises(ValueError):
            OracleConfig(**{**asdict(CFG), **bad}).validate()
    for bad in (
        dict(oracle_spp=0), dict(oracle_spp=2**28 + 1), dict(ref_spp=1), dict(ref_spp=2**28 + 1), dict(gammas=()),
        dict(budgets=(3,)), dict(row_width=0), dict(row_width=2**24), dict(defensive_fraction=-0.1),
    ):  # fmt: skip
        with pytest.raises(ValueError):
            OracleConfig(**{**asdict(CFG), **bad}).validate()
    OracleConfig(**asdict(CFG)).validate()
    with pytest.raises(ValueError, match="inside the repository"):
        evaluate_oracle(_synthetic_data(), CFG, REPO_ROOT / "oracle_out")
    assert not (REPO_ROOT / "oracle_out").exists()


def test_evaluate_rejects_inconsistent_data(tmp_path):
    d = _synthetic_data()
    with pytest.raises(ValueError, match="pilot"):
        evaluate_oracle(d, OracleConfig(**{**asdict(CFG), "pilot_spp": 5, "budgets": (8,)}), tmp_path)
    with pytest.raises(ValueError, match="pool"):
        evaluate_oracle(d, OracleConfig(**{**asdict(CFG), "oracle_spp": 32}), tmp_path)


def test_evaluate_mesh_pixel_without_finite_samples_stays_finite(tmp_path):
    d = _synthetic_data()
    d.pilot[0] = float("nan")  # mesh pixel 0: no finite pilot sample and no finite pool sample -> counts = 0
    d.pool[0] = float("nan")
    rep = evaluate_oracle(d, CFG, tmp_path)
    for r in rep.runs:
        assert all(
            math.isfinite(r[c])
            for c in ("mse_mesh", "psnr_mesh", "psnr_all", "flip_mesh", "flip_all", "mse_analytic_mesh")
        )
    assert json.loads((tmp_path / "summary.json").read_text())["n_mesh_pixels_without_samples"] == 1
    assert "no finite samples" in rep.summary_text


# ---------------------------------------------------------------------------------------------------------------------
# Jittered ray generation and row packing (Task 6)
# ---------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def cam():
    return Camera.from_args(
        eye=torch.tensor([1.0, 2.0, 3.0]),
        at=torch.zeros(3),
        up=torch.tensor([0.0, 1.0, 0.0]),
        fov=math.radians(45),
        width=7,
        height=5,
        device="cpu",
    )


# B Task 3's golden values (re-verified numerically for this plan with the half-split sampler)
@pytest.mark.parametrize(
    "pid,s,u,v",
    [
        (13, 0, 0.03014148212969303, 0.3582388758659363),
        (13, 1, 0.02100643329322338, 0.10488247871398926),
        (13, 2, 0.6295420527458191, 0.6791328191757202),
        (13, 3, 0.8542520403862, 0.7263462543487549),
        (13, 2**30, 0.827177882194519, 0.7716277241706848),
        (0, 1, 0.23129604756832123, 0.037553492933511734),
        (34, 5, 0.5679352879524231, 0.8855898976325989),
    ],
)
def test_pixel_jitter_golden_values(pid, s, u, v):
    j = pixel_jitter(torch.tensor([pid]), torch.tensor([s]), width=7)
    assert j[0, 0].item() == pytest.approx(u, abs=1e-7) and j[0, 1].item() == pytest.approx(v, abs=1e-7)


@pytest.mark.parametrize("pid", [0, 13, 34])
@pytest.mark.parametrize("h", [0, 1])
def test_pixel_jitter_each_half_is_stratified(pid, h):  # B's test, plus O's pool and reference offsets
    for k0 in (0, 16, ORACLE_SAMPLE_OFFSET >> 1, REF_SAMPLE_OFFSET >> 1):
        s = 2 * torch.arange(k0, k0 + 16) + h
        j = pixel_jitter(torch.full((16,), pid), s, width=7)
        assert j.dtype == torch.float32 and (j >= 0).all() and (j < 1).all()
        assert sorted(((j[:, 0] * 4).long() * 4 + (j[:, 1] * 4).long()).tolist()) == list(range(16))


def test_pixel_jitter_is_a_pure_function_of_ids():
    pid = torch.tensor([3, 3, 9, 1])
    s = torch.tensor([0, 5, 2, 2**30])
    perm = torch.tensor([2, 0, 3, 1])
    assert torch.equal(pixel_jitter(pid, s, 7)[perm], pixel_jitter(pid[perm], s[perm], 7))


def test_unjittered_rays_match_engine_grid(cam):
    py, px = generate_centered_pixel_coords(7, 5, device="cpu")
    o, d = generate_pinhole_rays(cam, coords_grid=[py, px])
    o2, d2 = oracle_rays(cam, torch.arange(35), torch.zeros(35, dtype=torch.long), jitter=False)
    torch.testing.assert_close(d2, d.float(), atol=1e-6, rtol=0)
    torch.testing.assert_close(o2, o.float())


def test_jittered_rays_stay_inside_their_pixel(cam):
    pid = torch.arange(35).repeat_interleave(8)
    s = torch.arange(8).repeat(35) + ORACLE_SAMPLE_OFFSET
    _, d = oracle_rays(cam, pid, s)
    _, centers = oracle_rays(cam, torch.arange(35), torch.zeros(35, dtype=torch.long), jitter=False)
    assert (d * centers.repeat_interleave(8, 0)).sum(-1).min() > math.cos(math.radians(45 / 5))  # within ~1 px


def test_pack_rows_pads_and_never_aliases():
    n = 10
    o = torch.arange(n * 3.0).reshape(n, 3)
    d = o + 100
    pid = torch.arange(n)
    s = pid + 5
    ro, rd, rp, rs, nv = pack_rows(o, d, pid, s, row_width=4)
    assert ro.shape == (1, 3, 4, 3) and rp.shape == (1, 3, 4) and rp.dtype == torch.int32 and nv == 10
    assert torch.equal(ro.reshape(-1, 3)[:10], o) and torch.equal(ro.reshape(-1, 3)[10:], o[-1:].expand(2, 3))
    ro.zero_()
    assert o.sum() > 0
    ro, *_ = pack_rows(o, d, pid, s, row_width=4096)
    assert ro.shape == (1, 1, 10, 3)
    big = torch.zeros(8193, 3)
    ids = torch.arange(8193)
    assert pack_rows(big, big, ids, ids)[2].shape == (1, 3, 2731)  # B's adaptive width: padding < R
    with pytest.raises(ValueError):
        pack_rows(o[:0], d[:0], pid[:0], s[:0])


def _has_interface_a() -> bool:
    try:
        from threedgrut_playground.tracer import Tracer
    except Exception:
        return False
    return "ray_pixel_id" in inspect.signature(Tracer.render_playground).parameters


requires_interface_a = pytest.mark.skipif(not _has_interface_a(), reason="interface A (plan A) not implemented")


@requires_interface_a
def test_gpu_oracle_smoke(playground_engine, tmp_path):
    e = playground_engine
    e.camera_type = "Pinhole"
    frame_id_before = e.frame_id
    cam, focus = oracle_camera(e, 64, 64)
    names = setup_oracle_scene(e, cam, focus)
    assert len(names) == 4 and frame_id_before == e.frame_id
    cfg = OracleConfig(oracle_spp=16, ref_spp=32, budgets=(8,), repeats=2)
    data = collect_oracle_data(e, cam, cfg)
    m = data.mesh_mask
    M = int(m.sum())
    assert 0 < M < 64 * 64 and data.pool.shape == (M, 16, 3) and data.pilot.shape == (4096, 4, 3)
    assert torch.isfinite(data.pool).float().mean() > 0.99 and (data.ref_count > 0).all()
    pid = data.mesh_index()[:256].repeat_interleave(4)
    s = torch.arange(4).repeat(256) + ORACLE_SAMPLE_OFFSET
    a = trace_samples(e, cam, pid.cuda(), s.cuda())
    b = trace_samples(e, cam, pid.cuda(), s.cuda())
    assert torch.equal(a["rgb"], b["rgb"]) and a["rgb"].shape == (1024, 3)
    rep = evaluate_oracle(data, cfg, tmp_path, device="cuda")
    assert data.ppd == pytest.approx(64 / 60.0) and len(rep.runs) == 9 * 1 * 2
    assert all(math.isfinite(r["psnr_mesh"]) and math.isfinite(r["flip_mesh"]) for r in rep.runs)
    assert e.frame_id == frame_id_before
