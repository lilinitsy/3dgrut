# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from threedgrut_playground.adaptive.evaluation import metrics as M


def _inputs(h=6, w=8):
    z = torch.zeros(h, w)
    return dict(
        ref_hdr=torch.full((h, w, 3), 0.5),
        ref_var_mean=z + 1e-6,
        ref_hi_mask=torch.ones(h, w, dtype=torch.bool),
        flags=torch.zeros(h, w, dtype=torch.bool),
        ecc_deg=z,
        weight=z + 1.0,
        ppd=8.0,
    )


def test_psnr_known_values_and_masks():
    ref = torch.zeros(4, 4, 3)
    assert M.psnr(ref + 0.1, ref) == pytest.approx(20.0, abs=1e-4)
    assert M.psnr(ref, ref) == math.inf
    assert math.isnan(M.psnr(ref + 0.1, ref, torch.zeros(4, 4, dtype=torch.bool)))


def test_sanity_objective():
    ref = torch.zeros(2, 2, 3)
    t = ref.clone()
    t[0, 0] = 1.0
    w = torch.tensor([[2.0, 1.0], [1.0, 1.0]])
    assert M.sanity_objective(t, ref, w) == pytest.approx(0.5)
    assert math.isnan(M.sanity_objective(t, ref, w, torch.zeros(2, 2, dtype=torch.bool)))


def test_ref_noise_flags_against_uniform_run():
    assert M.ref_noise_flags(torch.tensor([[1e-6, 1e-3, 1e-6]]), torch.tensor([[1e-4, 1e-3, 0.0]])).tolist() == [
        [False, True, True]
    ]


def test_primary_metric_follows_weight_model():
    assert M.PRIMARY_METRIC == {
        "cielab": ("flip_mesh", "lower"),
        "cielab_nomask": ("flip_mesh", "lower"),
        "tvi": ("cvvdp_mesh", "higher"),
    }


def test_reference_weights_builds_reference_ctx():
    seen = {}

    class Term:
        name = "probe"

        def __call__(self, ctx):
            seen["ctx"] = ctx
            return torch.full(ctx.luminance.shape, 3.0)

    ref, mask, nv = torch.rand(3, 5, 3), torch.ones(3, 5, dtype=torch.bool), torch.rand(3, 5)
    w = M.reference_weights([Term()], ref, mask, torch.zeros(3, 5, 3), camera=None, ref_noise_var=nv)
    assert w.shape == (3, 5) and seen["ctx"].mean_rgb is ref and seen["ctx"].noise_var is nv


def test_evaluate_image_keys_regions_flags_and_nonfinite(monkeypatch):
    monkeypatch.setattr(M, "cvvdp_available", lambda: False)
    inp = _inputs()
    test = inp["ref_hdr"].clone()
    test[0, 0, 0] = math.nan
    test[5, 7] = 0.9
    mesh = torch.zeros(6, 8, dtype=torch.bool)
    mesh[4:, 6:] = True
    out = M.evaluate_image(test, mesh_mask=mesh, **inp)
    assert tuple(out) == M.METRIC_KEYS and len(M.METRIC_KEYS) == 36
    assert out["nonfinite_px"] == 1 and math.isfinite(out["flip_all"]) and out["flip_mesh"] > 0
    assert math.isnan(out["psnr_mesh_ecc_5_10"]) and math.isnan(out["cvvdp_all"])
    flagged = {**inp, "flags": mesh.clone()}
    out2 = M.evaluate_image(test, mesh_mask=mesh, **flagged)
    assert out2["ref_flagged_frac_mesh"] == 1.0 and math.isnan(out2["flip_mesh_clean"])
    empty = M.evaluate_image(test, mesh_mask=torch.zeros(6, 8, dtype=torch.bool), **inp)
    assert math.isnan(empty["flip_mesh"]) and math.isnan(empty["sanity_objective_mesh"])


def test_flip_identity_error_and_ppd_dependence():
    ref = torch.rand(48, 48, 3)
    noisy = (ref + 0.1 * torch.randn(48, 48, 3, generator=torch.Generator().manual_seed(0))).clamp(0, 1)
    assert M.flip_error_map(ref, ref, 8.5).abs().max().item() < 1e-6
    assert M.flip_error_map(noisy, ref, 8.5).mean().item() != pytest.approx(
        M.flip_error_map(noisy, ref, 67.0).mean().item()
    )


def test_cvvdp_optional():
    pytest.importorskip("pycvvdp")
    ref = torch.rand(64, 64, 3)
    assert M.cvvdp_jod(ref, ref, 8.5) == pytest.approx(10.0, abs=0.05)
    assert M.cvvdp_jod((ref + 0.3 * torch.randn_like(ref)).clamp(0, 1), ref, 8.5) < 9.5
