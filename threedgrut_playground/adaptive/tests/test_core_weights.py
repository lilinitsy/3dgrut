# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for threedgrut_playground.adaptive.display and weights."""

import pytest
import torch

from threedgrut_playground.adaptive.display import (
    CIELAB_KAPPA,
    display_luminance_cd,
    lstar_slope,
    srgb_eotf,
    tvi,
)
from threedgrut_playground.adaptive.image_ops import luminance
from threedgrut_playground.adaptive.oracle import (
    display_clip_weight,
    luminance_adaptation_weight,
    perceptual_weight,
)
from threedgrut_playground.adaptive.protocols import WeightContext, WeightTerm
from threedgrut_playground.adaptive.weights import (
    ContrastMaskingTerm,
    DisplayClipTerm,
    LuminanceAdaptationTerm,
    TviAdaptationTerm,
    combine_weights,
)

LSTAR_W = [0.4886, 1.6657, 1.3649, 1.1957, 1.0827, 1.0, 0.9359, 0.8842, 0.8413, 0.8049, 0.7734]  # Y' = 0, 0.1, ..., 1
TVI_W = [55.1465, 1.0, 0.2838]  # Y' = 0, 0.5, 1


def _grey(v, shape=(16, 16)):
    return torch.full((*shape, 3), float(v), dtype=torch.float32)


def _ctx(rgb, noise_var=None, mask=None):
    h, w = rgb.shape[:2]
    return WeightContext(
        mean_rgb=rgb,
        luminance=luminance(rgb),
        mesh_mask=torch.ones(h, w, dtype=torch.bool) if mask is None else mask,
        view_dirs=torch.zeros(h, w, 3),
        camera=None,
        noise_var=noise_var,
    )


class _Stub:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __call__(self, ctx):
        return torch.full(ctx.luminance.shape, self.value)


def test_srgb_eotf_lstar_and_display_model():
    v = torch.tensor([0.0, 0.04045, 0.5, 1.0], dtype=torch.float64)
    torch.testing.assert_close(srgb_eotf(v), torch.tensor([0.0, 0.04045 / 12.92, 0.21404114, 1.0], dtype=torch.float64))
    assert lstar_slope(torch.tensor(0.0, dtype=torch.float64)).item() == pytest.approx(CIELAB_KAPPA / 12.92)
    torch.testing.assert_close(
        display_luminance_cd(_grey(1.0).double())[0, 0], torch.tensor(200.2, dtype=torch.float64)
    )
    assert (tvi(torch.tensor(100.0)) / tvi(torch.tensor(50.0))).item() == pytest.approx(2.0, rel=0.02)  # Weber
    assert (tvi(torch.tensor(0.01)) / tvi(torch.tensor(0.0025))).item() == pytest.approx(2.0, rel=0.02)  # De Vries-Rose


def test_default_luminance_term_is_the_lstar_slope():
    t = LuminanceAdaptationTerm()
    assert [t(_ctx(_grey(y / 10)))[0, 0].item() for y in range(11)] == pytest.approx(LSTAR_W, abs=1e-3)
    ys = torch.linspace(0, 1, 2001)
    w = t(_ctx(ys[None, :, None].expand(1, 2001, 3)))[0]  # one row of grey levels
    assert w.min().item() == pytest.approx(0.4886, abs=1e-3) and w.max().item() == pytest.approx(1.7003, abs=2e-3)
    assert abs(ys[w.argmax()].item() - 0.0922) < 1e-3


def test_default_luminance_term_is_pointwise_and_sanitizes():
    rgb = _grey(0.9, (64, 64))
    rgb[28:36, 28:36] = 0.2
    w = LuminanceAdaptationTerm()(_ctx(rgb))
    assert w[32, 32].item() == pytest.approx(1.3649, abs=1e-3) and w[0, 0].item() == pytest.approx(0.8049, abs=1e-3)
    for bad in (float("nan"), -1.0):
        assert LuminanceAdaptationTerm()(_ctx(_grey(bad)))[0, 0].item() == pytest.approx(0.4886, abs=1e-3)


def test_tvi_term_values_and_pooling():
    t = TviAdaptationTerm()
    assert [t(_ctx(_grey(y)))[0, 0].item() for y in (0.0, 0.5, 1.0)] == pytest.approx(TVI_W, rel=1e-3)
    rgb = _grey(0.9, (64, 64))
    rgb[28:36, 28:36] = 0.2  # small dark patch in a bright surround
    assert t(_ctx(rgb))[32, 32] < 0.5 * t(_ctx(_grey(0.2, (64, 64))))[32, 32]
    assert torch.isfinite(t(_ctx(_grey(float("nan"))))).all()


def test_contrast_masking_flat_texture_and_noise_correction():
    term = ContrastMaskingTerm()
    torch.testing.assert_close(term(_ctx(_grey(0.5))), torch.ones(16, 16))
    checker = ((torch.arange(16)[:, None] + torch.arange(16)[None, :]) % 2).float()
    tex = (0.1 + 0.4 * checker)[..., None].expand(16, 16, 3)
    assert term(_ctx(tex))[4:12, 4:12].max() < 0.1
    noisy = (0.5 + 0.02 * (2 * checker - 1))[..., None].expand(16, 16, 3)
    assert term(_ctx(noisy))[4:12, 4:12].max() < 0.5  # the self-masking effect R8 documents
    assert term(_ctx(noisy, noise_var=torch.full((16, 16), 0.02**2)))[4:12, 4:12].min() > 0.9


def test_contrast_masking_is_class_aware():
    mask = torch.zeros(16, 16, dtype=torch.bool)
    mask[:, :8] = True
    rgb = torch.where(mask[..., None], 0.9, 0.1).expand(16, 16, 3)  # flat within each class, edge between
    torch.testing.assert_close(ContrastMaskingTerm()(_ctx(rgb, mask=mask)), torch.ones(16, 16))


def test_display_clip_term_curve():
    t = DisplayClipTerm()
    vals = {m: t(_ctx(_grey(m)))[0, 0].item() for m in (0.5, 1.0, 1.025, 1.05, 1.5)}
    assert vals == pytest.approx({0.5: 1.0, 1.0: 1.0, 1.025: 0.5, 1.05: 0.0, 1.5: 0.0}, abs=1e-6)
    partly = _grey(1.5).clone()
    partly[..., 2] = 0.3  # only R,G clip -> still visible
    assert t(_ctx(partly))[0, 0].item() == 1.0
    assert t(_ctx(_grey(1.1), noise_var=torch.full((16, 16), 0.01)))[0, 0].item() == pytest.approx(0.5, abs=1e-6)


def test_combine_weights():
    w, per = combine_weights([_Stub("a", 2.0), _Stub("b", 3.0)], _ctx(_grey(0.2)))
    torch.testing.assert_close(w, torch.full((16, 16), 6.0))
    assert set(per) == {"a", "b"}
    w, per = combine_weights([], _ctx(_grey(0.2)))
    torch.testing.assert_close(w, torch.ones(16, 16))
    assert per == {}
    w, _ = combine_weights([_Stub("n", float("nan"))], _ctx(_grey(0.2)))
    assert torch.equal(w, torch.zeros(16, 16))
    with pytest.raises(ValueError, match="duplicate"):
        combine_weights([_Stub("a", 1.0), _Stub("a", 2.0)], _ctx(_grey(0.2)))
    wrong = _Stub("wrong", 1.0)
    wrong.__class__ = type("_Wrong", (_Stub,), {"__call__": lambda self, ctx: torch.ones(4, 4)})
    with pytest.raises(ValueError, match="shape"):
        combine_weights([wrong], _ctx(_grey(0.2)))


def test_terms_conform_to_protocol():
    terms = (LuminanceAdaptationTerm(), TviAdaptationTerm(), ContrastMaskingTerm(), DisplayClipTerm())
    assert all(isinstance(t, WeightTerm) for t in terms) and len({t.name for t in terms}) == 4


def test_default_terms_match_the_oracle_copies():
    g = torch.Generator().manual_seed(0)
    rgb = torch.rand(24, 24, 3, generator=g) * 1.3
    rgb[0, 0] = float("nan")
    rgb[1, 1] = -0.5
    nv = torch.rand(24, 24, generator=g) * 0.01
    for noise_var in (None, nv):
        ctx = _ctx(rgb, noise_var=noise_var)
        w, per = combine_weights([LuminanceAdaptationTerm(), DisplayClipTerm()], ctx)
        torch.testing.assert_close(
            per["luminance_adaptation"].double(), luminance_adaptation_weight(rgb).to(torch.float32).double()
        )
        torch.testing.assert_close(per["display_clip"].double(), display_clip_weight(rgb, noise_var).float().double())
        torch.testing.assert_close(w.double(), perceptual_weight(rgb, noise_var).float().double(), rtol=1e-5, atol=1e-6)


def test_display_clip_margin_uses_the_raw_variance_when_given():
    # Pixel 0: one 50x firefly among four pilot samples of 0.5 (raw mean 12.875); pixel 1: four samples at 2.0 (white).
    raw_mean = torch.tensor([[[12.875] * 3, [2.0] * 3]])
    wins_var = torch.tensor([[0.0072, 0.0]])  # winsorized variance of the mean: the firefly is clamped to 1.05
    raw_var = torch.tensor([[153.1, 0.0]])  # raw variance of the mean
    ctx = _ctx(raw_mean, noise_var=wins_var)
    assert (DisplayClipTerm()(ctx) == 0).all()  # fallback (oracle parity): winsorized margin clips it
    w = DisplayClipTerm()(WeightContext(**{**ctx.__dict__, "raw_var_of_mean": raw_var}))
    assert w[0, 0] > 0.4 and w[0, 1] == 0.0  # firefly pixel stays eligible; the truly white pixel still clips
