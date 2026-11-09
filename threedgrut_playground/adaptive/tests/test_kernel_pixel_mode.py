# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-pixel render mode (``Engine3DGRUT.pixel_mode``): MC sample counts and cone caps chosen per pixel in one launch.

Each mode is pinned against the existing path it replaces: n MC samples against the mean of n frames, a cone pixel
against a cone frame at the same cap, MC 1 against a plain frame. Scene: the roughness ladder (metal spheres on a
copper floor) at 96 px.
"""

import pytest
import torch

from threedgrut_playground.adaptive.tests.kernel_helpers import requires_gpu_ckpt

RES = 96
FRAME = 11


@pytest.fixture
def ladder(playground_engine):
    """The engine with the ladder scene, cones and LOD off, unclamped; pixel mode and LOD state reset afterwards."""
    from threedgrut_playground.adaptive.evaluation.cone_equal_time import configure, load_ladder

    engine = playground_engine
    saved = {k: getattr(engine, k) for k in ("use_gaussian_lod", "cone_camera_rays", "cone_max_sigma", "camera_fov")}
    inflate = engine._gs_bvh_inflate
    preset = load_ladder(engine)
    configure(engine, None)
    rays = engine.raygen(preset.to_camera(RES, RES, device=torch.device(engine.device)), use_spp=False)
    try:
        yield engine, rays.rays_ori.contiguous(), rays.rays_dir.contiguous()
    finally:
        engine.pixel_mode = None
        for k, v in saved.items():
            setattr(engine, k, v)
        engine.tracer.tracer_wrapper.clear_lod_bvh()
        engine._lod_slots = None
        engine._gs_bvh_inflate = inflate
        engine.tracer.build_gs_acc(gaussians=engine.scene_mog, rebuild=True, inflate=inflate)


def _frame(engine, o, d, frame_id=FRAME):
    engine.frame_id = frame_id
    out = engine._render_playground_hybrid(o.clone(), d.clone())
    return out["pred_features"][0].clone(), out["pred_opacity"][0].clone()


def _cones(engine, cap, lod=True):
    from threedgrut_playground.adaptive.evaluation.cone_equal_time import ConeSetting, configure

    configure(engine, ConeSetting(cap, 1.0))
    engine.use_gaussian_lod = lod


@requires_gpu_ckpt
def test_mc_one_sample_is_bit_identical_to_a_plain_frame(ladder):
    engine, o, d = ladder
    plain, plain_a = _frame(engine, o, d)
    engine.pixel_mode = torch.ones(RES, RES, device="cuda")
    moded, moded_a = _frame(engine, o, d)
    assert torch.equal(plain, moded) and torch.equal(plain_a, moded_a)


@requires_gpu_ckpt
def test_mc_n_samples_equal_the_mean_of_n_frames(ladder):
    engine, o, d = ladder
    n = 4
    mean = sum(_frame(engine, o, d, FRAME + i)[0] for i in range(n)) / n
    engine.pixel_mode = torch.full((RES, RES), float(n), device="cuda")
    moded, _ = _frame(engine, o, d)
    assert torch.allclose(moded, mean, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(moded, _frame(engine, o, d, FRAME + 100)[0], atol=1e-3)  # samples really differ


@requires_gpu_ckpt
def test_cone_mode_is_bit_identical_to_a_cone_frame(ladder):
    engine, o, d = ladder
    _cones(engine, 0.1)
    cone, cone_a = _frame(engine, o, d)
    engine.pixel_mode = torch.full((RES, RES), -0.1, device="cuda")
    moded, moded_a = _frame(engine, o, d)
    assert torch.equal(cone, moded) and torch.equal(cone_a, moded_a)


@requires_gpu_ckpt
def test_cone_mode_without_lod_is_bit_identical_to_a_cone_frame(ladder):
    engine, o, d = ladder
    _cones(engine, 0.01, lod=False)
    cone, _ = _frame(engine, o, d)
    engine.pixel_mode = torch.full((RES, RES), -0.01, device="cuda")
    moded, _ = _frame(engine, o, d)
    assert torch.equal(cone, moded)


@requires_gpu_ckpt
def test_zero_mode_renders_nothing(ladder):
    engine, o, d = ladder
    engine.pixel_mode = torch.zeros(RES, RES, device="cuda")
    _, alpha = _frame(engine, o, d)
    assert float(alpha.abs().max()) == 0.0


@requires_gpu_ckpt
def test_mixed_frame_matches_each_mode_on_its_pixels(ladder):
    """Left: MC 2 samples; middle: cone capped at 0.03; right: cone capped at 0.2. The slots are built for 0.2, and a
    cap variant (the 0.02-0.04 band slot inflated by exactly 0.03) serves the middle, so every region equals the frame
    it would come from on its own."""
    engine, o, d = ladder
    third = RES // 3
    mode = torch.full((RES, RES), 2.0, device="cuda")
    mode[:, third : 2 * third] = -0.03
    mode[:, 2 * third :] = -0.2
    mc2 = (_frame(engine, o, d, FRAME)[0] + _frame(engine, o, d, FRAME + 1)[0]) / 2
    _cones(engine, 0.03)
    c03, _ = _frame(engine, o, d)
    _cones(engine, 0.2)
    c20, _ = _frame(engine, o, d)
    engine.pixel_mode = mode
    moded, _ = _frame(engine, o, d)
    assert torch.allclose(moded[:, :third], mc2[:, :third], atol=1e-5, rtol=1e-5)
    assert torch.equal(moded[:, 2 * third :], c20[:, 2 * third :])
    assert torch.equal(moded[:, third : 2 * third], c03[:, third : 2 * third])
    variants = engine._lod_slots[3]  # slot 4 is the band [0.02, 0.04): slots 0 (thin), 1 [s_0, 0.005), 2, 3, 4
    assert len(variants) == 1 and variants[0][1] == 4 and variants[0][0] == pytest.approx(0.03), variants


@requires_gpu_ckpt
def test_cap_variants_follow_the_mode(ladder):
    """Changing the caps in the mode rebuilds only what changed; a cap variant never serves a different cap."""
    engine, o, d = ladder
    _cones(engine, 0.35)
    for caps in ((0.015, 0.35), (0.015, 0.06, 0.35), (0.1, 0.35)):
        mode = torch.full((RES, RES), -caps[-1], device="cuda")
        for i, c in enumerate(caps[:-1]):
            mode[i * 8 : (i + 1) * 8] = -c
        engine.pixel_mode = mode
        moded, _ = _frame(engine, o, d)
        for i, c in enumerate(caps[:-1]):
            engine.pixel_mode = None
            _cones(engine, c)
            single, _ = _frame(engine, o, d)
            assert torch.equal(moded[i * 8 : (i + 1) * 8], single[i * 8 : (i + 1) * 8]), f"cap {c} in mode {caps}"
            _cones(engine, 0.35)
