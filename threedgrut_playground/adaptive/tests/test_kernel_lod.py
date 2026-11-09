# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-slot LOD acceleration structures and the LOD cone trace (Task 2 of the Gaussian LOD plan).

Most tests use an identity LOD: every cut is the original leaves, so the slot structures differ only in inflation.
That isolates the slot and segment mechanics from merge quality.
"""

import pytest
import torch

from threedgrut_playground.adaptive.tests.kernel_helpers import (
    IMG_H,
    IMG_W,
    SMOOTH,
    bit_equal,
    configure_sphere,
    default_view,
    gradient_envmap,
    grid_ids,
    make_pinhole_rays,
    requires_gpu_ckpt,
    trace_playground,
)

CONE_TRACING = 8  # OptixPlaygroundRenderOptions.CONE_TRACING
IDENTITY_SIGMAS = [0.005, 0.01, 0.02]  # s_1..s_3: K = 2 merged levels, slots 0..3
WIDE_SIGMAS = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32]  # K = 6, slots 0..7
PIXEL_SPREAD = 0.004  # exaggerated camera spread, so camera rays alone cross every band of IDENTITY_SIGMAS
MAX_SIGMA = 0.02
# pred_dist is left out: the playground kernel never writes rayHitDistance, so it is uninitialized memory
BIT_KEYS = (
    "pred_features",
    "pred_opacity",
    "last_ray_o",
    "last_ray_d",
    "hits_count",
    "aov_gauss_depth",
    "aov_first_hit_transmittance",
)


def _identity_lod(mog, level_sigmas=IDENTITY_SIGMAS):
    from threedgrut_playground.lod.build import GaussianLOD

    features = mog.get_features()
    if mog.progressive_training:
        features = features * mog.get_active_feature_mask()
    n = mog.num_gaussians
    leaves = torch.arange(n, dtype=torch.int32, device="cuda")
    return GaussianLOD(
        positions=mog.positions.detach().contiguous(),
        rotations=mog.get_rotation().detach().contiguous(),
        scales=mog.get_scale().detach().contiguous(),
        densities=mog.get_density().detach().contiguous(),
        features=features.detach().contiguous(),
        num_original=n,
        slot_index=[leaves.clone() for _ in range(len(level_sigmas) - 1)],
        level_sigmas=list(level_sigmas),
        key="identity",
    )


@pytest.fixture
def lod_engine(playground_engine):
    """The playground engine; LOD slots are freed and the main BVH restored to the engine's inflation afterwards."""
    engine = playground_engine
    inflate = engine._gs_bvh_inflate
    try:
        yield engine
    finally:
        engine.tracer.tracer_wrapper.clear_lod_bvh()
        engine.tracer.build_gs_acc(gaussians=engine.scene_mog, rebuild=True, inflate=inflate)


def _mirror_view(engine):
    from threedgrut_playground.engine import OptixPrimitiveTypes

    configure_sphere(engine, OptixPrimitiveTypes.MIRROR)
    return make_pinhole_rays(*default_view(engine))


def _render(
    engine,
    rays_o,
    rays_d,
    *,
    cone_on,
    pixel_spread=PIXEL_SPREAD,
    spread_per_alpha=0.0,
    max_sigma=MAX_SIGMA,
    max_roughness=1.0,
    lod=None,
    lod_params=None,
):
    from threedgrut_playground.tracer import make_cone_params

    pixel_id, sample_index = grid_ids(1, IMG_H, IMG_W)
    out = trace_playground(
        engine,
        rays_o,
        rays_d,
        opts=SMOOTH | (CONE_TRACING if cone_on else 0),
        envmap=gradient_envmap(),
        ray_pixel_id=pixel_id,
        ray_sample_index=sample_index,
        cone_params=make_cone_params(pixel_spread, spread_per_alpha, max_sigma, max_roughness, 0),
        lod=lod,
        lod_params=lod_params,
    )
    return {k: v.clone() for k, v in out.items()}


def _build_identity(engine, level_sigmas=IDENTITY_SIGMAS):
    """Uninflated main BVH (slot 0) and slots 1..K+1 of the identity LOD, inflated by the upper band edges."""
    lod = _identity_lod(engine.scene_mog, level_sigmas)
    engine.tracer.build_gs_acc(gaussians=engine.scene_mog, rebuild=True, inflate=0.0)
    engine.tracer.build_lod_acc(lod, lod.level_sigmas)
    return lod


def _assert_bit_equal(a, b):
    for key in BIT_KEYS:
        assert bit_equal(a[key], b[key]), key


#################################
##        --- Layout ---        ##
#################################


def test_lod_params_layout():
    from threedgrut_playground.lod.build import GaussianLOD
    from threedgrut_playground.tracer import Tracer

    n = 4
    lod = GaussianLOD(
        positions=torch.zeros(6, 3),
        rotations=torch.zeros(6, 4),
        scales=torch.ones(6, 3),
        densities=torch.ones(6, 1),
        features=torch.zeros(6, 3),
        num_original=n,
        slot_index=[torch.tensor([4, 5], dtype=torch.int32)],
        level_sigmas=[0.005, 0.01],
        key="layout",
    )
    params = Tracer.make_lod_params(lod, 0.001)
    assert params.dtype == torch.float32
    assert params.device.type == "cpu"
    assert params.shape == (8,)
    # enabled, S = 3, band lower edges (0, s_0, s_1), index offsets (slot 0 unused, slot 1 = 0, slot 2 = N)
    assert params.tolist() == pytest.approx([1.0, 3.0, 0.0, 0.001, 0.005, 0.0, 0.0, 4.0])
    assert Tracer.make_lod_params(lod, 0.001, num_slots=2).tolist() == pytest.approx([1.0, 2.0, 0.0, 0.001, 0.0, 0.0])
    assert Tracer.make_lod_params(lod, 0.001, enabled=False)[0].item() == 0.0
    with pytest.raises(ValueError):
        Tracer.make_lod_params(lod, 0.001, num_slots=4)


#################################
##     --- Bit identity ---     ##
#################################


@requires_gpu_ckpt
@pytest.mark.parametrize("cone_on", [False, True])
def test_lod_off_bit_identical(lod_engine, cone_on):
    engine = lod_engine
    rays_o, rays_d = _mirror_view(engine)
    lod = _build_identity(engine)
    plain = _render(engine, rays_o, rays_d, cone_on=cone_on)
    params = engine.tracer.make_lod_params(lod, 0.0, enabled=False)
    with_lod = _render(engine, rays_o, rays_d, cone_on=cone_on, lod=lod, lod_params=params)
    _assert_bit_equal(plain, with_lod)


@requires_gpu_ckpt
def test_lod_thin_rays_match_cone_off(lod_engine):
    engine = lod_engine
    rays_o, rays_d = _mirror_view(engine)
    lod = _build_identity(engine)
    off = _render(engine, rays_o, rays_d, cone_on=False)
    thin = _render(engine, rays_o, rays_d, cone_on=True, lod=lod, lod_params=engine.tracer.make_lod_params(lod, 1e9))
    _assert_bit_equal(off, thin)


#################################
##  --- Segment equivalence --- ##
#################################


@requires_gpu_ckpt
def test_identity_lod_matches_per_gaussian_widening(lod_engine):
    engine = lod_engine
    rays_o, rays_d = _mirror_view(engine)
    lod = _identity_lod(engine.scene_mog)
    engine.tracer.build_lod_acc(lod, lod.level_sigmas)
    engine.tracer.build_gs_acc(gaussians=engine.scene_mog, rebuild=True, inflate=MAX_SIGMA)
    reference = _render(engine, rays_o, rays_d, cone_on=True)["pred_features"]
    engine.tracer.build_gs_acc(gaussians=engine.scene_mog, rebuild=True, inflate=0.0)
    thin = _render(engine, rays_o, rays_d, cone_on=False)["pred_features"]

    params = engine.tracer.make_lod_params(lod, 0.0)
    lod_out = _render(engine, rays_o, rays_d, cone_on=True, lod=lod, lod_params=params)["pred_features"]
    diff = (lod_out - reference).abs()
    print(f"identity LOD vs per-Gaussian widening: max |diff| {diff.max().item():.3e} mean {diff.mean().item():.3e}")
    sensitivity = (thin - reference).abs().mean().item()
    print(f"cone off vs per-Gaussian widening: mean |diff| {sensitivity:.3e}")
    assert sensitivity > 2e-3, "widening must change the render by more than the tolerance, or the test proves nothing"
    assert diff.mean().item() < 2e-3

    shifted = params.clone()
    num_slots = int(params[1].item())
    shifted[2 : 2 + num_slots] *= 1.03
    shifted_out = _render(engine, rays_o, rays_d, cone_on=True, lod=lod, lod_params=shifted)["pred_features"]
    shift_diff = (shifted_out - lod_out).abs()
    print(f"band edges +3%: max |diff| {shift_diff.max().item():.3e} mean {shift_diff.mean().item():.3e}")
    assert not bit_equal(shifted_out, lod_out), "shifting the band edges must change which slot traces some pieces"
    assert shift_diff.mean().item() < 2e-3


def _remapped_lod(mog, level_sigmas=IDENTITY_SIGMAS, seed=0):
    """The identity LOD's Gaussians as P = 2N particles: the originals, then a permuted copy of them. Cut k's table
    points into the copy and lists the originals in its own random order, so every cut still covers each original
    exactly once, but a kernel that ignored the tables or read another slot's table would fetch the wrong particles."""
    from threedgrut_playground.lod.build import GaussianLOD

    base = _identity_lod(mog, level_sigmas)
    n = base.num_original
    g = torch.Generator().manual_seed(seed)
    q = torch.randperm(n, generator=g).cuda()  # copy row i holds original q[i]
    inv_q = torch.empty_like(q)
    inv_q[q] = torch.arange(n, device="cuda")
    tables = []
    for _ in range(len(level_sigmas) - 1):
        order = torch.randperm(n, generator=g).cuda()  # entry e of the cut is original order[e]
        tables.append((n + inv_q[order]).to(torch.int32))

    def copy(t):
        return torch.cat([t, t[q]]).contiguous()

    return GaussianLOD(
        positions=copy(base.positions),
        rotations=copy(base.rotations),
        scales=copy(base.scales),
        densities=copy(base.densities),
        features=copy(base.features),
        num_original=n,
        slot_index=tables,
        level_sigmas=list(level_sigmas),
        key="remapped",
    )


@requires_gpu_ckpt
def test_remapped_lod_matches_identity_lod(lod_engine):
    engine = lod_engine
    rays_o, rays_d = _mirror_view(engine)
    identity = _build_identity(engine)
    params = engine.tracer.make_lod_params(identity, 0.0)
    expected = _render(engine, rays_o, rays_d, cone_on=True, lod=identity, lod_params=params)["pred_features"]

    remapped = _remapped_lod(engine.scene_mog)
    engine.tracer.build_lod_acc(remapped, remapped.level_sigmas)
    params = engine.tracer.make_lod_params(remapped, 0.0)
    out = _render(engine, rays_o, rays_d, cone_on=True, lod=remapped, lod_params=params)["pred_features"]
    diff = (out - expected).abs()
    print(
        f"remapped vs identity LOD: max |diff| {diff.max().item():.3e} mean {diff.mean().item():.3e}"
        f" bit-equal {bit_equal(out, expected)}"
    )
    assert torch.isfinite(out).all()
    assert diff.mean().item() < 2e-3

    # Negative control: slots 2 and 3 read each other's tables, so their hits fetch the wrong particles
    swapped = params.clone()
    num_slots = int(params[1].item())
    offsets = 2 + num_slots
    swapped[offsets + 2], swapped[offsets + 3] = params[offsets + 3], params[offsets + 2]
    wrong = _render(engine, rays_o, rays_d, cone_on=True, lod=remapped, lod_params=swapped)["pred_features"]
    wrong_diff = (wrong - expected).abs().mean().item()
    print(f"swapped tables vs identity LOD: mean |diff| {wrong_diff:.3e}")
    assert wrong_diff > 2e-3, "a wrong table must fail the tolerance, or the test proves nothing"


@requires_gpu_ckpt
@pytest.mark.parametrize("target_slot", [1, 2])
def test_constant_width_ray_selects_one_slot(lod_engine, target_slot):
    # Every ray has coneSpread == 0: pixel_spread 0, and spread_per_alpha 0 so the roughness-r PBR bounce adds no spread.
    # (coneSigma only grows by coneSpread * t in the kernel, so a constant-width ray always has width 0.) Band lower
    # edges of slots 1..target are 0, so width 0 lies in target's band; a correct split traces only that slot.
    from threedgrut_playground.engine import OptixPrimitiveTypes

    from threedgrut_playground.adaptive.tests.test_kernel_cone import _metal

    engine = lod_engine
    _metal(engine, "lod_constant_width", roughness_factor=0.1)
    configure_sphere(engine, OptixPrimitiveTypes.PBR, "lod_constant_width")
    rays_o, rays_d = make_pinhole_rays(*default_view(engine))
    lod = _build_identity(engine)
    params = engine.tracer.make_lod_params(lod, 0.0)
    num_slots = int(params[1].item())
    params[3 : 2 + target_slot + 1] = 0.0
    only = params.clone()
    only[2 + target_slot + 1 : 2 + num_slots] = 1e9
    a = _render(engine, rays_o, rays_d, cone_on=True, pixel_spread=0.0, lod=lod, lod_params=params)
    b = _render(engine, rays_o, rays_d, cone_on=True, pixel_spread=0.0, lod=lod, lod_params=only)
    assert torch.isfinite(a["pred_features"]).all()
    torch.testing.assert_close(a["pred_features"], b["pred_features"], atol=1e-6, rtol=0)
    # The slots' inflated proxies reorder hits, so tracing a slot instead of the main BVH changes some bits
    plain = _render(engine, rays_o, rays_d, cone_on=True, pixel_spread=0.0)
    assert not bit_equal(a["pred_features"], plain["pred_features"]), "the LOD slots must be traced"


@requires_gpu_ckpt
def test_rays_starting_inside_band_finite(lod_engine):
    # Reflected cones start at coneSigma = PIXEL_SPREAD * t_hit (inside slot 2's band) and grow past the top band.
    from threedgrut_playground.engine import OptixPrimitiveTypes

    from threedgrut_playground.adaptive.tests.test_kernel_cone import _metal

    engine = lod_engine
    _metal(engine, "lod_rough", roughness_factor=0.6)
    configure_sphere(engine, OptixPrimitiveTypes.PBR, "lod_rough")
    rays_o, rays_d = make_pinhole_rays(*default_view(engine))
    lod = _build_identity(engine, WIDE_SIGMAS)
    params = engine.tracer.make_lod_params(lod, 0.001)
    out = _render(
        engine, rays_o, rays_d, cone_on=True, spread_per_alpha=1.414, max_sigma=0.32, lod=lod, lod_params=params
    )
    for key, value in out.items():
        if value.is_floating_point():
            assert torch.isfinite(value).all(), key


#################################
##  --- Incremental slots ---  ##
#################################


def _f32(x):
    return torch.tensor(x, dtype=torch.float32).item()


@requires_gpu_ckpt
def test_build_lod_acc_is_incremental(lod_engine):
    engine = lod_engine
    tracer = engine.tracer
    rays_o, rays_d = _mirror_view(engine)
    lod = _identity_lod(engine.scene_mog)
    engine.tracer.build_gs_acc(gaussians=engine.scene_mog, rebuild=True, inflate=0.0)
    tracer.clear_lod_acc()
    assert tracer.build_lod_acc(lod, [0.005, 0.01, 0.015]) == [1, 2, 3]
    assert tracer.tracer_wrapper.lod_slot_inflations()[:5] == [-1.0, _f32(0.005), _f32(0.01), _f32(0.015), -1.0]
    assert tracer.build_lod_acc(lod, [0.005, 0.01, 0.015]) == [], "unchanged slots are kept"
    assert tracer.build_lod_acc(lod, [0.005, 0.01, 0.02]) == [3], "a cap change within a band rebuilds the top slot"
    params = tracer.make_lod_params(lod, 0.0)
    incremental = _render(engine, rays_o, rays_d, cone_on=True, lod=lod, lod_params=params)
    assert tracer.build_lod_acc(lod, [0.005, 0.01]) == [], "a narrower cap reuses the lower slots"
    assert tracer.build_lod_acc(lod, [0.005, 0.01, 0.02]) == [], "and keeps the slot above them"

    # Growing the slot count builds only the added slots (here a wider LOD with the same first cuts)
    wide = _identity_lod(engine.scene_mog, WIDE_SIGMAS)
    assert tracer.build_lod_acc(wide, WIDE_SIGMAS[:3]) == [1, 2, 3], "another LOD rebuilds every slot"
    assert tracer.build_lod_acc(wide, WIDE_SIGMAS[:5]) == [4, 5]

    # A slot rebuilt in place traces exactly like one built from scratch
    tracer.clear_lod_acc()
    assert tracer.build_lod_acc(lod, [0.005, 0.01, 0.02]) == [1, 2, 3]
    fresh = _render(engine, rays_o, rays_d, cone_on=True, lod=lod, lod_params=params)
    _assert_bit_equal(incremental, fresh)

    # A clear from outside the tracer is noticed through the C++ slot state
    tracer.tracer_wrapper.clear_lod_bvh()
    assert tracer.build_lod_acc(lod, [0.005, 0.01, 0.02]) == [1, 2, 3]
    with pytest.raises(ValueError):
        tracer.build_lod_acc(lod, [0.005, -0.01])
    with pytest.raises(RuntimeError):
        n = lod.num_original
        tracer.tracer_wrapper.build_lod_bvh(
            1, lod.positions[:n], lod.rotations[:n], lod.scales[:n], lod.densities[:n], float("nan")
        )


@requires_gpu_ckpt
def test_widening_clamped_to_slot_inflation(lod_engine):
    # The top slot is inflated by 0.02. With the clamp, a cone cap of 0.04 widens its Gaussians by at most 0.02, so the
    # frame is bit-identical to cap 0.02. Negative control: the same slot inflated by 0.04 at cap 0.04 differs, so the
    # cones do get wider than 0.02 in this view.
    engine = lod_engine
    rays_o, rays_d = _mirror_view(engine)
    lod = _build_identity(engine)
    params = engine.tracer.make_lod_params(lod, 0.0)
    at_inflation = _render(engine, rays_o, rays_d, cone_on=True, max_sigma=0.02, lod=lod, lod_params=params)
    above = _render(engine, rays_o, rays_d, cone_on=True, max_sigma=0.04, lod=lod, lod_params=params)
    _assert_bit_equal(at_inflation, above)
    assert engine.tracer.build_lod_acc(lod, [0.005, 0.01, 0.04]) == [3]
    wider = _render(engine, rays_o, rays_d, cone_on=True, max_sigma=0.04, lod=lod, lod_params=params)
    assert torch.isfinite(wider["pred_features"]).all()
    assert not bit_equal(wider["pred_features"], above["pred_features"]), "cones must exceed 0.02 somewhere"


#################################
##      --- Host checks ---    ##
#################################


@requires_gpu_ckpt
@pytest.mark.parametrize("num_slots", [float("nan"), float("inf"), 1e30, -1.0, 2.5, 11.0])
def test_lod_params_bad_slot_count_rejected(lod_engine, num_slots):
    # S is range-checked as a float before the cast to int (casting NaN or a huge float is undefined behaviour)
    engine = lod_engine
    rays_o, rays_d = _mirror_view(engine)
    params = torch.tensor([0.0, num_slots], dtype=torch.float32)
    with pytest.raises(RuntimeError, match="S must be an integer"):
        _render(engine, rays_o, rays_d, cone_on=True, lod_params=params)


@requires_gpu_ckpt
@pytest.mark.parametrize("offset", [float("inf"), float("nan"), 1e30, -4.0])
def test_lod_params_bad_offset_rejected(lod_engine, offset):
    engine = lod_engine
    rays_o, rays_d = _mirror_view(engine)
    lod = _build_identity(engine)
    params = engine.tracer.make_lod_params(lod, 0.0)
    num_slots = int(params[1].item())
    params[2 + num_slots + 2] = offset
    with pytest.raises(RuntimeError, match="index table"):
        _render(engine, rays_o, rays_d, cone_on=True, lod=lod, lod_params=params)
