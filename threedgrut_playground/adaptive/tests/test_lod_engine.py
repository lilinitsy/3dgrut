# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gaussian LOD in the engine: the on-disk cache, the slot plan and the render rule (Task 3 of the Gaussian LOD plan).

The CPU tests cover the cache key, the cache round trip and the slot plan. The GPU tests drive
``Engine3DGRUT._render_playground_hybrid`` and write the cache to a temporary directory, never to the real one.
"""

import copy
import os

import pytest
import torch

from threedgrut_playground.adaptive.tests.kernel_helpers import (
    IMG_W,
    bit_equal,
    configure_sphere,
    default_view,
    make_pinhole_rays,
    requires_gpu_ckpt,
    restore_engine,
    snapshot_engine,
)

K7_SIGMAS = [0.005 * 2**k for k in range(8)]  # s_1..s_8 for the defaults (base 0.005, K = 7)


def _small_lod(key="small"):
    from threedgrut_playground.lod.build import build_lod

    g = torch.Generator().manual_seed(3)
    n = 64
    pos = torch.rand(n, 3, generator=g)
    rot = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=1)
    scl = 0.002 + 0.01 * torch.rand(n, 3, generator=g)
    dns = 0.1 + 0.8 * torch.rand(n, 1, generator=g)
    feat = torch.rand(n, 3, generator=g)
    return build_lod(pos, rot, scl, dns, feat, base_sigma=0.005, num_levels=4, key=key)


#################################
##        --- Cache ---        ##
#################################


def test_cache_key_changes_with_checkpoint(tmp_path):
    from threedgrut_playground.lod.cache import lod_cache_key

    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    a.write_bytes(b"a")
    b.write_bytes(b"a")
    stat = os.stat(a)
    os.utime(b, ns=(stat.st_atime_ns, stat.st_mtime_ns))  # same mtime: only the path differs
    base = lod_cache_key(str(a), 100, 0.005, 7)
    assert lod_cache_key(str(a), 100, 0.005, 7) == base
    variants = {
        "path": lod_cache_key(str(b), 100, 0.005, 7),
        "count": lod_cache_key(str(a), 101, 0.005, 7),
        "base_sigma": lod_cache_key(str(a), 100, 0.0051, 7),
        "num_levels": lod_cache_key(str(a), 100, 0.005, 6),
    }
    os.utime(a, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    variants["mtime"] = lod_cache_key(str(a), 100, 0.005, 7)
    for name, key in variants.items():
        assert key != base, name
    assert len(set(variants.values())) == len(variants)
    # The engine passes the mtime it saw when it loaded the checkpoint, so a file rewritten later cannot be keyed to
    # the Gaussians already in memory
    assert lod_cache_key(str(a), 100, 0.005, 7, mtime_ns=stat.st_mtime_ns) == base


def test_cache_roundtrip(tmp_path):
    from threedgrut_playground.lod.cache import cache_path, load_or_build

    calls = []

    def build():
        calls.append(1)
        return _small_lod("k1")

    first = load_or_build("k1", build, cache_dir=str(tmp_path))
    second = load_or_build("k1", build, cache_dir=str(tmp_path))
    assert len(calls) == 1, "build_fn must not run when the cache holds the key"
    assert os.path.isfile(cache_path("k1", str(tmp_path)))
    # Plain tensors and builtins only, so the file also loads with torch's safe default
    torch.load(cache_path("k1", str(tmp_path)), weights_only=True)

    for name in ("positions", "rotations", "scales", "densities", "features"):
        a, b = getattr(first, name), getattr(second, name)
        assert a.dtype == b.dtype and torch.equal(a, b.to(a.device)), name
    assert second.num_original == first.num_original
    assert second.level_sigmas == first.level_sigmas
    assert second.key == first.key == "k1"
    assert len(second.slot_index) == len(first.slot_index)
    for a, b in zip(first.slot_index, second.slot_index):
        assert b.dtype == torch.int32 and torch.equal(a, b.to(a.device))
    assert torch.equal(first.particle_density(), second.particle_density().to(first.positions.device))


def test_cache_unreadable_file_rebuilds(tmp_path, capsys):
    from threedgrut_playground.lod.cache import cache_path, load_or_build

    os.makedirs(tmp_path, exist_ok=True)
    with open(cache_path("bad", str(tmp_path)), "wb") as f:
        f.write(b"not a torch file")
    calls = []

    def build():
        calls.append(1)
        return _small_lod("bad")

    lod = load_or_build("bad", build, cache_dir=str(tmp_path))
    assert calls == [1] and lod.key == "bad"
    assert "unreadable" in capsys.readouterr().out
    load_or_build("bad", build, cache_dir=str(tmp_path))
    assert calls == [1], "the rebuilt hierarchy must have replaced the bad file"


#################################
##      --- Slot plan ---      ##
#################################


@pytest.mark.parametrize(
    "thin, cap, expected",
    [
        (0.001, 0.005, (2, 0.005)),  # default: slot 0 plus slot 1 (originals inflated by s_1)
        (0.001, 0.02, (4, 0.02)),  # band lower edges 0, 0.001, 0.005, 0.01 are below 0.02; 0.02 is not
        (0.001, 0.32, (8, 0.32)),
        (0.001, 2.0, (9, 0.64)),  # clamped to s_8, every slot
        (0.001, 0.0005, (1, 0.0005)),  # below the thin threshold: only the uninflated main BVH
        (1e9, 0.02, (1, 0.02)),  # thin threshold above the cap: every cone width is thin
        (0.0, 0.005, (2, 0.005)),
    ],
)
def test_plan_lod_slots(thin, cap, expected):
    from threedgrut_playground.lod.slots import plan_lod_slots

    num_slots, max_sigma = plan_lod_slots(K7_SIGMAS, thin, cap)
    assert num_slots == expected[0]
    assert max_sigma == pytest.approx(expected[1])


def test_plan_lod_slots_respects_slot_limit():
    from threedgrut_playground.lod.slots import plan_lod_slots
    from threedgrut_playground.tracer import MAX_LOD_SLOTS

    sigmas = [0.005 * 2**k for k in range(12)]  # K = 11: more slots than the kernel has
    num_slots, max_sigma = plan_lod_slots(sigmas, 0.001, 100.0)
    assert num_slots == MAX_LOD_SLOTS
    assert max_sigma == pytest.approx(sigmas[MAX_LOD_SLOTS - 2])  # inflation of the highest slot that exists


#################################
##     --- Engine (GPU) ---    ##
#################################

LOD_SETTINGS = (
    "use_cone_tracing",
    "cone_max_sigma",
    "cone_max_roughness",
    "cone_split",
    "use_gaussian_lod",
    "lod_base_sigma",
    "lod_num_levels",
    "lod_thin_sigma",
    "lod_cache_dir",
    "frame_id",
)


@pytest.fixture(scope="module")
def lod_cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("lod_cache")


@pytest.fixture
def lod_engine(playground_engine, lod_cache_dir):
    """The playground engine with a temporary LOD cache; LOD freed and the plain main BVH restored afterwards."""
    engine = playground_engine
    old = {name: getattr(engine, name) for name in LOD_SETTINGS}
    density = engine.scene_mog.density
    engine.lod_cache_dir = str(lod_cache_dir)
    engine.cone_max_roughness = 1.0  # the brushed copper sphere takes the cone path
    try:
        yield engine
    finally:
        for name, value in old.items():
            setattr(engine, name, value)
        edited = engine.scene_mog.density is not density
        engine.scene_mog.density = density
        engine.release_gaussian_lod()
        if engine._gs_bvh_inflate != 0.0 or edited:
            engine._gs_bvh_inflate = 0.0
            engine.tracer.build_gs_acc(gaussians=engine.scene_mog, rebuild=True, inflate=0.0)


def _slice_half(engine):
    """What a slicing plane does (ps_gui ``_recompute_slice_planes``): a new density tensor with the Gaussians on the
    camera's (+x) side of the scene center emptied."""
    mog = engine.scene_mog
    density = copy.deepcopy(mog.density)
    x = mog.positions[:, 0]
    with torch.no_grad():
        density[x > x.median()] = -100000
    mog.density = density


def _setup(engine):
    from threedgrut_playground.engine import OptixPrimitiveTypes

    configure_sphere(engine, OptixPrimitiveTypes.PBR, "brushed_copper")
    return make_pinhole_rays(*default_view(engine))


def _frame(engine, rays_o, rays_d, *, cone, lod, cap=0.005, split=0.0, frame_id=11):
    engine.use_cone_tracing, engine.use_gaussian_lod = cone, lod
    engine.cone_max_sigma, engine.cone_split, engine.frame_id = cap, split, frame_id
    out = engine._render_playground_hybrid(rays_o.clone(), rays_d.clone())
    return out["pred_features"].clone()


def _spy_render(engine, monkeypatch):
    """Record the cone and LOD parameter blocks of every ``render_playground`` call."""
    calls = []
    original = engine.tracer.render_playground

    def spy(*args, **kwargs):
        calls.append((kwargs.get("cone_params"), kwargs.get("lod_params"), kwargs.get("lod")))
        return original(*args, **kwargs)

    monkeypatch.setattr(engine.tracer, "render_playground", spy)
    return calls


def _spy_slot_builds(engine, monkeypatch, rebuilt=None):
    """Record the inflations of every ``build_lod_acc`` call (and, into ``rebuilt``, the slots it built)."""
    builds = []
    original = engine.tracer.build_lod_acc

    def spy(lod, inflations):
        builds.append(list(inflations))
        slots = original(lod, inflations)
        if rebuilt is not None:
            rebuilt.append(slots)
        return slots

    monkeypatch.setattr(engine.tracer, "build_lod_acc", spy)
    return builds


@requires_gpu_ckpt
def test_engine_lod_toggle_off_bit_identical(lod_engine):
    engine = lod_engine
    rays_o, rays_d = _setup(engine)
    snap = snapshot_engine(engine)
    fresh_off = _frame(engine, rays_o, rays_d, cone=False, lod=False)
    fresh_cone = _frame(engine, rays_o, rays_d, cone=True, lod=False)

    lod_frames = [_frame(engine, rays_o, rays_d, cone=True, lod=True, cap=cap) for cap in (0.005, 0.02, 0.005)]
    for frame in lod_frames:
        assert torch.isfinite(frame).all()
    # At cap 0.005 only slot 1 is enabled: the originals inflated by s_1 = 0.005, the same proxies as the inflated
    # main BVH, so the frame may equal the LOD-off cone frame. Cap 0.02 adds the cut slots and must change it
    print(f"LOD at cap 0.005 bit-identical to the LOD-off cone frame: {bit_equal(lod_frames[0], fresh_cone)}")
    assert not bit_equal(lod_frames[1], lod_frames[0]), "the cut slots must be traced, or the test proves nothing"
    assert bit_equal(lod_frames[2], lod_frames[0]), "rebuilding the slots for the same cap gives the same frame"
    assert engine._gs_bvh_inflate == 0.0, "with LOD on, the main BVH stays uninflated"

    restore_engine(engine, snap)
    assert bit_equal(_frame(engine, rays_o, rays_d, cone=False, lod=False), fresh_off)
    assert bit_equal(_frame(engine, rays_o, rays_d, cone=True, lod=False), fresh_cone)
    # And a cone-off frame with the LOD switch still on is plain cone off too
    assert bit_equal(_frame(engine, rays_o, rays_d, cone=False, lod=True), fresh_off)


@requires_gpu_ckpt
def test_engine_lod_split_mc_half_equals_cone_off(lod_engine):
    engine = lod_engine
    rays_o, rays_d = _setup(engine)
    half = int(round(0.5 * IMG_W))
    off = _frame(engine, rays_o, rays_d, cone=False, lod=False)
    split = _frame(engine, rays_o, rays_d, cone=True, lod=True, split=0.5)
    assert bit_equal(split[..., :half, :], off[..., :half, :])
    assert not bit_equal(split[..., half + 1 :, :], off[..., half + 1 :, :]), "the cone half must differ"
    plain_split = _frame(engine, rays_o, rays_d, cone=True, lod=False, split=0.5)
    print(f"LOD off: MC half bit-identical to cone off: {bit_equal(plain_split[..., :half, :], off[..., :half, :])}")


@requires_gpu_ckpt
def test_engine_clamps_cap_above_top_band(lod_engine, capsys, monkeypatch):
    engine = lod_engine
    rays_o, rays_d = _setup(engine)
    assert engine.lod_num_levels == 7 and engine.lod_base_sigma == 0.005
    calls = _spy_render(engine, monkeypatch)
    capsys.readouterr()
    frame = _frame(engine, rays_o, rays_d, cone=True, lod=True, cap=2.0)
    first = capsys.readouterr().out
    assert "0.64" in first and "clamp" in first.lower(), first
    assert torch.isfinite(frame).all()
    cone_params, lod_params, lod = calls[-1]
    assert cone_params[2].item() == pytest.approx(0.64)
    assert int(lod_params[1].item()) == 9
    assert engine.cone_max_sigma == 2.0, "the setting itself is kept; only the cap used for LOD frames is clamped"
    _frame(engine, rays_o, rays_d, cone=True, lod=True, cap=2.0)
    assert "clamp" not in capsys.readouterr().out.lower(), "the warning is printed once"


@requires_gpu_ckpt
def test_engine_enables_only_slots_below_cap(lod_engine, monkeypatch):
    engine = lod_engine
    rays_o, rays_d = _setup(engine)
    calls = _spy_render(engine, monkeypatch)
    builds = _spy_slot_builds(engine, monkeypatch)

    _frame(engine, rays_o, rays_d, cone=True, lod=True, cap=0.02)
    _, lod_params, lod = calls[-1]
    num_slots = int(lod_params[1].item())
    lower = lod_params[2 : 2 + num_slots].tolist()
    assert num_slots == 4
    assert lower == pytest.approx([0.0, engine.lod_thin_sigma, 0.005, 0.01])
    assert all(edge < 0.02 for edge in lower)
    assert builds == [pytest.approx([0.005, 0.01, 0.02])]

    _frame(engine, rays_o, rays_d, cone=True, lod=True, cap=0.02)
    assert len(builds) == 1, "slots are rebuilt only when the enabled set or the cap changes"
    _frame(engine, rays_o, rays_d, cone=True, lod=True, cap=0.03)
    assert builds[-1] == pytest.approx([0.005, 0.01, 0.02, 0.03]), "the top slot is inflated by the cap"
    assert int(calls[-1][1][1].item()) == 5


@pytest.mark.parametrize(
    "thin, cap, expected",
    [
        (0.002, 0.005, [0.005]),
        (0.002, 0.003, [0.003]),  # slot 1 is the top slot: inflated by the cap, not s_1
        (0.002, 0.015, [0.005, 0.01, 0.015]),
        (0.002, 0.02, [0.005, 0.01, 0.02]),  # cap on a band edge
        (0.002, 0.05, [0.005, 0.01, 0.02, 0.04, 0.05]),
        (0.002, 2.0, K7_SIGMAS),  # clamped to s_8
        (0.002, 0.001, []),  # only the main BVH
    ],
)
def test_lod_slot_inflations_top_is_cap(thin, cap, expected):
    from threedgrut_playground.lod.slots import lod_slot_inflations, plan_lod_slots

    num_slots, max_sigma = plan_lod_slots(K7_SIGMAS, thin, cap)
    inflations = lod_slot_inflations(K7_SIGMAS, num_slots, max_sigma)
    assert inflations == pytest.approx(expected)
    if inflations:
        assert inflations[-1] == pytest.approx(max_sigma), "the top enabled slot is inflated by the clamped cap"
        assert inflations[:-1] == pytest.approx(K7_SIGMAS[: num_slots - 2]), "lower slots by their band's upper edge"


@requires_gpu_ckpt
def test_engine_cap_change_rebuilds_only_changed_slots(lod_engine, monkeypatch):
    engine = lod_engine
    rays_o, rays_d = _setup(engine)
    rebuilt = []
    builds = _spy_slot_builds(engine, monkeypatch, rebuilt)
    f32 = lambda x: torch.tensor(x, dtype=torch.float32).item()

    engine.release_gaussian_lod()
    _frame(engine, rays_o, rays_d, cone=True, lod=True, cap=0.015)
    assert rebuilt[-1] == [1, 2, 3]
    assert engine.tracer.tracer_wrapper.lod_slot_inflations()[1:4] == [f32(0.005), f32(0.01), f32(0.015)]

    _frame(engine, rays_o, rays_d, cone=True, lod=True, cap=0.018)  # same band: only the top slot
    assert builds[-1] == pytest.approx([0.005, 0.01, 0.018]) and rebuilt[-1] == [3]
    _frame(engine, rays_o, rays_d, cone=True, lod=True, cap=0.018)
    assert len(builds) == 2, "the same cap rebuilds nothing"

    _frame(engine, rays_o, rays_d, cone=True, lod=True, cap=0.08)  # two more slots: the old top and the added ones
    assert builds[-1] == pytest.approx([0.005, 0.01, 0.02, 0.04, 0.08]) and rebuilt[-1] == [3, 4, 5]
    _frame(engine, rays_o, rays_d, cone=True, lod=True, cap=0.16)  # one more slot on a band edge: only the added one
    assert rebuilt[-1] == [6]

    _frame(engine, rays_o, rays_d, cone=True, lod=True, cap=0.003)  # slot 1 alone, inflated by the cap
    assert builds[-1] == pytest.approx([0.003]) and rebuilt[-1] == [1]
    assert engine.tracer.tracer_wrapper.lod_slot_inflations()[1] == f32(0.003)
    frame = _frame(engine, rays_o, rays_d, cone=True, lod=True, cap=0.003)
    assert torch.isfinite(frame).all()


@requires_gpu_ckpt
def test_engine_lod_cache_build_then_load(lod_engine, tmp_path, capsys):
    engine = lod_engine
    engine.lod_cache_dir = str(tmp_path)
    engine.release_gaussian_lod()
    capsys.readouterr()
    built = engine.ensure_gaussian_lod()
    assert "built" in capsys.readouterr().out
    assert engine.ensure_gaussian_lod() is built, "the hierarchy stays in memory"
    engine.release_gaussian_lod()
    loaded = engine.ensure_gaussian_lod()
    assert "loaded" in capsys.readouterr().out
    assert loaded.key == built.key and loaded.positions.device == built.positions.device
    assert torch.equal(loaded.particle_density(), built.particle_density())
    assert loaded.num_original == engine.scene_mog.num_gaussians


@requires_gpu_ckpt
def test_engine_lod_edited_scene_bypasses_cache(lod_engine, tmp_path, capsys):
    engine = lod_engine
    cache_dir = tmp_path / "cache"
    engine.lod_cache_dir = str(cache_dir)
    engine.release_gaussian_lod()
    clean = engine.ensure_gaussian_lod()
    (name,) = os.listdir(cache_dir)
    before = os.stat(cache_dir / name)
    original = engine.scene_mog.density

    _slice_half(engine)  # no invalidation call: replacing the density tensor is noticed on its own
    capsys.readouterr()
    edited = engine.ensure_gaussian_lod()
    out = capsys.readouterr().out
    assert "not cached" in out and "loaded" not in out, out
    assert edited is not clean
    n = edited.num_original
    assert bit_equal(edited.densities[:n], engine.scene_mog.get_density())
    assert not torch.equal(edited.densities[:n], clean.densities[:n])
    after = os.stat(cache_dir / name)
    assert os.listdir(cache_dir) == [name], "an edited scene must not be saved"
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)

    # Built from scratch with an empty cache dir: still nothing written
    empty = tmp_path / "empty"
    engine.lod_cache_dir = str(empty)
    engine.release_gaussian_lod()
    engine.ensure_gaussian_lod()
    assert not empty.exists() or os.listdir(empty) == []

    # Undoing the slice (clean values in a new tensor, as the GUI does) makes the cache usable again
    engine.lod_cache_dir = str(cache_dir)
    engine.scene_mog.density = torch.nn.Parameter(original.detach().clone(), requires_grad=original.requires_grad)
    capsys.readouterr()
    restored = engine.ensure_gaussian_lod()
    assert "loaded" in capsys.readouterr().out
    assert torch.equal(restored.particle_density(), clean.particle_density())


@requires_gpu_ckpt
def test_engine_lod_after_slice_mc_half_equals_cone_off(lod_engine):
    engine = lod_engine
    rays_o, rays_d = _setup(engine)
    half = int(round(0.5 * IMG_W))
    clean_split = _frame(engine, rays_o, rays_d, cone=True, lod=True, split=0.5)

    _slice_half(engine)
    engine.invalidate_gaussian_lod()
    off = _frame(engine, rays_o, rays_d, cone=False, lod=False)
    split = _frame(engine, rays_o, rays_d, cone=True, lod=True, split=0.5)
    assert not bit_equal(off[..., :half, :], clean_split[..., :half, :]), "the slice must be visible"
    assert bit_equal(split[..., :half, :], off[..., :half, :])
    assert not bit_equal(split[..., half + 1 :, :], clean_split[..., half + 1 :, :]), "the cone half honours it too"
