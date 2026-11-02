# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU tests of the evaluation harness on the real checkpoint (skipped without CUDA / ``THREEDGRUT_TEST_CKPT``).

Every test takes the kernel plan's ``playground_engine`` fixture and builds its scene with ``setup_glossy_scene``.
"""

import csv
import dataclasses
import json
import math
import time
import warnings
from pathlib import Path

import pytest
import torch

from threedgrut_playground.adaptive.evaluation.core_api import (
    AdaptiveRenderer,
    SimpleBudget,
    StochasticRounder,
    UniformAllocator,
    VarianceOfMeanEstimator,
    luminance,
    make_weight_terms,
)
from threedgrut_playground.adaptive.evaluation.reference import (
    REF_MAX_SPP,
    cache_key,
    camera_descriptor,
    load_reference,
    reference_bundle,
    scene_fingerprint,
)
from threedgrut_playground.adaptive.evaluation.sampling import (
    PLUMBING_BOUNDS,
    check_plumbing,
    z_statistics,
)
from threedgrut_playground.adaptive.evaluation.scenes import (
    PRESETS,
    SceneVisibilityError,
    build_glossy_scene,
    glossy_layout,
    scene_frame,
    setup_glossy_scene,
    verify_scene_visibility,
)
from threedgrut_playground.adaptive.evaluation.sweep import main
from threedgrut_playground.adaptive.tests.kernel_helpers import resolve_ckpt

CKPT_PATH = str(resolve_ckpt()) if resolve_ckpt() else None

_lum = luminance


def _renderer(e):
    return AdaptiveRenderer(
        e,
        VarianceOfMeanEstimator(),
        make_weight_terms("cielab"),
        UniformAllocator(),
        StochasticRounder(seed=0),
        SimpleBudget(4096),
    )


def test_setup_glossy_scene_objects_materials_and_visibility(playground_engine):
    e = playground_engine
    setup_glossy_scene(e)
    setup_glossy_scene(e)  # idempotent; verification runs again
    by_id = {m.material_id: name for name, m in e.primitives.registered_materials.items()}
    got = sorted((o.geometry_type, by_id[int(o.material_id[0])]) for o in e.primitives.objects.values())
    assert got == sorted(
        [
            ("Quad", "brushed_copper"),
            ("Sphere", "rose_gold"),
            ("Sphere", "polished_marble"),
            ("Sphere", "blue_plastic"),
            ("Sphere", "oak_wood"),
        ]
    )
    assert all(int(o.primitive_type) == 4 for o in e.primitives.objects.values())


def test_meshes_in_view_but_behind_gaussians_are_rejected(playground_engine):
    """D6 occlusion, not frustum culling: the shifted meshes stay centered in the close view (visible once Gaussians
    are disabled) but sit behind the ground, so with Gaussians on the visibility check must fail."""
    e = playground_engine
    base = PRESETS["garden_glossy"]
    frame = scene_frame(e.scene_mog.positions.detach().cpu())
    fwd = glossy_layout(frame, base).cameras["close"].forward()
    buried = dataclasses.replace(base, mesh_shift=tuple((2.0 * fwd).tolist()))  # 2 spreads further along the view ray
    lay = glossy_layout(frame, buried)
    e.camera_type = "Pinhole"
    build_glossy_scene(e, lay)
    # Gaussians off: the shifted geometry is smaller but in view. 2 spreads further away it covers 0.131 (close) and
    # 0.054 (overview) of the pixels, below the preset's 0.25 close threshold, so this call uses the thresholds / 4.
    free = dataclasses.replace(buried, min_visible_fraction={k: v / 4 for k, v in base.min_visible_fraction.items()})
    e.disable_gaussian_tracing = True
    cov_free = verify_scene_visibility(e, lay, free)  # geometry is in view: passes
    assert cov_free["close"]["total"] >= base.min_visible_fraction["close"] / 4
    e.disable_gaussian_tracing = False
    with pytest.raises(SceneVisibilityError):
        verify_scene_visibility(e, lay, buried)


# ---------------------------------------------------------------------------------------------------------------------
# reference (Task 6)
# ---------------------------------------------------------------------------------------------------------------------


def test_reference_cache_hit_is_identical_and_fast(playground_engine, tmp_path):
    e = playground_engine
    cam = setup_glossy_scene(e)["close"].to_camera(96, 96)
    r = _renderer(e)
    kw = dict(ckpt_path=CKPT_PATH, spp_nonmesh=16, pilot_spp=4)
    t0 = time.perf_counter()
    a = reference_bundle(r, cam, 32, tmp_path, **kw)
    t1 = time.perf_counter()
    b = reference_bundle(r, cam, 32, tmp_path, **kw)
    t2 = time.perf_counter()
    assert torch.isfinite(a.hdr).all() and torch.equal(a.hdr.cpu(), b.hdr.cpu()) and (t2 - t1) < (t1 - t0)
    assert int(a.count[a.hi_mask].min()) == 32 and int(a.count[~a.hi_mask].min()) == 16
    assert a.mesh_mask.float().mean() > 0.2 and bool((a.hi_mask | ~a.mesh_mask).all())


def test_reference_cache_never_loads_stale_state(playground_engine, tmp_path):
    """Review Focus 3: every engine-state change that alters the image changes the key; a half-written entry, a spp
    outside the reserved sample-index range and an out-dir inside the repo are rejected."""
    e = playground_engine
    cam = setup_glossy_scene(e)["close"].to_camera(32, 32)
    r = _renderer(e)
    cd = camera_descriptor(cam)
    params = {"spp": 8, "spp_nonmesh": 4, "pilot_spp": 4}

    def key():
        return cache_key(scene_fingerprint(e, CKPT_PATH), cd, params)

    k0 = key()
    assert key() == k0
    keys = {k0}
    mat = next(iter(e.primitives.registered_materials.values()))
    rough = mat.roughness_factor
    density = e.scene_mog.density.detach().clone()
    bounces = e.max_pbr_bounces
    try:
        mat.roughness_factor = rough + 0.25
        keys.add(key())
        mat.roughness_factor = rough
        with torch.no_grad():
            e.scene_mog.density.view(-1)[0] += 1.0
        keys.add(key())
        e.scene_mog.density.data.copy_(density)
        assert key() == k0
        e.max_pbr_bounces += 1
        keys.add(key())
        e.max_pbr_bounces -= 1
        obj = next(iter(e.primitives.objects.values()))
        obj.transform.translate(torch.tensor([0.0, 0.01, 0.0], device=obj.transform.device))
        e.primitives.dirty = True
        keys.add(key())
    finally:
        mat.roughness_factor = rough
        e.scene_mog.density.data.copy_(density)
        e.max_pbr_bounces = bounces
        setup_glossy_scene(e, verify=False)  # undo the transform edit
    assert len(keys) == 5

    ref = reference_bundle(r, cam, 8, tmp_path, ckpt_path=CKPT_PATH, spp_nonmesh=4)
    (ref.cache_dir / "count.npy").write_bytes(b"truncated")  # a crash mid-write
    assert load_reference(ref.cache_dir, ref.key) is None
    again = reference_bundle(r, cam, 8, tmp_path, ckpt_path=CKPT_PATH, spp_nonmesh=4)
    assert again.key == ref.key and torch.equal(again.hdr.cpu(), ref.hdr.cpu())  # recomputed, deterministic
    assert load_reference(ref.cache_dir, ref.key) is not None
    with pytest.raises(ValueError, match="REF_MAX_SPP"):
        reference_bundle(r, cam, REF_MAX_SPP + 1, tmp_path, ckpt_path=CKPT_PATH, spp_nonmesh=4)
    with pytest.raises(ValueError, match="inside the repository"):
        reference_bundle(r, cam, 8, Path(__file__).parent, ckpt_path=CKPT_PATH, spp_nonmesh=4)


def test_plumbing_reference_vs_long_uniform_render(playground_engine, tmp_path):
    """Plumbing check (not a scientific result): reference and B's render estimate the same pixel integrals.

    The two-sided z bounds assume i.i.d. Monte-Carlo noise, which holds on mesh pixels (stochastic PBR bounces). A
    Gaussian-only pixel is deterministic per ray: its only randomness is the Owen-scrambled Sobol pixel jitter, whose
    error falls faster than the 1/n that ``variance_of_mean`` models (B's R2), so z << 1 there by design (measured
    z_median ~0.004, the same between two plain uniform renders without any reference). So the two-sided check runs
    on the mesh pixels and the non-mesh pixels get the one-sided hard check (a blend or jitter-offset bug inflates z).
    """
    e = playground_engine
    cam = setup_glossy_scene(e)["close"].to_camera(96, 96)
    r = _renderer(e)
    ref = reference_bundle(r, cam, 512, tmp_path, ckpt_path=CKPT_PATH, spp_nonmesh=512)
    res = r.render(cam, budget_rays=256 * 96 * 96, pilot_spp=4, iterations=2, region="all", sample_offset=0)
    d2 = (_lum(res.mean_hdr) - _lum(ref.hdr)).double() ** 2
    v = res.stats.variance_of_mean().double() + ref.var_mean.double()
    mesh, nonmesh = ref.mesh_mask, ~ref.mesh_mask  # one-sided check includes the dilated silhouette ring
    z_all, z_nonmesh = z_statistics(d2, v), z_statistics(d2[nonmesh], v[nonmesh])
    stats = {**z_statistics(d2[mesh], v[mesh]), "exact_mismatch": z_all["exact_mismatch"], "slope": -1.0}
    stats.update(z_mean_nonmesh=z_nonmesh["z_mean"], z_median_nonmesh=z_nonmesh["z_median"], n_mesh=int(mesh.sum()))
    assert z_nonmesh["z_mean"] <= PLUMBING_BOUNDS["z_mean_hard"][1], stats
    for msg in check_plumbing(stats):  # hard violations raise PlumbingError
        warnings.warn(msg, stacklevel=2)


def test_plumbing_uniform_error_falls_as_one_over_n(playground_engine, tmp_path):
    from threedgrut_playground.adaptive.evaluation.analysis import convergence_slope

    e = playground_engine
    cam = setup_glossy_scene(e)["close"].to_camera(96, 96)
    r = _renderer(e)
    ref = reference_bundle(r, cam, 2048, tmp_path, ckpt_path=CKPT_PATH, spp_nonmesh=64)
    n_mesh = int(ref.mesh_mask.sum())
    ns, mses, excluded = [], [], []
    for k, n in enumerate((8, 32, 128)):
        res = r.render(
            cam,
            budget_rays=4 * 96 * 96 + (n - 4) * n_mesh,
            pilot_spp=4,
            iterations=2,
            region="mesh",
            sample_offset=k * 2**24,
            defensive_fraction=0.0,
        )
        # Controller amendment: display-clipped mesh pixels get only the pilot under every allocator, so they are
        # excluded (their error does not fall with n).
        m = res.mesh_mask
        clip = res.maps[-1].get("weight_display_clip")
        if clip is not None:
            m = m & (clip.to(m.device) > 0)
        excluded.append(int(res.mesh_mask.sum()) - int(m.sum()))
        ns.append(res.counts[m].double().mean().item())
        mses.append(((_lum(res.mean_hdr) - _lum(ref.hdr))[m].double() ** 2).mean().item())
    floor = ref.var_mean[ref.mesh_mask].double().mean().item()
    stats = {"z_median": 1.0, "z_mean": 1.0, "exact_mismatch": 0.0, "slope": convergence_slope(ns, mses, floor)}
    for msg in check_plumbing(stats):
        warnings.warn(f"{msg}; ns={ns} mses={mses} floor={floor} excluded_clipped_px={excluded}", stacklevel=2)


# ---------------------------------------------------------------------------------------------------------------------
# budget sweep (Task 8)
# ---------------------------------------------------------------------------------------------------------------------


def test_sweep_smoke(playground_engine, tmp_path):
    rc = main(
        [
            "--ckpt",
            CKPT_PATH,
            "--out-dir",
            str(tmp_path),
            "--budgets",
            "8,16",
            "--allocators",
            "uniform,variance_g1,perceptual_g1,plain_uniform",
            "--seeds",
            "0,1",
            "--resolution",
            "128x96",
            "--iterations",
            "2",
            "--max-per-pixel",
            "32",
            "--ref-spp",
            "2048",
            "--ref-spp-nonmesh",
            "64",
        ],
        engine=playground_engine,
    )
    assert rc == 0
    d = next(tmp_path.glob("sweep_*"))
    with open(d / "results.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 16 and {r["sample_offset"] for r in rows} == {"0", str(2**24)}
    for r in rows:
        assert abs(float(r["budget_rel_err"])) <= 0.01 and math.isfinite(float(r["flip_mesh"]))
        assert float(r["wall_ms"]) > 0 and float(r["ppd"]) == pytest.approx(96 / 60)
    by = {(r["allocator"], r["seed"], r["budget_spp"]): r for r in rows}
    assert (
        by[("plain_uniform", "0", "8")]["ref_flagged_frac_mesh"]
        == by[("perceptual_g1", "0", "8")]["ref_flagged_frac_mesh"]
    )
    with open(d / "paired.csv", newline="") as fh:
        paired = list(csv.DictReader(fh))
    assert {(p["allocator"], p["baseline"]) for p in paired if p["metric"] == "flip_mesh"} == {
        ("variance_g1", "uniform"),
        ("perceptual_g1", "uniform"),
        ("perceptual_g1", "variance_g1"),
    }
    with open(d / "decision.json") as fh:
        decision = json.load(fh)
    assert "rule" in decision and all(v["verdict"] in {"beats", "loses", "inconclusive"} for v in decision["verdicts"])
    assert "caveats" in decision and "eligibility_note" in decision
    assert (d / "plots" / "close_flip_mesh_vs_wall_ms.png").exists()
    with open(d / "summary.json") as fh:
        summary = json.load(fh)
    assert summary["cameras"]["close"]["skipped_budgets"] == [] and "notes" in summary
