# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests of the reference cache: key stability / sensitivity, source hashing and crash-safe cache I/O."""

import json
import math

import numpy as np
import torch
from kaolin.render.camera import Camera

from threedgrut_playground.adaptive.evaluation.reference import (
    HOST_SOURCE_FILES,
    cache_key,
    camera_descriptor,
    hash_sources,
    load_reference,
    save_reference,
)

FP = {
    "ckpt": {"path": "C:/a.pt", "mtime_ns": 1},
    "gaussians": {"n": 3, "sha": "ab"},
    "render_config": "rc",
    "meshes": "cd",
    "materials": [],
    "kernel": {"max_pbr_bounces": 15},
    "kernel_sources": "k1",
    "host_sources": "h1",
    "code": "c1",
}
PARAMS = {"spp": 256, "spp_nonmesh": 64, "pilot_spp": 4}


def _cam(w=16, h=8, eye=(0.0, 0.0, -5.0)):
    return Camera.from_args(
        eye=torch.tensor(eye),
        at=torch.zeros(3),
        up=torch.tensor([0.0, -1.0, 0.0]),
        fov=math.radians(60),
        width=w,
        height=h,
    )


def test_cache_key_stable_and_sensitive():
    cd = camera_descriptor(_cam())
    k = cache_key(FP, cd, PARAMS)
    assert k == cache_key(json.loads(json.dumps(FP)), camera_descriptor(_cam()), dict(PARAMS)) and len(k) == 16
    variants = [
        cache_key(FP, cd, {**PARAMS, "spp": 512}),
        cache_key(FP, cd, {**PARAMS, "pilot_spp": 8}),
        cache_key(FP, camera_descriptor(_cam(w=32)), PARAMS),
        cache_key(FP, camera_descriptor(_cam(eye=(0.0, -1.0, -5.0))), PARAMS),
    ]
    variants += [
        cache_key({**FP, f: "changed"}, cd, PARAMS)
        for f in ("ckpt", "gaussians", "render_config", "meshes", "kernel", "kernel_sources", "host_sources", "code")
    ]
    assert len({k, *variants}) == 13


def test_host_sources_exist_in_repo():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    assert all((repo / p).is_file() for p in HOST_SOURCE_FILES)


def test_hash_sources_tracks_file_edits(tmp_path):
    (tmp_path / "k" / "sub").mkdir(parents=True)
    (tmp_path / "k" / "sub" / "a.cuh").write_text("x")
    (tmp_path / "f.py").write_text("y")
    h1 = hash_sources([tmp_path / "k", tmp_path / "f.py"], tmp_path)
    (tmp_path / "k" / "sub" / "a.cuh").write_text("x2")
    assert hash_sources([tmp_path / "k", tmp_path / "f.py"], tmp_path) != h1


def test_save_load_roundtrip_and_misses(tmp_path):
    arrays = {
        "hdr": np.random.rand(8, 16, 3).astype(np.float32),
        "var_mean": np.zeros((8, 16), np.float32),
        "display_var_mean": np.zeros((8, 16), np.float32),
        "mesh_mask": np.zeros((8, 16), bool),
        "hi_mask": np.ones((8, 16), bool),
        "count": np.ones((8, 16), np.int32),
    }
    save_reference(tmp_path / "k1", arrays, {"key": "k1"})
    got = load_reference(tmp_path / "k1", "k1")
    assert got is not None and all(np.array_equal(got[0][n], a) for n, a in arrays.items())
    assert load_reference(tmp_path / "k1", "other") is None
    (tmp_path / "k1" / "hi_mask.npy").unlink()
    assert load_reference(tmp_path / "k1", "k1") is None
    (tmp_path / "k1" / "meta.json").unlink()
    assert load_reference(tmp_path / "k1", "k1") is None and not list((tmp_path / "k1").glob("*.tmp"))
