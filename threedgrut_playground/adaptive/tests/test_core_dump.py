# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass

import imageio.v2
import numpy as np
import pytest
import torch

from threedgrut_playground.adaptive.dump import MapDumper, describe_component


def test_map_dumper_layout(tmp_path):
    d = MapDumper(tmp_path / "run")
    d.dump_config({"seed": 3, "sample_offset": 0})
    assert json.loads((tmp_path / "run" / "config.json").read_text())["seed"] == 3
    maps = {
        "rates": torch.rand(4, 5),
        "mesh_mask": torch.zeros(4, 5, dtype=torch.bool),
        "flat": torch.ones(4, 5),
        "bad": torch.full((4, 5), float("nan")),
    }
    it = d.dump_iteration(0, maps)
    assert it == tmp_path / "run" / "iter_000" and np.load(it / "mesh_mask.npy").dtype == np.bool_
    for name in maps:
        img = imageio.v2.imread(it / f"{name}.png")
        assert img.shape[:2] == (4, 5) and img.dtype == np.uint8


def test_dump_final_layout(tmp_path):
    f = MapDumper(tmp_path).dump_final(
        torch.rand(4, 5, 3),
        torch.ones(4, 5, dtype=torch.long),
        torch.zeros(4, 5, dtype=torch.bool),
        {"first_hit_t": torch.zeros(4, 5, 1)},
        {"rays_launched": 20},
        mean_hdr_half_b=torch.rand(4, 5, 3),
    )
    for name in (
        "mean_hdr.npy",
        "mean_hdr_half_b.npy",
        "counts.npy",
        "counts.png",
        "mesh_mask.npy",
        "aov_first_hit_t.npy",
        "timings.json",
    ):
        assert (f / name).is_file()


def test_dump_final_without_half_b_and_non_json_timings(tmp_path):
    f = MapDumper(tmp_path).dump_final(
        torch.rand(2, 3, 3),
        torch.ones(2, 3),
        torch.zeros(2, 3, dtype=torch.bool),
        {},
        {"t": np.float32(1.5), "p": tmp_path},
    )
    assert not (f / "mean_hdr_half_b.npy").exists()
    assert "t" in json.loads((f / "timings.json").read_text())


def test_describe_component():
    @dataclass(frozen=True)
    class Foo:
        a: int = 2

    assert describe_component(Foo()) == {"class": "Foo", "a": 2} and describe_component(object()) == {"class": "object"}
    assert describe_component(Foo) == {"class": "type"}  # a dataclass *type* is not an instance


def test_png_normalization_edge_cases(tmp_path):
    ramp = torch.linspace(0, 1, 20).reshape(4, 5)
    with_inf = ramp.clone()
    with_inf[0, 0] = float("inf")
    maps = {
        "ramp": ramp,
        "inf": with_inf,
        "nan": torch.full((4, 5), float("nan")),
        "const": torch.full((4, 5), 7.0),
        "ints": torch.arange(20).reshape(4, 5),
        "half": ramp.half(),
        "mask": torch.eye(4, 5, dtype=torch.bool),
    }
    it = MapDumper(tmp_path).dump_iteration(2, maps)
    assert it.name == "iter_002"
    for name in maps:
        img = imageio.v2.imread(it / f"{name}.png")
        assert img.shape[:2] == (4, 5) and img.dtype == np.uint8
    ramp_img = imageio.v2.imread(it / "ramp.png").astype(int)
    assert len(np.unique(ramp_img.reshape(20, -1), axis=0)) > 10  # normalized to full range
    assert np.array_equal(imageio.v2.imread(it / "nan.png"), imageio.v2.imread(it / "const.png"))
    mask_img = imageio.v2.imread(it / "mask.png")
    assert set(np.unique(mask_img)) == {0, 255}
    assert mask_img.ndim == 2 or (mask_img[..., 0] == mask_img[..., 1]).all()


def test_non_2d_maps_get_npy_only(tmp_path):
    it = MapDumper(tmp_path).dump_iteration(0, {"rgb": torch.rand(4, 5, 3), "bf": torch.rand(4, 5, 1).bfloat16()})
    assert (it / "rgb.npy").is_file() and not (it / "rgb.png").exists()
    assert np.load(it / "bf.npy").dtype == np.float32 and not (it / "bf.png").exists()


def test_map_name_cannot_escape_directory(tmp_path):
    with pytest.raises(ValueError):
        MapDumper(tmp_path).dump_iteration(0, {"../evil": torch.zeros(2, 2)})
