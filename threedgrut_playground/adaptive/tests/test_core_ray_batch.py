# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from threedgrut_playground.adaptive import oracle
from threedgrut_playground.adaptive.rays import (
    RayBatch,
    drop_padding,
    launch_padding,
    launch_shape,
    max_padding,
)


def _batch(n: int) -> RayBatch:
    origins = torch.arange(n * 3, dtype=torch.float32).reshape(n, 3) + 0.5
    directions = torch.tensor([0.0, 0.6, 0.8]).expand(n, 3).contiguous()
    pixel_id = torch.arange(n, dtype=torch.int32)
    return RayBatch(origins=origins, directions=directions, pixel_id=pixel_id, sample_index=2 * pixel_id)


@pytest.mark.parametrize(
    "n,row,shape", [(10, 4096, (1, 1, 10)), (10, 4, (1, 3, 4)), (5000, 4096, (1, 2, 2500)), (8193, 4096, (1, 3, 2731))]
)
def test_to_launch_shapes_and_minimal_padding(n, row, shape):
    ray_o, ray_d, pid, sidx, nv = _batch(n).to_launch(row_width=row)
    assert pid.shape == shape and ray_o.shape == (*shape, 3) and nv == n
    assert ray_d.shape == (*shape, 3) and sidx.shape == shape
    assert pid.dtype == torch.int32 and sidx.dtype == torch.int32 and pid.numel() - n < shape[1]


def test_padding_replicates_last_ray_and_round_trips():
    b = _batch(10)
    ray_o, ray_d, pid, sidx, n = b.to_launch(row_width=4)
    assert torch.equal(ray_o.reshape(-1, 3)[10], b.origins[9])
    assert pid.reshape(-1)[11] == b.pixel_id[9] and sidx.reshape(-1)[11] == b.sample_index[9]
    assert torch.equal(drop_padding(ray_o, n), b.origins)
    assert torch.equal(drop_padding(pid[..., None], n), b.pixel_id[:, None])


def test_to_launch_never_aliases_inputs_even_for_exact_rows():
    b = _batch(8)
    before = b.origins.clone()
    ray_o, ray_d, *_ = b.to_launch(row_width=4)
    ray_o.fill_(float("nan"))
    ray_d.fill_(float("nan"))
    assert torch.equal(b.origins, before) and torch.isfinite(b.directions).all()


def test_split_preserves_order():
    parts = _batch(8).split(max_rays=3)
    assert [p.num_rays for p in parts] == [3, 3, 2]
    assert torch.equal(torch.cat([p.pixel_id for p in parts]), _batch(8).pixel_id)


@pytest.mark.parametrize(
    "field,value",
    [
        ("origins", torch.zeros(4, 3, dtype=torch.float64)),
        ("pixel_id", torch.zeros(4, dtype=torch.int64)),
        ("sample_index", torch.zeros(5, dtype=torch.int32)),
        ("directions", torch.zeros(4, 2)),
    ],
)
def test_rejects_bad_fields(field, value):
    kw = dict(
        origins=torch.zeros(4, 3),
        directions=torch.zeros(4, 3),
        pixel_id=torch.zeros(4, dtype=torch.int32),
        sample_index=torch.zeros(4, dtype=torch.int32),
    )
    kw[field] = value
    with pytest.raises(ValueError):
        RayBatch(**kw)


def test_empty_batch_cannot_launch():
    with pytest.raises(ValueError, match="empty"):
        _batch(0).to_launch()


@pytest.mark.parametrize(
    "n,row,maxr", [(10, 4096, 2**22), (10, 4, 2**22), (8193, 4096, 2**22), (1001, 100, 300), (0, 4096, 2**22)]
)
def test_launch_padding_matches_to_launch(n, row, maxr):
    expect = sum(_batch(c.num_rays).to_launch(row)[2].numel() - c.num_rays for c in _batch(n).split(maxr)) if n else 0
    assert launch_padding(n, row, maxr) == expect <= max_padding(n, row, maxr)
    assert launch_padding(4096, 4096) == 0 and launch_shape(5000) == (2, 2500)


@pytest.mark.parametrize("n", [1, 7, 4096, 4097, 9999, 20000])
def test_layout_matches_oracle_pack_rows(n):
    b = _batch(n)
    mine = b.to_launch()
    theirs = oracle.pack_rows(b.origins, b.directions, b.pixel_id, b.sample_index)
    for a, c in zip(mine[:4], theirs[:4]):
        assert torch.equal(a, c)
    assert mine[4] == theirs[4] == n
