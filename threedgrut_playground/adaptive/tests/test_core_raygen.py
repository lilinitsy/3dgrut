# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from kaolin.render.camera import (
    Camera,
    CameraFOV,
    generate_centered_pixel_coords,
    generate_pinhole_rays,
)

from threedgrut_playground.adaptive import oracle
from threedgrut_playground.adaptive.rays import (
    HALF_SALT,
    INT32_MAX,
    JITTER_HASH_X,
    JITTER_HASH_Y,
    center_view_dirs,
    generate_rays,
    pixel_jitter,
    rays_for_counts,
)


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


def project_to_pixel(cam, d):
    """Inverse of kaolin's pinhole map: world directions [N, 3] -> continuous pixel coordinates (x, y)."""
    c = d.double() @ cam.extrinsics.R[0].double().T  # world -> camera space
    tan_h = float(cam.tan_half_fov(CameraFOV.HORIZONTAL))
    tan_v = float(cam.tan_half_fov(CameraFOV.VERTICAL))
    z = -c[:, 2]
    x = (c[:, 0] / z / tan_h + 1.0) * cam.width / 2.0
    y = (-c[:, 1] / z / tan_v + 1.0) * cam.height / 2.0
    return x, y


def _all(p):
    return torch.ones(p, dtype=torch.int64), torch.zeros(p, dtype=torch.int64)


def test_constants():
    assert (JITTER_HASH_X, JITTER_HASH_Y, HALF_SALT, INT32_MAX) == (19349663, 96925573, 0x9E3779B9, 2**31 - 1)


def test_unjittered_matches_engine_pinhole_grid(cam):
    b = rays_for_counts(cam, *_all(35), jitter=False)
    py, px = generate_centered_pixel_coords(7, 5, device=cam.device)
    o, d = generate_pinhole_rays(cam, coords_grid=[py, px])  # exactly what Engine3DGRUT._raygen_pinhole calls
    torch.testing.assert_close(b.origins, o.float(), atol=1e-6, rtol=0)
    torch.testing.assert_close(b.directions, d.float(), atol=1e-6, rtol=0)


# Values computed on CPU with utils/rng.py while writing the plan (half-split scheme; even s reproduce the
# old single-sequence values at index s>>1, odd s use the HALF_SALT-scrambled sequence).
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
def test_pixel_jitter_golden_values(pid, s, u, v):  # pins the generator plan C's reference depends on
    j = pixel_jitter(torch.tensor([pid]), torch.tensor([s]), width=7)
    assert j[0, 0].item() == pytest.approx(u, abs=1e-7) and j[0, 1].item() == pytest.approx(v, abs=1e-7)


@pytest.mark.parametrize("pid", [0, 13, 34])
@pytest.mark.parametrize("h", [0, 1])
def test_pixel_jitter_each_half_is_stratified(pid, h):
    for k0 in (0, 16, 2**29):
        s = 2 * torch.arange(k0, k0 + 16) + h  # 16 consecutive samples of half h
        j = pixel_jitter(torch.full((16,), pid), s, width=7)
        assert (j >= 0).all() and (j < 1).all()
        cells = (j[:, 0] * 4).long() * 4 + (j[:, 1] * 4).long()
        assert sorted(cells.tolist()) == list(range(16))


def test_halves_are_not_paired():
    # parity split of one Sobol sequence puts samples 2k and 2k+1 in opposite x-halves of the pixel; the
    # half-split sampler must not
    pid = torch.arange(1024).repeat_interleave(2)
    s = torch.tensor([0, 1]).repeat(1024)
    j = pixel_jitter(pid, s, width=32).reshape(1024, 2, 2)
    opposite = ((j[:, 0, 0] < 0.5) != (j[:, 1, 0] < 0.5)).double().mean().item()
    assert 0.4 < opposite < 0.6  # independent halves -> about 0.5 (was 1.0)


def test_pixel_jitter_is_a_pure_function_of_ids():
    pid = torch.tensor([3, 3, 9, 1])
    s = torch.tensor([0, 5, 2, 7])
    perm = torch.tensor([2, 0, 3, 1])
    assert torch.equal(pixel_jitter(pid, s, 7)[perm], pixel_jitter(pid[perm], s[perm], 7))


def test_jittered_rays_stay_inside_their_pixel(cam):
    b = rays_for_counts(cam, torch.full((35,), 8, dtype=torch.int64), torch.zeros(35, dtype=torch.int64))
    x, y = project_to_pixel(cam, b.directions)  # inverse of the pinhole map (cam has R from look-at)
    px, py = b.pixel_id.long() % 7, b.pixel_id.long() // 7
    assert ((x >= px - 1e-4) & (x < px + 1 + 1e-4) & (y >= py - 1e-4) & (y < py + 1 + 1e-4)).all()
    assert not torch.allclose(x - px, torch.full_like(x, 0.5))  # actually jittered


def test_rays_for_counts_layout_and_offset(cam):
    counts = torch.zeros(35, dtype=torch.int64)
    counts[[0, 2, 3]] = torch.tensor([2, 3, 1])
    base = torch.full((35,), 2**30, dtype=torch.int64)
    base[2] += 1
    b = rays_for_counts(cam, counts, base)
    assert b.pixel_id.tolist() == [0, 0, 2, 2, 2, 3]
    assert b.sample_index.tolist() == [2**30, 2**30 + 1, 2**30 + 1, 2**30 + 2, 2**30 + 3, 2**30]


@pytest.mark.parametrize("base", [2**31 - 2, -1])
def test_sample_index_out_of_int32_is_rejected(cam, base):
    counts = torch.zeros(35, dtype=torch.int64)
    counts[0] = 3
    with pytest.raises(ValueError, match="sample_index"):
        rays_for_counts(cam, counts, torch.full((35,), base, dtype=torch.int64))


def test_center_view_dirs_unit(cam):
    v = center_view_dirs(cam)
    assert v.shape == (5, 7, 3)
    torch.testing.assert_close(v.norm(dim=-1), torch.ones(5, 7))


# --- additions beyond the brief ---------------------------------------------------------------------------------------


def test_rays_batch_dtypes_and_empty(cam):
    b = rays_for_counts(cam, torch.tensor([1, 2] + [0] * 33), torch.tensor([0, 5] + [0] * 33))
    assert b.num_rays == 3
    assert b.origins.dtype == b.directions.dtype == torch.float32
    assert b.pixel_id.dtype == b.sample_index.dtype == torch.int32
    e = rays_for_counts(cam, torch.zeros(35, dtype=torch.int64), torch.zeros(35, dtype=torch.int64))
    assert e.num_rays == 0 and e.pixel_id.dtype == torch.int32


def test_counts_shape_mismatch_rejected(cam):
    with pytest.raises(ValueError, match="35"):
        rays_for_counts(cam, torch.ones(34, dtype=torch.int64), torch.zeros(34, dtype=torch.int64))
    with pytest.raises(ValueError):
        rays_for_counts(cam, torch.ones(35, dtype=torch.int64), torch.zeros(34, dtype=torch.int64))
    with pytest.raises(ValueError, match="counts"):
        rays_for_counts(cam, -torch.ones(35, dtype=torch.int64), torch.zeros(35, dtype=torch.int64))


def test_generate_rays_rejects_out_of_range_inputs(cam):
    with pytest.raises(ValueError, match="sample_index"):
        generate_rays(cam, torch.tensor([0]), torch.tensor([2**31]))
    with pytest.raises(ValueError, match="sample_index"):
        generate_rays(cam, torch.tensor([0]), torch.tensor([-1]))
    with pytest.raises(ValueError, match="pixel_id"):
        generate_rays(cam, torch.tensor([35]), torch.tensor([0]))
    with pytest.raises(ValueError, match="same shape"):
        generate_rays(cam, torch.tensor([0, 1]), torch.tensor([0]))


def test_sample_index_upper_bound_is_accepted(cam):
    b = generate_rays(cam, torch.tensor([3]), torch.tensor([INT32_MAX]))
    assert b.sample_index.tolist() == [INT32_MAX]


def test_matches_oracle_jitter_and_rays(cam):
    g = torch.Generator().manual_seed(0)
    pid = torch.randint(0, 35, (500,), generator=g)
    s = torch.randint(0, INT32_MAX, (500,), generator=g)
    assert torch.equal(pixel_jitter(pid, s, 7), oracle.pixel_jitter(pid, s, 7))
    for jitter in (True, False):
        b = generate_rays(cam, pid, s, jitter=jitter)
        o, d = oracle.oracle_rays(cam, pid, s, jitter=jitter)
        assert torch.equal(b.origins, o) and torch.equal(b.directions, d)
