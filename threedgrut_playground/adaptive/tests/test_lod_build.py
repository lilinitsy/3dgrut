# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gaussian LOD hierarchy: merge rule, merge tree, cuts and the level layout (pure torch)."""

import pickle

import pytest
import torch

from threedgrut.utils.misc import quaternion_to_so3
from threedgrut_playground.lod import build_lod, build_merge_tree, merge_gaussians, select_cut

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(params=DEVICES)
def device(request):
    return torch.device(request.param)


def _identity_quats(n, device):
    q = torch.zeros(n, 4, device=device)
    q[:, 0] = 1.0
    return q


def _random_gaussians(n, device, feat_dim=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(n, 3, generator=g)
    rot = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=1)
    scl = 0.001 + 0.02 * torch.rand(n, 3, generator=g)
    dns = 0.05 + 0.95 * torch.rand(n, 1, generator=g)
    feat = torch.randn(n, feat_dim, generator=g)
    return tuple(t.to(device) for t in (pos, rot, scl, dns, feat))


def _area(scl):
    s0, s1, s2 = scl[:, 0], scl[:, 1], scl[:, 2]
    return s0 * s1 + s0 * s2 + s1 * s2


def _covariance(rot, scl):
    R = quaternion_to_so3(rot)
    return R @ torch.diag_embed(scl * scl) @ R.transpose(1, 2)


def _merge_all(pos, rot, scl, dns, feat):
    child_to_parent = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
    return merge_gaussians(pos, rot, scl, dns, feat, child_to_parent, 1)


def _ancestor_or_self_counts(tree, in_cut):
    """For every leaf, how many nodes on its path to the root are marked in ``in_cut``."""
    n = int(tree.is_leaf.sum())
    cur = torch.arange(n, device=tree.parent.device)
    counts = torch.zeros(n, dtype=torch.long, device=cur.device)
    alive = torch.ones(n, dtype=torch.bool, device=cur.device)
    while alive.any():
        counts[alive] += in_cut[cur[alive]].long()
        cur = torch.where(alive, tree.parent[cur.clamp_min(0)], cur)
        alive = alive & (cur >= 0)
    return counts


def test_merge_two_equal_gaussians_moments(device):
    pos = torch.tensor([[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0]], device=device)
    rot = _identity_quats(2, device)
    scl = torch.full((2, 3), 0.1, device=device)
    dns = torch.full((2, 1), 0.5, device=device)
    feat = torch.tensor([[1.0, 2.0, 3.0], [3.0, -2.0, 0.0]], device=device)

    m_pos, m_rot, m_scl, m_dns, m_feat = _merge_all(pos, rot, scl, dns, feat)

    torch.testing.assert_close(m_pos, torch.zeros(1, 3, device=device), atol=1e-6, rtol=0)
    expected_cov = torch.diag(torch.tensor([0.01 + 0.04, 0.01, 0.01], device=device))
    torch.testing.assert_close(_covariance(m_rot, m_scl)[0], expected_cov, atol=1e-6, rtol=0)
    expected_dns = min(1.0, 2 * 0.5 * 0.03 / float(_area(m_scl)[0]))
    torch.testing.assert_close(m_dns, torch.full((1, 1), expected_dns, device=device), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(m_feat, feat.mean(dim=0, keepdim=True), atol=1e-6, rtol=0)


def test_merge_weights_by_density_and_area(device):
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=device)
    rot = _identity_quats(2, device)
    scl = torch.tensor([[0.1, 0.2, 0.05], [0.3, 0.1, 0.1]], device=device)
    dns = torch.tensor([[0.2], [0.6]], device=device)
    feat = torch.tensor([[0.0], [1.0]], device=device)

    m_pos, _, _, _, m_feat = _merge_all(pos, rot, scl, dns, feat)

    w_raw = dns[:, 0] * _area(scl)
    w = w_raw / w_raw.sum()
    assert float(w[1]) > 0.5
    torch.testing.assert_close(m_pos[0], w[1] * pos[1], atol=1e-6, rtol=0)
    torch.testing.assert_close(m_feat[0, 0], w[1], atol=1e-6, rtol=0)


def test_merged_rotation_is_proper(device):
    num_parents, per_parent = 16, 3
    pos, rot, scl, dns, feat = _random_gaussians(num_parents * per_parent, device, seed=1)
    scl = scl * 5.0
    child_to_parent = torch.arange(num_parents, device=device).repeat_interleave(per_parent)

    m_pos, m_rot, m_scl, _, _ = merge_gaussians(pos, rot, scl, dns, feat, child_to_parent, num_parents)

    R = quaternion_to_so3(m_rot)
    torch.testing.assert_close(torch.linalg.det(R), torch.ones(num_parents, device=device), atol=1e-5, rtol=0)
    torch.testing.assert_close(m_rot.norm(dim=1), torch.ones(num_parents, device=device), atol=1e-6, rtol=0)

    # Moment matching, computed independently in float64.
    p64, d64 = pos.double(), dns.double()[:, 0]
    w_raw = d64 * _area(scl.double())
    w = (
        w_raw
        / torch.zeros(num_parents, dtype=torch.float64, device=device).index_add(0, child_to_parent, w_raw)[
            child_to_parent
        ]
    )
    mean = torch.zeros(num_parents, 3, dtype=torch.float64, device=device).index_add(
        0, child_to_parent, w[:, None] * p64
    )
    d = p64 - mean[child_to_parent]
    child_cov = _covariance(rot.double(), scl.double()) + d[:, :, None] * d[:, None, :]
    expected = torch.zeros(num_parents, 3, 3, dtype=torch.float64, device=device).index_add(
        0, child_to_parent, w[:, None, None] * child_cov
    )
    torch.testing.assert_close(m_pos.double(), mean, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(_covariance(m_rot.double(), m_scl.double()), expected, rtol=1e-4, atol=1e-8)


def test_flat_cluster_stays_flat(device):
    n = 64
    g = torch.Generator().manual_seed(2)
    pos = torch.zeros(n, 3)
    pos[:, :2] = torch.rand(n, 2, generator=g)
    pos = pos.to(device)
    rot = _identity_quats(n, device)
    scl = torch.tensor([0.05, 0.05, 1e-4], device=device).expand(n, 3).contiguous()
    dns = torch.full((n, 1), 0.8, device=device)
    feat = torch.zeros(n, 3, device=device)

    tree = build_merge_tree(pos, rot, scl, dns, feat)
    root = int(torch.nonzero(tree.parent == -1)[0])
    for merged in (_merge_all(pos, rot, scl, dns, feat)[2][0], tree.scales[root]):
        assert float(merged.min()) < 0.05 * float(merged.max())


def test_tree_is_binary_and_rooted(device):
    n = 1000
    tree = build_merge_tree(*_random_gaussians(n, device))
    num_nodes = tree.parent.shape[0]

    assert int((tree.parent == -1).sum()) == 1
    assert bool(tree.is_leaf[:n].all()) and not bool(tree.is_leaf[n:].any())
    assert bool((tree.parent[tree.parent >= 0] >= n).all())
    children = torch.bincount(tree.parent[tree.parent >= 0], minlength=num_nodes)
    assert bool((children[:n] == 0).all())
    assert bool(((children[n:] == 1) | (children[n:] == 2)).all())
    for t in (tree.positions, tree.rotations, tree.scales, tree.densities, tree.features, tree.granularity):
        assert t.shape[0] == num_nodes


def test_granularity_monotone(device):
    tree = build_merge_tree(*_random_gaussians(1000, device))
    child = torch.nonzero(tree.parent >= 0)[:, 0]
    assert bool((tree.granularity[tree.parent[child]] >= tree.granularity[child]).all())
    torch.testing.assert_close(tree.granularity[:1000], tree.scales[:1000].amax(dim=1), atol=0, rtol=0)


def test_cut_is_exact_cover(device):
    n = 1000
    tree = build_merge_tree(*_random_gaussians(n, device))
    num_nodes = tree.parent.shape[0]
    root = int(torch.nonzero(tree.parent == -1)[0])

    for s in (0.0, 0.001, 0.01, 0.1, 10.0):
        cut = select_cut(tree, s)
        in_cut = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        in_cut[cut] = True
        assert bool((_ancestor_or_self_counts(tree, in_cut) == 1).all()), s
        if s == 10.0:
            assert cut.tolist() == [root]
        if s == 0.0:
            assert cut.tolist() == list(range(n))


def test_cut_keeps_wide_leaves(device):
    n, wide = 200, 17
    pos, rot, scl, dns, feat = _random_gaussians(n, device)
    scl = torch.full_like(scl, 0.001)
    scl[wide] = 0.5
    tree = build_merge_tree(pos, rot, scl, dns, feat)

    cut = select_cut(tree, 0.01)
    assert wide in cut.tolist()
    torch.testing.assert_close(tree.scales[wide], scl[wide], atol=0, rtol=0)
    torch.testing.assert_close(tree.positions[wide], pos[wide], atol=0, rtol=0)


def test_build_lod_layout(device):
    n = 500
    pos, rot, scl, dns, feat = _random_gaussians(n, device)
    lod = build_lod(pos, rot, scl, dns, feat, key="unit")
    num_particles = lod.positions.shape[0]

    assert lod.num_original == n and lod.key == "unit"
    for got, expected in zip(
        (lod.positions, lod.rotations, lod.scales, lod.densities, lod.features), (pos, rot, scl, dns, feat)
    ):
        assert got.shape[0] == num_particles
        assert torch.equal(got[:n], expected)
    assert len(lod.slot_index) == 7
    for slots in lod.slot_index:
        assert slots.dtype == torch.int32
        assert int(slots.min()) >= 0 and int(slots.max()) < num_particles
    assert lod.level_sigmas == [0.005 * 2**k for k in range(8)]

    particle_density = lod.particle_density()
    assert particle_density.shape == (num_particles, 12)
    expected_rows = torch.concat([pos, dns, rot, scl, torch.zeros_like(dns)], dim=1)
    assert torch.equal(particle_density[:n], expected_rows)
    assert lod.particle_density() is particle_density
    assert bool((lod.densities > 0).all()) and bool((lod.densities <= 1).all())


def test_lod_pickle_drops_particle_density_cache(device):
    lod = build_lod(*_random_gaussians(100, device))
    lod.particle_density()
    restored = pickle.loads(pickle.dumps(lod))
    assert "_particle_density" not in restored.__dict__
    assert torch.equal(restored.particle_density(), lod.particle_density())


def test_degenerate_inputs_finite(device):
    n = 64
    pos = torch.full((n, 3), 0.3, device=device)
    rot = _identity_quats(n, device)
    scl = torch.full((n, 3), 1e-9, device=device)
    dns = torch.ones(n, 1, device=device)
    feat = torch.ones(n, 4, device=device)

    for out in (_merge_all(pos, rot, scl, dns, feat), build_lod(pos, rot, scl, dns, feat)):
        if isinstance(out, tuple):
            m_pos, m_rot, m_scl, m_dns, m_feat = out
        else:
            m_pos, m_rot, m_scl, m_dns, m_feat = out.positions, out.rotations, out.scales, out.densities, out.features
        for t in (m_pos, m_rot, m_scl, m_dns, m_feat):
            assert bool(torch.isfinite(t).all())
        torch.testing.assert_close(m_rot.norm(dim=1), torch.ones(m_rot.shape[0], device=device), atol=1e-6, rtol=0)
        assert bool((m_dns <= 1).all()) and bool((m_dns > 0).all())
