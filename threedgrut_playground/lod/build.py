# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training-free Gaussian level of detail: a binary merge tree and the cuts through it.

The merge follows Kerbl et al. 2024 (Hierarchical 3D Gaussians): children are weighted by
density x surface-area proxy, moments are matched, and the parent's density is the
area-weighted coverage clamped to 1. Level k is the cut of nodes whose extent (largest std)
is at most s_k while their parent's is larger, so every Gaussian in it is at most as wide as
the cone that reads it.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from threedgrut.export.partition import so3_to_quaternion_wxyz
from threedgrut.utils.misc import quaternion_to_so3

MIN_SCALE = 1e-7


def _area_proxy(scl: torch.Tensor) -> torch.Tensor:
    s0, s1, s2 = scl[:, 0], scl[:, 1], scl[:, 2]
    return s0 * s1 + s0 * s2 + s1 * s2


@dataclass(eq=False)
class GaussianLOD:
    """All LOD particles: the N originals first, then the interior nodes some cut uses.

    ``slot_index[k - 1]`` holds the global particle indices of merged level k (k = 1..K), and
    ``level_sigmas`` is [s_1, ..., s_{K+1}]. The particle buffer from ``particle_density()`` is
    cached on the object but never pickled, so a loaded LOD rebuilds it lazily.
    """

    positions: torch.Tensor
    rotations: torch.Tensor
    scales: torch.Tensor
    densities: torch.Tensor
    features: torch.Tensor
    num_original: int
    slot_index: list[torch.Tensor]
    level_sigmas: list[float]
    key: str = ""

    def particle_density(self) -> torch.Tensor:
        """[P, 12] particle buffer in the ``render_playground`` layout (pos, density, rot, scale, 0)."""
        cached: Optional[torch.Tensor] = self.__dict__.get("_particle_density")
        if cached is None:
            dns = self.densities
            cached = torch.concat([self.positions, dns, self.rotations, self.scales, torch.zeros_like(dns)], dim=1)
            self.__dict__["_particle_density"] = cached.contiguous()
        return self.__dict__["_particle_density"]

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_particle_density", None)
        return state


@dataclass(eq=False)
class MergeTree:
    """Binary merge tree; node i < N is input Gaussian i, interior nodes follow."""

    positions: torch.Tensor
    rotations: torch.Tensor
    scales: torch.Tensor
    densities: torch.Tensor
    features: torch.Tensor
    parent: torch.Tensor  # long, -1 at the root
    is_leaf: torch.Tensor  # bool
    granularity: torch.Tensor  # max(largest scale, granularity of the children)


@torch.no_grad()
def merge_gaussians(
    pos: torch.Tensor,
    rot: torch.Tensor,
    scl: torch.Tensor,
    dns: torch.Tensor,
    feat: torch.Tensor,
    child_to_parent: torch.Tensor,
    num_parents: int,
):
    """Merge children into ``num_parents`` parents by moment matching.

    Children are weighted by w'_i = density_i * area_i (area = s0 s1 + s0 s2 + s1 s2). The parent's
    covariance is sum w_i (Sigma_i + d_i d_i^T), decomposed into a proper rotation and scales of at
    least 1e-7; its density is min(1, sum w'_i / area_parent). Computed in float64, returned in the
    input dtype. Returns (pos [M,3], rot [M,4] wxyz, scl [M,3], dns [M,1], feat [M,F]).
    """
    dtype, device = pos.dtype, pos.device
    idx = child_to_parent.long()
    p64, s64, d64, f64 = pos.double(), scl.double(), dns.double()[:, 0], feat.double()

    w_raw = d64 * _area_proxy(s64)
    w_sum = torch.zeros(num_parents, dtype=torch.float64, device=device).index_add_(0, idx, w_raw)
    count = torch.bincount(idx, minlength=num_parents).double()
    # All-zero weights (zero density) fall back to a plain average.
    w = torch.where(w_sum[idx] > 0, w_raw / w_sum[idx].clamp_min(1e-300), 1.0 / count[idx])

    mean = torch.zeros(num_parents, 3, dtype=torch.float64, device=device).index_add_(0, idx, w[:, None] * p64)
    d = p64 - mean[idx]
    R = quaternion_to_so3(rot.double())
    child_cov = R @ torch.diag_embed(s64 * s64) @ R.transpose(1, 2) + d[:, :, None] * d[:, None, :]
    cov = torch.zeros(num_parents, 3, 3, dtype=torch.float64, device=device).index_add_(
        0, idx, w[:, None, None] * child_cov
    )
    cov = 0.5 * (cov + cov.transpose(1, 2))

    # Batched cuSOLVER eigh fails above ~30k matrices and is slow below that; LAPACK does 700k 3x3 in under 1 s.
    eigval, eigvec = (t.to(device) for t in torch.linalg.eigh(cov.cpu()))
    flip = torch.linalg.det(eigvec) < 0
    eigvec[:, :, 2] = torch.where(flip[:, None], -eigvec[:, :, 2], eigvec[:, :, 2])
    m_scl = eigval.clamp_min(0.0).sqrt().clamp_min(MIN_SCALE)
    m_rot = torch.nn.functional.normalize(so3_to_quaternion_wxyz(eigvec), dim=1)

    m_dns = (w_sum / _area_proxy(m_scl)).clamp(max=1.0)
    m_feat = torch.zeros(num_parents, f64.shape[1], dtype=torch.float64, device=device).index_add_(
        0, idx, w[:, None] * f64
    )
    return mean.to(dtype), m_rot.to(dtype), m_scl.to(dtype), m_dns[:, None].to(dtype), m_feat.to(feat.dtype)


def _kd_order(pos: torch.Tensor) -> torch.Tensor:
    """Permutation of the points by a kd median split of each segment's longest bounding-box axis."""
    n, device = pos.shape[0], pos.device
    order = torch.arange(n, device=device)
    seg = torch.zeros(n, dtype=torch.long, device=device)
    rows = torch.arange(n, device=device)
    while True:
        counts = torch.bincount(seg)
        if int(counts.max()) <= 1:
            return order
        p = pos[order]
        num_seg, seg3 = counts.shape[0], seg[:, None].expand(n, 3)
        lo = torch.full((num_seg, 3), float("inf"), dtype=p.dtype, device=device).scatter_reduce_(0, seg3, p, "amin")
        hi = torch.full((num_seg, 3), float("-inf"), dtype=p.dtype, device=device).scatter_reduce_(0, seg3, p, "amax")
        axis = (hi - lo).argmax(dim=1)
        key = p[rows, axis[seg]]
        by_key = torch.sort(key, stable=True).indices
        by_seg = torch.sort(seg[by_key], stable=True).indices
        perm = by_key[by_seg]
        order, seg = order[perm], seg[perm]
        starts = torch.cumsum(counts, 0) - counts
        upper = (rows - starts[seg]) >= (counts[seg] + 1) // 2
        label = 2 * seg + upper.long()
        seg = torch.cat([seg.new_zeros(1), torch.cumsum((label[1:] != label[:-1]).long(), 0)])


@torch.no_grad()
def build_merge_tree(pos, rot, scl, dns, feat) -> MergeTree:
    """Order the Gaussians by kd median split, then pair consecutive nodes bottom-up.

    An odd last node is carried up unchanged, so every interior node has two children and the tree
    has 2N - 1 nodes, with leaves at 0..N-1 in input order.
    """
    n, device = pos.shape[0], pos.device
    num_nodes = 2 * n - 1
    t_pos = pos.new_empty(num_nodes, 3)
    t_rot = rot.new_empty(num_nodes, 4)
    t_scl = scl.new_empty(num_nodes, 3)
    t_dns = dns.new_empty(num_nodes, 1)
    t_feat = feat.new_empty(num_nodes, feat.shape[1])
    t_pos[:n], t_rot[:n], t_scl[:n], t_dns[:n], t_feat[:n] = pos, rot, scl, dns, feat
    parent = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    granularity = scl.new_empty(num_nodes)
    granularity[:n] = scl.amax(dim=1)

    level = _kd_order(pos)
    next_id = n
    while level.shape[0] > 1:
        num_pairs = level.shape[0] // 2
        children = level[: 2 * num_pairs]
        local = torch.arange(num_pairs, device=device).repeat_interleave(2)
        merged = merge_gaussians(
            t_pos[children], t_rot[children], t_scl[children], t_dns[children], t_feat[children], local, num_pairs
        )
        ids = torch.arange(next_id, next_id + num_pairs, device=device)
        t_pos[ids], t_rot[ids], t_scl[ids], t_dns[ids], t_feat[ids] = merged
        parent[children] = ids[local]
        child_g = granularity[children].view(num_pairs, 2).amax(dim=1)
        granularity[ids] = torch.maximum(merged[2].amax(dim=1), child_g)
        next_id += num_pairs
        level = torch.cat([ids, level[2 * num_pairs :]])

    is_leaf = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    is_leaf[:n] = True
    return MergeTree(t_pos, t_rot, t_scl, t_dns, t_feat, parent, is_leaf, granularity)


def select_cut(tree: MergeTree, s: float) -> torch.Tensor:
    """Node indices n with (leaf or g(n) <= s) and (root or g(parent) > s): one per root-to-leaf path."""
    g = tree.granularity
    is_root = tree.parent < 0
    parent_g = g[tree.parent.clamp_min(0)]
    keep = (tree.is_leaf | (g <= s)) & (is_root | (parent_g > s))
    return torch.nonzero(keep)[:, 0]


@torch.no_grad()
def build_lod(pos, rot, scl, dns, feat, base_sigma=0.005, num_levels=7, key="") -> GaussianLOD:
    """Merge tree plus cuts at s_k = base_sigma * 2^(k-1), k = 1..num_levels, as one particle set."""
    n = pos.shape[0]
    tree = build_merge_tree(pos, rot, scl, dns, feat)
    level_sigmas = [base_sigma * 2**k for k in range(num_levels + 1)]
    cuts = [select_cut(tree, s) for s in level_sigmas[:num_levels]]

    used = torch.zeros(tree.parent.shape[0], dtype=torch.bool, device=pos.device)
    for cut in cuts:
        used[cut] = True
    used[:n] = False
    interior = torch.nonzero(used)[:, 0]
    remap = torch.full_like(tree.parent, -1)
    remap[:n] = torch.arange(n, device=pos.device)
    remap[interior] = torch.arange(n, n + interior.shape[0], device=pos.device)

    return GaussianLOD(
        positions=torch.cat([pos, tree.positions[interior]]),
        rotations=torch.cat([rot, tree.rotations[interior]]),
        scales=torch.cat([scl, tree.scales[interior]]),
        densities=torch.cat([dns, tree.densities[interior]]),
        features=torch.cat([feat, tree.features[interior]]),
        num_original=n,
        slot_index=[remap[cut].to(torch.int32) for cut in cuts],
        level_sigmas=level_sigmas,
        key=key,
    )
