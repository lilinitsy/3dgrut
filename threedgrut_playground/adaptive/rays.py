# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flat ray batches with ``(pixel_id, sample_index)`` identities and adaptive-width launch padding."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from kaolin.render.camera import generate_pinhole_rays

from threedgrut_playground.utils.rng import rng_torch_low_discrepancy

Tensor = torch.Tensor

# Per-pixel seed hash shared with the playground's SPP low-discrepancy mode.
JITTER_HASH_X = 19349663
JITTER_HASH_Y = 96925573
# D12: odd and even samples of a pixel are two independent Owen-scrambled Sobol sequences; this salt separates them.
HALF_SALT = 0x9E3779B9
INT32_MAX = 2**31 - 1

DEFAULT_ROW_WIDTH = 4096
DEFAULT_MAX_RAYS = 2**22


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def launch_shape(n: int, row_width: int = DEFAULT_ROW_WIDTH) -> tuple[int, int]:
    """``(R, rw)`` used to launch ``n`` rays: ``R = ceil(n/row_width)``, ``rw = ceil(n/R)`` (so padding < R)."""
    if n < 1:
        raise ValueError("cannot launch an empty ray batch (n must be >= 1)")
    if row_width < 1:
        raise ValueError(f"row_width must be >= 1, got {row_width}")
    rows = _ceil_div(n, row_width)
    return rows, _ceil_div(n, rows)


def _chunk_sizes(n: int, max_rays: int) -> tuple[int, int]:
    if max_rays < 1:
        raise ValueError(f"max_rays must be >= 1, got {max_rays}")
    return n // max_rays, n % max_rays


def launch_padding(n: int, row_width: int = DEFAULT_ROW_WIDTH, max_rays: int = DEFAULT_MAX_RAYS) -> int:
    """Exact padding of tracing ``n`` rays in chunks of ``<= max_rays``: sum over chunks of ``R*rw - n_chunk``."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    full, rest = _chunk_sizes(n, max_rays)
    total = 0
    if full:
        rows, rw = launch_shape(max_rays, row_width)
        total += full * (rows * rw - max_rays)
    if rest:
        rows, rw = launch_shape(rest, row_width)
        total += rows * rw - rest
    return total


def max_padding(n: int, row_width: int = DEFAULT_ROW_WIDTH, max_rays: int = DEFAULT_MAX_RAYS) -> int:
    """Upper bound ``ceil(n/row_width) + ceil(n/max_rays)`` on :func:`launch_padding`, for budget validation."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if row_width < 1 or max_rays < 1:
        raise ValueError("row_width and max_rays must be >= 1")
    return _ceil_div(n, row_width) + _ceil_div(n, max_rays)


def drop_padding(t: Tensor, n_valid: int) -> Tensor:
    """``[1, R, rw, *C]`` launch output -> ``[n_valid, *C]`` (padding rays dropped)."""
    if t.dim() < 3 or t.shape[0] != 1:
        raise ValueError(f"expected a launch-shaped tensor [1, R, rw, ...], got {tuple(t.shape)}")
    flat = t.reshape(t.shape[1] * t.shape[2], *t.shape[3:])
    if not 0 <= n_valid <= flat.shape[0]:
        raise ValueError(f"n_valid={n_valid} out of range for {flat.shape[0]} launched rays")
    return flat[:n_valid]


@dataclass
class RayBatch:
    """Flat rays with identities.

    ``origins``/``directions`` are ``[N, 3]`` float32; ``pixel_id``/``sample_index`` are ``[N]`` int32.
    """

    origins: Tensor
    directions: Tensor
    pixel_id: Tensor
    sample_index: Tensor

    def __post_init__(self) -> None:
        for name in ("origins", "directions"):
            t = getattr(self, name)
            if t.dtype != torch.float32:
                raise ValueError(f"{name} must be float32, got {t.dtype}")
            if t.dim() != 2 or t.shape[1] != 3:
                raise ValueError(f"{name} must have shape [N, 3], got {tuple(t.shape)}")
        n = self.origins.shape[0]
        if self.directions.shape[0] != n:
            raise ValueError(f"directions has {self.directions.shape[0]} rays, origins has {n}")
        for name in ("pixel_id", "sample_index"):
            t = getattr(self, name)
            if t.dtype != torch.int32:
                raise ValueError(f"{name} must be int32, got {t.dtype}")
            if t.dim() != 1 or t.shape[0] != n:
                raise ValueError(f"{name} must have shape [{n}], got {tuple(t.shape)}")
        device = self.origins.device
        for name in ("directions", "pixel_id", "sample_index"):
            if getattr(self, name).device != device:
                raise ValueError(f"{name} is on {getattr(self, name).device}, origins on {device}")

    @property
    def num_rays(self) -> int:
        return int(self.origins.shape[0])

    def to_launch(self, row_width: int = DEFAULT_ROW_WIDTH) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
        """Pack into fresh ``[1, R, rw, ...]`` tensors; padding repeats ray ``N-1`` with its ids.

        Returns ``(ray_o, ray_d, pixel_id, sample_index, n_valid)``. Always newly allocated (the tracer overwrites
        its input rays), never aliasing this batch.
        """
        n = self.num_rays
        if n == 0:
            raise ValueError("cannot launch an empty RayBatch")
        rows, rw = launch_shape(n, row_width)
        idx = torch.arange(rows * rw, device=self.origins.device).clamp_(max=n - 1)

        def take(t: Tensor) -> Tensor:
            out = t[idx]  # advanced indexing always copies
            return out.reshape(1, rows, rw, *out.shape[1:]).contiguous()

        return take(self.origins), take(self.directions), take(self.pixel_id), take(self.sample_index), n

    def split(self, max_rays: int) -> list["RayBatch"]:
        """Consecutive chunks of ``<= max_rays`` rays, order preserved (views of this batch; empty batch -> [])."""
        if max_rays < 1:
            raise ValueError(f"max_rays must be >= 1, got {max_rays}")
        return [
            RayBatch(
                self.origins[i : i + max_rays],
                self.directions[i : i + max_rays],
                self.pixel_id[i : i + max_rays],
                self.sample_index[i : i + max_rays],
            )
            for i in range(0, self.num_rays, max_rays)
        ]


def pixel_jitter(pixel_id: Tensor, sample_index: Tensor, width: int) -> Tensor:
    """Sub-pixel offsets ``[N, 2]`` float32 in ``[0, 1)`` for rays keyed by ``(pixel_id, sample_index)``.

    Half-split Owen-scrambled Sobol (D1, D12): sample ``s`` of pixel ``p`` belongs to half ``h = s & 1``, uses scramble
    seed ``hash(p) ^ h * HALF_SALT`` and Sobol index ``s >> 1``. A pure function of its inputs; each half is stratified
    on its own and the two halves are independent.
    """
    pid = pixel_id.long()
    px, py = pid % width, pid // width
    s = sample_index.long()
    base = (px * JITTER_HASH_X + py * JITTER_HASH_Y) & 0xFFFFFFFF  # same per-pixel seed hash as SPP's LD mode
    seed = (base ^ ((s & 1) * HALF_SALT)) & 0xFFFFFFFF  # D12: own scramble per half h = s & 1
    u, v = rng_torch_low_discrepancy(s >> 1, seed)  # Owen-scrambled Sobol at index s >> 1
    return torch.stack([u, v], -1).float().clamp_(0.0, 1.0 - 2.0**-24)  # float32 rounding of x*2^-32 can hit 1.0


def _empty_batch(device: torch.device) -> RayBatch:
    return RayBatch(
        torch.empty(0, 3, dtype=torch.float32, device=device),
        torch.empty(0, 3, dtype=torch.float32, device=device),
        torch.empty(0, dtype=torch.int32, device=device),
        torch.empty(0, dtype=torch.int32, device=device),
    )


def _check_sample_index(sidx: Tensor) -> None:
    if sidx.numel() and (int(sidx.min()) < 0 or int(sidx.max()) > INT32_MAX):
        raise ValueError(
            f"sample_index outside int32 range [0, {INT32_MAX}]: min={int(sidx.min())}, max={int(sidx.max())}"
        )


def generate_rays(camera, pixel_id: Tensor, sample_index: Tensor, *, jitter: bool = True) -> RayBatch:
    """Pinhole rays for ``(pixel_id, sample_index)`` pairs; pixel ``p`` is ``(x, y) = (p % W, p // W)`` (row-major).

    Coordinates are ``x = px + u``, ``y = py + v`` with ``(u, v)`` from :func:`pixel_jitter` (D1: always jittered), or
    0.5 when ``jitter=False``, which reproduces the engine's centered pixel grid and exists only for parity tests.
    """
    device = camera.device
    if pixel_id.shape != sample_index.shape or pixel_id.dim() != 1:
        raise ValueError(
            f"pixel_id and sample_index must be 1-D tensors of the same shape, got {tuple(pixel_id.shape)} "
            f"and {tuple(sample_index.shape)}"
        )
    pid = pixel_id.to(device).long()
    sidx = sample_index.to(device).long()
    if pid.numel() == 0:
        return _empty_batch(device)
    n_pixels = int(camera.width) * int(camera.height)
    if int(pid.min()) < 0 or int(pid.max()) >= n_pixels:
        raise ValueError(f"pixel_id outside [0, {n_pixels}): min={int(pid.min())}, max={int(pid.max())}")
    _check_sample_index(sidx)
    px, py = pid % camera.width, pid // camera.width
    if jitter:
        uv = pixel_jitter(pid, sidx, camera.width)
        u, v = uv[:, 0], uv[:, 1]
    else:
        u = v = 0.5
    x = px.float() + u
    y = py.float() + v
    origins, directions = generate_pinhole_rays(camera, coords_grid=[y, x])
    return RayBatch(origins.float().contiguous(), directions.float().contiguous(), pid.int(), sidx.int())


def rays_for_counts(camera, counts: Tensor, sample_base: Tensor, *, jitter: bool = True) -> RayBatch:
    """``counts[p]`` rays for every pixel ``p`` (row-major), grouped by pixel; ray ``k`` of pixel ``p`` gets sample
    index ``sample_base[p] + k`` (absolute, so the renderer passes ``sample_offset + counter``).

    Any issued index outside ``[0, INT32_MAX]`` raises ``ValueError`` naming ``sample_index``.
    """
    device = camera.device
    n_pixels = int(camera.width) * int(camera.height)
    if tuple(counts.shape) != (n_pixels,) or tuple(sample_base.shape) != (n_pixels,):
        raise ValueError(
            f"counts and sample_base must have shape ({n_pixels},) = H*W (one entry per pixel), "
            f"got {tuple(counts.shape)} and {tuple(sample_base.shape)}"
        )
    counts = counts.to(device).long()
    sample_base = sample_base.to(device).long()
    if bool((counts < 0).any()):
        raise ValueError("counts must be non-negative")
    pid = torch.repeat_interleave(torch.arange(n_pixels, device=device), counts)  # rays grouped by pixel
    starts = torch.cumsum(counts, 0) - counts
    sidx = sample_base[pid] + (torch.arange(pid.numel(), device=device) - torch.repeat_interleave(starts, counts))
    _check_sample_index(sidx)
    return generate_rays(camera, pid, sidx, jitter=jitter)


def center_view_dirs(camera) -> Tensor:
    """Unit view directions ``[H, W, 3]`` float32 through the pixel centers (unjittered), for view-dependent terms."""
    n = int(camera.width) * int(camera.height)
    pid = torch.arange(n, device=camera.device)
    b = generate_rays(camera, pid, torch.zeros_like(pid), jitter=False)
    return b.directions.reshape(int(camera.height), int(camera.width), 3)
