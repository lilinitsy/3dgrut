# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-pixel HDR sample statistics: float64 RGB and luma moments (raw and winsorized), counts and parity halves."""

from __future__ import annotations

from typing import Optional, Union

import torch

from threedgrut_playground.adaptive.image_ops import luminance

# D13: allocation variance is computed on samples whose channels are clamped to [0, 1 + WINSOR_DELTA].
WINSOR_DELTA = 0.05


class SampleStats:
    """Streaming per-pixel statistics of HDR samples, stored flat over ``P = height * width`` pixels.

    Fields (all float64 unless noted): ``sum`` and ``sum_sq`` ``[P, 3]`` (raw RGB); ``lum_sum``, ``lum_sum_sq``
    (raw Rec.709 luma); ``lum_w_sum``, ``lum_w_sum_sq`` (luma of ``clamp(rgb, 0, 1 + WINSOR_DELTA)``, clamped per
    channel); ``count`` and ``n_nonfinite`` ``[P]`` int64. Samples with a non-finite channel are excluded from every
    moment and from ``count``; they only increment ``n_nonfinite``.

    With ``track_halves`` the statistics of samples with even / odd ``sample_index`` are also kept as child
    ``SampleStats`` (``half(0)`` / ``half(1)``), which are built with ``track_halves=False``.
    """

    def __init__(self, height: int, width: int, device: Union[str, torch.device], track_halves: bool = True):
        if height < 1 or width < 1:
            raise ValueError(f"height and width must be >= 1, got {height}x{width}")
        self._height = int(height)
        self._width = int(width)
        self.device = torch.device(device)
        p = self._height * self._width
        f64 = dict(dtype=torch.float64, device=self.device)
        i64 = dict(dtype=torch.int64, device=self.device)
        self.sum = torch.zeros(p, 3, **f64)
        self.sum_sq = torch.zeros(p, 3, **f64)
        self.lum_sum = torch.zeros(p, **f64)
        self.lum_sum_sq = torch.zeros(p, **f64)
        self.lum_w_sum = torch.zeros(p, **f64)
        self.lum_w_sum_sq = torch.zeros(p, **f64)
        self.count = torch.zeros(p, **i64)
        self.n_nonfinite = torch.zeros(p, **i64)
        self._halves: Optional[tuple[SampleStats, SampleStats]] = None
        if track_halves:
            self._halves = (
                SampleStats(height, width, self.device, track_halves=False),
                SampleStats(height, width, self.device, track_halves=False),
            )

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    @property
    def num_pixels(self) -> int:
        return self._height * self._width

    @property
    def num_nonfinite(self) -> int:
        return int(self.n_nonfinite.sum().item())

    # ------------------------------------------------------------------ accumulation

    def add(self, pixel_id: torch.Tensor, rgb_hdr: torch.Tensor, sample_index: torch.Tensor) -> None:
        """Accumulate ``N`` samples: ``pixel_id [N]`` int in ``[0, P)``, ``rgb_hdr [N, 3]``, ``sample_index [N]`` int."""
        n = pixel_id.shape[0] if pixel_id.dim() == 1 else -1
        if pixel_id.dim() != 1 or rgb_hdr.shape != (n, 3) or sample_index.shape != (n,):
            raise ValueError(
                f"expected pixel_id [N], rgb_hdr [N, 3], sample_index [N]; got {tuple(pixel_id.shape)}, "
                f"{tuple(rgb_hdr.shape)}, {tuple(sample_index.shape)}"
            )
        if n == 0:
            return
        pid = pixel_id.to(self.device).to(torch.int64)
        if int(pid.min()) < 0 or int(pid.max()) >= self.num_pixels:
            raise ValueError(f"pixel_id out of range [0, {self.num_pixels})")
        rgb = rgb_hdr.to(self.device)
        self._accumulate(pid, rgb)
        if self._halves is not None:
            parity = sample_index.to(self.device).to(torch.int64) & 1
            for h in (0, 1):
                m = parity == h
                self._halves[h]._accumulate(pid[m], rgb[m])

    def _accumulate(self, pid: torch.Tensor, rgb: torch.Tensor) -> None:
        if pid.numel() == 0:
            return
        ok = torch.isfinite(rgb).all(dim=1)
        bad = ~ok
        if bool(bad.any()):
            self.n_nonfinite.index_add_(0, pid[bad], torch.ones(int(bad.sum()), dtype=torch.int64, device=self.device))
        pid, y = pid[ok], rgb[ok].to(torch.float64)
        if pid.numel() == 0:
            return
        lum = luminance(y)
        lw = luminance(y.clamp(0.0, 1.0 + WINSOR_DELTA))
        self.sum.index_add_(0, pid, y)
        self.sum_sq.index_add_(0, pid, y * y)
        self.lum_sum.index_add_(0, pid, lum)
        self.lum_sum_sq.index_add_(0, pid, lum * lum)
        self.lum_w_sum.index_add_(0, pid, lw)
        self.lum_w_sum_sq.index_add_(0, pid, lw * lw)
        self.count.index_add_(0, pid, torch.ones(pid.numel(), dtype=torch.int64, device=self.device))

    # ------------------------------------------------------------------ queries

    def counts(self) -> torch.Tensor:
        """Finite sample counts ``[H, W]`` int64."""
        return self.count.reshape(self._height, self._width).clone()

    def mean(self) -> torch.Tensor:
        """Raw RGB mean ``[H, W, 3]`` float32 (0 where ``count == 0``)."""
        n = self.count.to(torch.float64).clamp_min(1.0)[:, None]
        m = torch.where(self.count[:, None] > 0, self.sum / n, torch.zeros_like(self.sum))
        return m.to(torch.float32).reshape(self._height, self._width, 3)

    def _unbiased_var(self, s: torch.Tensor, ss: torch.Tensor) -> torch.Tensor:
        """Unbiased variance of luma moments ``(s, ss)`` in float64 (0 where ``count < 2``), shape ``[P]``."""
        n = self.count.to(torch.float64)
        has = self.count >= 2
        n_safe = torch.where(has, n, torch.ones_like(n))
        var = ((ss - s * s / n_safe) / (n_safe - 1.0).clamp_min(1.0)).clamp_min(0.0)
        return torch.where(has, var, torch.zeros_like(var))

    def _of_mean(self, var: torch.Tensor) -> torch.Tensor:
        n_safe = self.count.to(torch.float64).clamp_min(1.0)
        return torch.where(self.count >= 2, var / n_safe, torch.zeros_like(var))

    def _hw(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float32).reshape(self._height, self._width)

    def variance(self) -> torch.Tensor:
        """Unbiased variance of one raw luma sample ``[H, W]`` float32 (0 where ``count < 2``)."""
        return self._hw(self._unbiased_var(self.lum_sum, self.lum_sum_sq))

    def variance_of_mean(self) -> torch.Tensor:
        """``variance / count`` ``[H, W]`` float32 (0 where ``count < 2``)."""
        return self._hw(self._of_mean(self._unbiased_var(self.lum_sum, self.lum_sum_sq)))

    def display_variance(self) -> torch.Tensor:
        """Unbiased variance of one winsorized luma sample (D13) ``[H, W]`` float32; used for allocation."""
        return self._hw(self._unbiased_var(self.lum_w_sum, self.lum_w_sum_sq))

    def display_variance_of_mean(self) -> torch.Tensor:
        """Winsorized ``variance / count`` (D13) ``[H, W]`` float32."""
        return self._hw(self._of_mean(self._unbiased_var(self.lum_w_sum, self.lum_w_sum_sq)))

    def half(self, parity: int) -> "SampleStats":
        """Child statistics of samples with ``sample_index & 1 == parity``. Raises ValueError on a child."""
        if self._halves is None:
            raise ValueError("half() is unavailable: these stats were built with track_halves=False")
        if parity not in (0, 1):
            raise ValueError(f"parity must be 0 or 1, got {parity!r}")
        return self._halves[parity]
