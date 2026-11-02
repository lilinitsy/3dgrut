# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rounders that turn fractional ray rates into integer counts, with seeded (spatially unbiased) tie-breaking."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _sanitize(rates: torch.Tensor) -> torch.Tensor:
    """float64 copy with non-finite values set to 0 and negatives clamped to 0."""
    return torch.nan_to_num(rates.to(torch.float64), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)


def _generator(device: torch.device, seed: int, step: int) -> torch.Generator:
    """Generator on ``device`` seeded from ``(seed, step)``; the modulus keeps the value in the int64 range."""
    return torch.Generator(device=device).manual_seed((int(seed) * 1_000_003 + int(step)) % 2**63)


@dataclass(frozen=True)
class DeterministicRounder:
    """Largest-remainder rounding with an exact total ``round(sum(rates))``.

    Ties between equal fractional parts are broken by a permutation seeded from ``(seed, step)``, so they carry no
    top-of-image bias. The result is reproducible for a given ``(seed, step)`` and varies across steps.
    """

    seed: int = 0

    def round(self, rates: torch.Tensor, step: int) -> torch.Tensor:
        """Return int64 counts with the shape of ``rates`` and ``sum == round(sum(sanitized rates))``."""
        r = _sanitize(rates).reshape(-1)
        base = r.floor()
        frac = r - base
        total = int(round(float(r.sum())))
        k = max(total - int(base.sum()), 0)
        g = _generator(r.device, self.seed, step)
        perm = torch.randperm(r.numel(), generator=g, device=r.device)
        _, order = torch.sort(frac[perm], descending=True, stable=True)
        add = torch.zeros(r.numel(), dtype=torch.int64, device=r.device)
        add[perm[order[:k]]] = 1
        return (base.to(torch.int64) + add).reshape(rates.shape)


@dataclass(frozen=True)
class StochasticRounder:
    """Unbiased stochastic rounding: ``floor(r) + (u < frac(r))`` with ``u`` uniform, seeded from ``(seed, step)``."""

    seed: int = 0

    def round(self, rates: torch.Tensor, step: int) -> torch.Tensor:
        """Return int64 counts with the shape of ``rates``; each is ``floor(r)`` or ``floor(r) + 1``."""
        r = _sanitize(rates)
        base = r.floor()
        g = _generator(r.device, self.seed, step)
        u = torch.rand(r.shape, generator=g, dtype=torch.float64, device=r.device)
        return base.to(torch.int64) + (u < (r - base)).to(torch.int64)
