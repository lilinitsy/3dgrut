# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch cost telemetry: per-launch timings and a least-squares ``ms = overhead + slope * n_rays`` fit per kind."""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import torch


@dataclass
class LaunchRecord:
    kind: str
    iteration: int
    n_rays: int
    ms: Optional[float] = None


@dataclass(frozen=True)
class CostFit:
    overhead_ms: Optional[float]  # None for method == "ratio"
    ms_per_ray: float
    n_launches: int
    method: str  # "regression", or "ratio" when fewer than 2 distinct n_rays


class CostTelemetry:
    """Accumulates :class:`LaunchRecord` until :meth:`reset`, so fits can span several renders.

    With CUDA events the elapsed time is only resolved (one device synchronize) when records are read.
    """

    def __init__(self, use_cuda_events: Optional[bool] = None):
        if use_cuda_events is None:
            use_cuda_events = torch.cuda.is_available()
        self.use_cuda_events = bool(use_cuda_events)
        self._records: List[LaunchRecord] = []
        self._pending: List[Tuple[LaunchRecord, "torch.cuda.Event", "torch.cuda.Event"]] = []

    @property
    def num_records(self) -> int:
        return len(self._records)

    def add_record(self, rec: LaunchRecord) -> None:
        self._records.append(rec)

    def reset(self) -> None:
        self._records.clear()
        self._pending.clear()

    @contextmanager
    def launch(self, kind: str, iteration: int, n_rays: int) -> Iterator[LaunchRecord]:
        """Time the enclosed block. A block that raises leaves no record. Launches may nest."""
        rec = LaunchRecord(kind, int(iteration), int(n_rays))
        self._records.append(rec)
        try:
            if self.use_cuda_events:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                yield rec
                end.record()
                self._pending.append((rec, start, end))
            else:
                t0 = time.perf_counter()
                yield rec
                rec.ms = (time.perf_counter() - t0) * 1000.0
        except BaseException:
            self._drop(rec)
            raise

    def _drop(self, rec: LaunchRecord) -> None:
        self._records[:] = [r for r in self._records if r is not rec]
        self._pending[:] = [p for p in self._pending if p[0] is not rec]

    def _resolve(self) -> None:
        if not self._pending:
            return
        torch.cuda.synchronize()
        for rec, start, end in self._pending:
            rec.ms = float(start.elapsed_time(end))
        self._pending.clear()

    def records(self, start: int = 0) -> List[LaunchRecord]:
        """Records from index ``start`` (a list copy); resolves pending CUDA timings with a single synchronize."""
        self._resolve()
        return list(self._records[start:])

    def total_ms(self, start: int = 0) -> float:
        """Sum of resolved launch times from ``start``. Nested launches are counted in full each."""
        return float(sum(r.ms for r in self.records(start) if _valid(r)))

    def ms_per_ray(self) -> Dict[str, CostFit]:
        """Per-kind fits plus pooled ``"class:mesh"`` (kinds ending ``_mesh``) and ``"class:nonmesh"`` fits."""
        by_kind: Dict[str, List[LaunchRecord]] = {}
        for r in self.records():
            if _valid(r):
                by_kind.setdefault(r.kind, []).append(r)
        groups = dict(by_kind)
        for label, suffix in (("class:mesh", "_mesh"), ("class:nonmesh", "_nonmesh")):
            pooled = [r for kind, rs in by_kind.items() if kind.endswith(suffix) for r in rs]
            if pooled:
                groups[label] = pooled
        return {name: _fit(rs) for name, rs in groups.items()}


def _valid(r: LaunchRecord) -> bool:
    return r.ms is not None and math.isfinite(r.ms)


def _fit(recs: List[LaunchRecord]) -> CostFit:
    n = torch.tensor([r.n_rays for r in recs], dtype=torch.float64)
    ms = torch.tensor([r.ms for r in recs], dtype=torch.float64)
    if len(torch.unique(n)) >= 2:
        design = torch.stack([torch.ones_like(n), n], dim=1)
        sol = torch.linalg.lstsq(design, ms.unsqueeze(1)).solution.squeeze(1)
        return CostFit(float(sol[0]), float(sol[1]), len(recs), "regression")
    total_rays = float(n.sum())
    return CostFit(None, float(ms.sum()) / total_rays if total_rays > 0 else 0.0, len(recs), "ratio")
