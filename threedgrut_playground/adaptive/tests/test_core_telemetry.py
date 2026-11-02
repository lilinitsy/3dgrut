# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from threedgrut_playground.adaptive.telemetry import CostTelemetry, LaunchRecord


def test_regression_recovers_overhead_and_slope_per_kind_and_class():
    t = CostTelemetry(use_cuda_events=False)
    for n in (100, 200, 400):
        t.add_record(LaunchRecord("adaptive_mesh", 0, n, 2.0 + 0.01 * n))
        t.add_record(LaunchRecord("pilot_nonmesh", -1, n, 1.0 + 0.002 * n))
    t.add_record(LaunchRecord("pilot_mesh", -1, 300, 5.0))
    fits = t.ms_per_ray()
    assert fits["adaptive_mesh"].method == "regression"
    assert fits["adaptive_mesh"].overhead_ms == pytest.approx(2.0)
    assert fits["adaptive_mesh"].ms_per_ray == pytest.approx(0.01)
    assert fits["pilot_mesh"].method == "ratio" and fits["pilot_mesh"].ms_per_ray == pytest.approx(5.0 / 300)
    assert fits["class:nonmesh"].ms_per_ray == pytest.approx(0.002) and fits["class:mesh"].n_launches == 4


def test_launch_context_records_and_per_render_slicing():
    t = CostTelemetry(use_cuda_events=False)
    with t.launch("pilot", -1, 5):
        pass
    start = t.num_records
    with t.launch("adaptive_mesh", 0, 7):
        pass
    assert [r.n_rays for r in t.records(start)] == [7] and t.records()[0].ms >= 0
    assert t.total_ms(start) == pytest.approx(t.records(start)[0].ms)
    t.reset()
    assert t.records() == [] and t.ms_per_ray() == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_events_measure_gpu_work():
    t = CostTelemetry(use_cuda_events=True)
    x = torch.rand(2048, 2048, device="cuda")
    with t.launch("adaptive_mesh", 0, 1):
        _ = x @ x
    assert t.records()[0].ms > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_records_synchronize_once_and_nested_launches_resolve(monkeypatch):
    t = CostTelemetry(use_cuda_events=True)
    x = torch.rand(1024, 1024, device="cuda")
    with t.launch("outer", 0, 10):
        with t.launch("inner", 0, 5):
            _ = x @ x
        _ = x @ x
    calls = []
    real = torch.cuda.synchronize
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: calls.append(1) or real(*a, **k))
    recs = t.records()
    assert len(calls) == 1 and all(r.ms is not None and r.ms > 0 for r in recs)
    assert recs[0].ms >= recs[1].ms  # outer encloses inner
    t.records()
    assert len(calls) == 1  # nothing pending the second time


def test_single_distinct_n_rays_falls_back_to_ratio():
    t = CostTelemetry(use_cuda_events=False)
    for ms in (1.0, 3.0):
        t.add_record(LaunchRecord("k", 0, 100, ms))
    fit = t.ms_per_ray()["k"]
    assert fit.method == "ratio" and fit.overhead_ms is None and fit.n_launches == 2
    assert fit.ms_per_ray == pytest.approx(4.0 / 200)


def test_nested_launches_get_separate_records():
    t = CostTelemetry(use_cuda_events=False)
    with t.launch("outer", 0, 10) as outer, t.launch("inner", 0, 5) as inner:
        pass
    assert [r.kind for r in t.records()] == ["outer", "inner"]
    assert outer.ms >= inner.ms >= 0


def test_failed_launch_is_dropped():
    t = CostTelemetry(use_cuda_events=False)
    with pytest.raises(RuntimeError), t.launch("boom", 0, 3):
        raise RuntimeError("x")
    assert t.num_records == 0
    with t.launch("ok", 0, 3):
        pass
    assert [r.kind for r in t.records()] == ["ok"]


def test_records_are_a_copy_and_unresolved_ms_ignored_by_fits():
    t = CostTelemetry(use_cuda_events=False)
    t.add_record(LaunchRecord("k", 0, 10, None))
    t.add_record(LaunchRecord("k", 0, 20, float("nan")))
    t.records().clear()
    assert t.num_records == 2
    assert t.ms_per_ray() == {}
    assert t.total_ms() == 0.0
