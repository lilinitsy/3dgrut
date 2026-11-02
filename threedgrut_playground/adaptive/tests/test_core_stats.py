# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for threedgrut_playground.adaptive.stats."""

import numpy as np
import pytest
import torch

from threedgrut_playground.adaptive.stats import WINSOR_DELTA, SampleStats

g = torch.Generator().manual_seed(0)
COUNTS = [0, 1, 2, 5, 9, 30]


@pytest.fixture(scope="module")
def filled_data():
    st = SampleStats(2, 3, "cpu")
    pid = torch.cat([torch.full((n,), p, dtype=torch.int32) for p, n in enumerate(COUNTS)])
    rgb = torch.rand(pid.numel(), 3, generator=g) * 4.0
    st.add(pid, rgb, torch.arange(pid.numel(), dtype=torch.int32))
    return st, pid, rgb, COUNTS


def test_mean_and_unbiased_luma_variance_match_numpy(filled_data):
    st, pid, rgb, counts = filled_data
    lum = rgb.double() @ torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float64)
    for p, n in enumerate(counts):
        sel = pid == p
        exp_mean = rgb[sel].double().mean(0) if n else torch.zeros(3, dtype=torch.float64)
        torch.testing.assert_close(st.mean().reshape(6, 3)[p].double(), exp_mean, atol=1e-6, rtol=0)
        exp_var = float(np.var(lum[sel].numpy(), ddof=1)) if n >= 2 else 0.0
        assert st.variance().reshape(6)[p].item() == pytest.approx(exp_var, rel=1e-5, abs=1e-12)
        assert st.variance_of_mean().reshape(6)[p].item() == pytest.approx(
            exp_var / n if n >= 2 else 0, rel=1e-5, abs=1e-12
        )
    assert st.counts().tolist() == [[0, 1, 2], [5, 9, 30]] and st.counts().dtype == torch.int64


def test_shapes_and_dtypes(filled_data):
    st = filled_data[0]
    assert (st.height, st.width, st.num_pixels) == (2, 3, 6)
    assert st.mean().shape == (2, 3, 3) and st.mean().dtype == torch.float32
    for t in (st.variance(), st.variance_of_mean(), st.display_variance(), st.display_variance_of_mean()):
        assert t.shape == (2, 3) and t.dtype == torch.float32
    for t in (st.sum, st.sum_sq, st.lum_sum, st.lum_sum_sq, st.lum_w_sum, st.lum_w_sum_sq):
        assert t.dtype == torch.float64
    assert st.sum.shape == (6, 3) and st.lum_sum.shape == (6,)
    assert st.count.dtype == torch.int64 and st.n_nonfinite.dtype == torch.int64


def test_half_stats_partition_samples(filled_data):
    st = filled_data[0]
    a, b = st.half(0), st.half(1)
    assert torch.equal(a.count + b.count, st.count)
    torch.testing.assert_close(a.sum + b.sum, st.sum)
    torch.testing.assert_close(a.lum_sum_sq + b.lum_sum_sq, st.lum_sum_sq)
    assert [a.count[5].item(), b.count[5].item()] == [15, 15]  # pixel 5 holds global indices 17..46
    with pytest.raises(ValueError):
        a.half(0)


def test_half_requires_track_halves_and_valid_parity():
    st = SampleStats(1, 1, "cpu", track_halves=False)
    with pytest.raises(ValueError):
        st.half(0)
    with pytest.raises(ValueError, match="parity"):
        SampleStats(1, 1, "cpu").half(2)


def test_nonfinite_samples_are_dropped_and_counted():
    st = SampleStats(1, 2, "cpu")
    rgb = torch.tensor([[1.0, 1, 1], [float("nan"), 0, 0], [float("inf"), 1, 1], [3.0, 3, 3]])
    st.add(torch.tensor([0, 0, 1, 1], dtype=torch.int32), rgb, torch.arange(4, dtype=torch.int32))
    assert st.count.tolist() == [1, 1] and st.n_nonfinite.tolist() == [1, 1] and st.num_nonfinite == 2
    assert torch.isfinite(st.mean()).all() and st.half(1).n_nonfinite.tolist() == [1, 0]


def test_float64_accumulation_survives_large_offset():
    st = SampleStats(1, 1, "cpu")
    vals = (1000.0 + 0.1 * torch.randn(1000, generator=g, dtype=torch.float64)).float()
    st.add(torch.zeros(1000, dtype=torch.int32), vals[:, None].expand(-1, 3), torch.arange(1000, dtype=torch.int32))
    assert st.variance().item() == pytest.approx(float(np.var(vals.double().numpy(), ddof=1)), rel=1e-4)


def test_display_variance_is_winsorized_raw_is_kept():
    st = SampleStats(1, 1, "cpu")
    st.add(
        torch.zeros(4, dtype=torch.int32),
        torch.tensor([0.0, 3.0, 0.0, 3.0])[:, None].expand(4, 3),
        torch.arange(4, dtype=torch.int32),
    )
    assert st.variance().item() == pytest.approx(3.0)  # raw: 4 * 1.5^2 / 3
    assert st.display_variance().item() == pytest.approx(4 * 0.525**2 / 3)  # samples become {0, 1.05}
    assert st.display_variance_of_mean().item() == pytest.approx(4 * 0.525**2 / 3 / 4)
    assert st.half(0).display_variance().item() == 0.0  # half A holds two zeros


def test_winsorization_is_per_channel_before_luma():
    st = SampleStats(1, 1, "cpu")
    cap = 1.0 + WINSOR_DELTA
    rgb = torch.tensor([[10.0, -2.0, 0.5], [0.2, 0.3, 4.0]])
    st.add(torch.zeros(2, dtype=torch.int32), rgb, torch.arange(2, dtype=torch.int32))
    w = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float64)
    lw = rgb.double().clamp(0, cap) @ w
    assert st.lum_w_sum.item() == pytest.approx(lw.sum().item())
    assert st.lum_w_sum_sq.item() == pytest.approx((lw**2).sum().item())
    assert st.sum[0].tolist() == pytest.approx(rgb.double().sum(0).tolist())  # raw sums untouched


def test_winsorized_moments_match_oracle_pilot_stats():
    from threedgrut_playground.adaptive import oracle

    samples = torch.randn(7, 12, 3, generator=g) * 2.0 + 0.8
    samples[2, 3] = float("nan")
    st = SampleStats(1, 7, "cpu")
    pid = torch.arange(7, dtype=torch.int32).repeat_interleave(12)
    st.add(pid, samples.reshape(-1, 3), torch.arange(84, dtype=torch.int32))
    _, sigma, n_valid = oracle.pilot_stats(samples, winsor_delta=WINSOR_DELTA)
    assert torch.equal(st.count, n_valid)
    got = st.display_variance().reshape(-1).double().sqrt()
    torch.testing.assert_close(got, sigma, rtol=1e-5, atol=1e-6)
    _, sigma_raw, _ = oracle.pilot_stats(samples)
    torch.testing.assert_close(st.variance().reshape(-1).double().sqrt(), sigma_raw, rtol=1e-5, atol=1e-6)


def test_add_accumulates_across_calls_and_duplicate_ids():
    one = SampleStats(1, 2, "cpu")
    two = SampleStats(1, 2, "cpu")
    pid = torch.tensor([0, 1, 0, 0, 1], dtype=torch.int32)
    rgb = torch.rand(5, 3, generator=g)
    si = torch.arange(5, dtype=torch.int32)
    one.add(pid, rgb, si)
    two.add(pid[:2], rgb[:2], si[:2])
    two.add(pid[2:], rgb[2:], si[2:])
    assert torch.equal(one.count, two.count) and one.count.tolist() == [3, 2]
    torch.testing.assert_close(one.sum, two.sum)
    torch.testing.assert_close(one.half(1).sum, two.half(1).sum)


def test_add_rejects_out_of_range_pixel_ids():
    with pytest.raises(ValueError, match="pixel_id"):
        SampleStats(2, 2, "cpu").add(
            torch.tensor([4], dtype=torch.int32), torch.ones(1, 3), torch.zeros(1, dtype=torch.int32)
        )
    with pytest.raises(ValueError, match="pixel_id"):
        SampleStats(2, 2, "cpu").add(
            torch.tensor([-1], dtype=torch.int32), torch.ones(1, 3), torch.zeros(1, dtype=torch.int32)
        )


def test_add_rejects_bad_shapes():
    st = SampleStats(2, 2, "cpu")
    with pytest.raises(ValueError):
        st.add(torch.zeros(2, dtype=torch.int32), torch.ones(3, 3), torch.zeros(2, dtype=torch.int32))
    with pytest.raises(ValueError):
        st.add(torch.zeros(2, dtype=torch.int32), torch.ones(2, 4), torch.zeros(2, dtype=torch.int32))


def test_add_empty_batch_is_a_noop():
    st = SampleStats(1, 1, "cpu")
    st.add(torch.zeros(0, dtype=torch.int32), torch.zeros(0, 3), torch.zeros(0, dtype=torch.int32))
    assert st.count.tolist() == [0]


def test_empty_stats_are_zero_not_nan():
    st = SampleStats(3, 2, "cpu")
    for t in (st.mean(), st.variance(), st.variance_of_mean(), st.display_variance(), st.display_variance_of_mean()):
        assert torch.equal(t, torch.zeros_like(t))
