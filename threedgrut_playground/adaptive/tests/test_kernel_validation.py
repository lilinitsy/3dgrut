# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the per-ray id validation helper in ``threedgrut_playground.tracer``."""

import pytest
import torch

from threedgrut_playground.tracer import _validate_ray_ids


def _ray_o() -> torch.Tensor:
    return torch.zeros(1, 4, 5, 3)


def test_validate_neither_returns_false():
    assert _validate_ray_ids(_ray_o(), None, None) is False


@pytest.mark.parametrize("which", ["pixel_only", "sample_only"])
def test_validate_only_one_raises(which):
    ids = torch.zeros(1, 4, 5, dtype=torch.int32)
    pixel_id, sample_index = (ids, None) if which == "pixel_only" else (None, ids)
    with pytest.raises(ValueError, match="both"):
        _validate_ray_ids(_ray_o(), pixel_id, sample_index)


def test_validate_non_tensor_raises():
    with pytest.raises(ValueError, match="torch.Tensor"):
        _validate_ray_ids(_ray_o(), [[0]], [[0]])


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_validate_wrong_dtype_raises(dtype):
    ids = torch.zeros(1, 4, 5, dtype=dtype)
    with pytest.raises(ValueError, match="int32"):
        _validate_ray_ids(_ray_o(), ids, ids.clone())


@pytest.mark.parametrize("shape", [(1, 4, 5, 1), (4, 5), (1, 4, 6)])
def test_validate_wrong_shape_raises(shape):
    ids = torch.zeros(shape, dtype=torch.int32)
    with pytest.raises(ValueError, match="shape"):
        _validate_ray_ids(_ray_o(), ids, ids.clone())


def test_validate_cpu_tensor_raises():
    ids = torch.zeros(1, 4, 5, dtype=torch.int32)
    with pytest.raises(ValueError, match="CUDA"):
        _validate_ray_ids(_ray_o(), ids, ids.clone())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_validate_cuda_ok_returns_true():
    ray_o = torch.zeros(1, 4, 5, 3, device="cuda")
    ids = torch.zeros(1, 4, 5, dtype=torch.int32, device="cuda")
    assert _validate_ray_ids(ray_o, ids, ids.clone()) is True
