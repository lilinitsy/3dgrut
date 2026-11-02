# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the adaptive-sampling playground tests."""

import pytest
import torch

from threedgrut_playground.adaptive.tests.kernel_helpers import (
    build_engine,
    ensure_test_materials,
    has_gpu_and_ckpt,
    restore_engine,
    snapshot_engine,
)


@pytest.fixture(scope="session")
def _playground_engine_session():
    """One engine for the whole session. Tests use it through ``playground_engine``, which resets it."""
    if not has_gpu_and_ckpt():
        pytest.skip("needs CUDA and THREEDGRUT_TEST_CKPT")
    return build_engine()


@pytest.fixture
def playground_engine(_playground_engine_session):
    """The session engine, with scene, settings, environment and global RNG restored after each test."""
    engine = _playground_engine_session
    snap = snapshot_engine(engine)
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(0)
            ensure_test_materials(engine)
            yield engine
    finally:
        restore_engine(engine, snap)
