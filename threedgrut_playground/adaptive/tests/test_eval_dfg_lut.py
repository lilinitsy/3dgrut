# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The generated split-sum table matches its generator and is physically plausible."""

import importlib.util
import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
HEADER = REPO_ROOT / "threedgrut_playground/include/playground/kernels/cuda/dfgLut.cuh"


def _generator():
    spec = importlib.util.spec_from_file_location("make_dfg_lut", REPO_ROOT / "scripts/cone/make_dfg_lut.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _header_table():
    text = HEADER.read_text(encoding="utf-8")
    pairs = re.findall(r"\{([-0-9.e+]+)f, ([-0-9.e+]+)f\}", text)
    assert len(pairs) == 32 * 32
    return np.array(pairs, dtype=np.float64).reshape(32, 32, 2)


def test_header_matches_generator():
    gen = _generator()
    table = _header_table()
    for j, i in [(3, 20), (15, 31), (31, 7)]:
        expected = gen.compute_cell(i / 31.0, j / 31.0)
        np.testing.assert_allclose(table[j, i], expected, atol=1e-6, rtol=0)


def test_table_is_plausible():
    table = _header_table()
    assert (table >= 0.0).all()
    assert (table.sum(axis=-1) <= 1.0 + 1e-3).all()
