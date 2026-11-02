# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from threedgrut_playground.adaptive.evaluation import core_api as C


def test_every_core_name_resolves():
    missing = [n for n in C.CORE_NAMES if getattr(C, n, None) is None]
    assert not missing, f"plan-B names to reconcile in core_api.py: {missing}"
    assert (C.L_PEAK, C.L_BLACK, C.MESH_VISIBLE_T, C.TOPUP_ROUNDS) == (200.0, 0.2, 0.5, 2)


def test_weight_models():
    assert C.make_weight_terms("cielab")[0].name == C.LuminanceAdaptationTerm.name
    assert C.make_weight_terms("tvi")[0].name == C.TviAdaptationTerm.name
    assert [t.name for t in C.make_weight_terms("cielab_nomask")] == [
        C.LuminanceAdaptationTerm.name,
        C.DisplayClipTerm.name,
    ]  # the masking ablation: cielab minus ContrastMaskingTerm
    with pytest.raises(ValueError):
        C.make_weight_terms("lab")


def test_flip_installed():
    import flip_evaluator  # noqa: F401  (D3)
