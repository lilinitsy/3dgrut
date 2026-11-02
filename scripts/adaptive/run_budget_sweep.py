# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adaptive-sampling budget sweep: equal-ray allocator comparisons on the glossy test scene against a cached jittered
reference, with paired 95% intervals and a pre-stated decision rule.

Run from the repository root (outputs must go outside the repository)::

    python scripts/adaptive/run_budget_sweep.py --ckpt runs/garden_3dgrt/garden-2210_125032/ckpt_last.pt \\
        --out-dir C:/Users/me/3dgrut_runs/sweep_v1 [--budgets 8,16,32,64,128] [--seeds 0,1,2,3,4] \\
        [--cameras close,overview] [--resolution 512x512]

``--help`` lists every option. Do not pass ``--dump-maps`` for timed comparisons (wall_ms then includes disk I/O).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from threedgrut_playground.adaptive.evaluation.sweep import main

if __name__ == "__main__":
    raise SystemExit(main())
