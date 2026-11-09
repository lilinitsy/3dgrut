# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Equal-time comparison of cone tracing against Monte Carlo on the roughness ladder. See
threedgrut_playground/adaptive/evaluation/cone_equal_time.py for what each column means.

Usage: python scripts/cone/equal_time.py --ckpt <ckpt.pt> --out-dir <folder outside the repo>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from threedgrut_playground.adaptive.evaluation.cone_equal_time import main

if __name__ == "__main__":
    raise SystemExit(main())
