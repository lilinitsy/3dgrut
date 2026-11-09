# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How far can the sampling rate on mesh pixels drop before the error is visible? See evaluation/reflection_rate.py.

Usage: python scripts/adaptive/run_reflection_rate.py --ckpt <ckpt.pt> --out-dir <folder outside the repo>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from threedgrut_playground.adaptive.evaluation.reflection_rate import main

if __name__ == "__main__":
    raise SystemExit(main())
