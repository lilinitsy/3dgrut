# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Display-referred LDR conversion and output-path helpers for the evaluation harness."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


def to_display_ldr(hdr: torch.Tensor) -> torch.Tensor:
    """``clamp(nan_to_num(hdr, nan=0, posinf=1, neginf=0), 0, 1)`` as float32 (no exposure, no gamma)."""
    return torch.clamp(torch.nan_to_num(hdr.float(), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def count_nonfinite(hdr: torch.Tensor) -> int:
    """Number of pixels of ``hdr[H, W, 3]`` with any non-finite channel."""
    return int((~torch.isfinite(hdr)).any(dim=-1).sum().item())


def save_png(ldr: torch.Tensor, path) -> None:
    """Write an ``[H, W, 3]`` float image in [0, 1] as an 8-bit PNG."""
    from PIL import Image

    arr = (ldr.detach().float().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path)


def ensure_outside_repo(path, repo_root: Optional[Path] = None) -> Path:
    """Resolve ``path`` and reject it if it is the repository root or inside it."""
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[3]
    resolved = Path(path).resolve()
    if resolved == root or root in resolved.parents:
        raise ValueError(f"{path} is inside the repository {root}; outputs must go outside the repo")
    return resolved
