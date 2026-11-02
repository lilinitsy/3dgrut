# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Debug dumps of adaptive-sampling maps (``.npy`` plus ``.png`` previews) and component descriptions."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def describe_component(obj: Any) -> dict:
    """``{"class": name, **fields}`` for a dataclass instance, else ``{"class": name}``."""
    out = {"class": type(obj).__name__}
    if is_dataclass(obj) and not isinstance(obj, type):
        out.update(asdict(obj))
    return out


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.dtype == torch.bfloat16:
            x = x.float()
        return x.numpy()
    return np.asarray(x)


def _check_name(name: str) -> str:
    if not name or any(c in name for c in "/\\") or name in (".", ".."):
        raise ValueError(f"invalid map name {name!r}")
    return name


def _to_png(arr: np.ndarray) -> np.ndarray:
    """uint8 image for a 2-D map: bool -> 0/255 grey; else viridis over the [p1, p99] of the finite values."""
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8) * 255
    a = arr.astype(np.float64)
    finite = a[np.isfinite(a)]
    norm = np.zeros(a.shape, dtype=np.float64)
    if finite.size:
        lo, hi = np.percentile(finite, [1, 99])
        if hi > lo:
            norm = np.clip(np.nan_to_num((a - lo) / (hi - lo), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    try:
        import matplotlib

        rgba = matplotlib.colormaps["viridis"](norm)
        return (rgba[..., :3] * 255.0 + 0.5).astype(np.uint8)
    except Exception:  # matplotlib is optional
        return (norm * 255.0 + 0.5).astype(np.uint8)


def _write_png(path: Path, arr: np.ndarray) -> None:
    import imageio.v2 as imageio

    imageio.imwrite(path, _to_png(arr))


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")


class MapDumper:
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _save_map(self, directory: Path, name: str, value: Any, prefix: str = "", png: bool = True) -> None:
        arr = _to_numpy(value)
        stem = prefix + _check_name(name)
        np.save(directory / f"{stem}.npy", arr)
        if png and arr.ndim == 2:
            _write_png(directory / f"{stem}.png", arr)

    def dump_config(self, config: Any) -> Path:
        path = self.out_dir / "config.json"
        _write_json(path, config)
        return path

    def dump_iteration(self, i: int, maps: Dict[str, Any]) -> Path:
        d = self.out_dir / f"iter_{int(i):03d}"
        d.mkdir(parents=True, exist_ok=True)
        for name, value in maps.items():
            self._save_map(d, name, value)
        return d

    def dump_final(
        self,
        mean_hdr: Any,
        counts: Any,
        mesh_mask: Any,
        aovs: Dict[str, Any],
        timings: Any,
        mean_hdr_half_b: Optional[Any] = None,
    ) -> Path:
        d = self.out_dir / "final"
        d.mkdir(parents=True, exist_ok=True)
        self._save_map(d, "mean_hdr", mean_hdr, png=False)
        if mean_hdr_half_b is not None:
            self._save_map(d, "mean_hdr_half_b", mean_hdr_half_b, png=False)
        self._save_map(d, "counts", counts)
        self._save_map(d, "mesh_mask", mesh_mask, png=False)
        for name, value in aovs.items():
            self._save_map(d, name, value, prefix="aov_", png=False)
        _write_json(d / "timings.json", timings)
        return d
