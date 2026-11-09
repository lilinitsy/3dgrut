# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""On-disk cache of ``GaussianLOD`` hierarchies, keyed by the checkpoint they were built from.

A file holds a plain dict of CPU tensors and builtins (never the pickled dataclass), so it loads with torch's safe
``weights_only=True`` default. Files are written to a temporary name and renamed, so a crash never leaves a partial
file under a valid key; an unreadable file is rebuilt and replaced.
"""

import hashlib
import os
from typing import Callable, Optional

import torch

from .build import GaussianLOD

DEFAULT_CACHE_DIR = "C:/Users/ville/3dgrut_runs/lod_cache"
# Bump when the build or the file layout changes, so older files are never read as current ones
CACHE_FORMAT_VERSION = 1

_TENSOR_FIELDS = ("positions", "rotations", "scales", "densities", "features")


def lod_cache_key(
    ckpt_path: str, num_gaussians: int, base_sigma: float, num_levels: int, mtime_ns: Optional[int] = None
) -> str:
    """Hex hash of the absolute checkpoint path, its mtime (``os.stat`` unless ``mtime_ns`` is given), the Gaussian
    count, both build parameters and the cache format version."""
    if mtime_ns is None:
        mtime_ns = os.stat(ckpt_path).st_mtime_ns
    parts = (
        f"v{CACHE_FORMAT_VERSION}",
        os.path.normcase(os.path.abspath(ckpt_path)),
        str(int(mtime_ns)),
        str(int(num_gaussians)),
        repr(float(base_sigma)),
        str(int(num_levels)),
    )
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:32]


def cache_path(key: str, cache_dir: str = DEFAULT_CACHE_DIR) -> str:
    return os.path.join(cache_dir, f"lod_{key}.pt")


def _to_state(lod: GaussianLOD) -> dict:
    state = {name: getattr(lod, name).detach().cpu().contiguous() for name in _TENSOR_FIELDS}
    state["slot_index"] = [t.detach().to(device="cpu", dtype=torch.int32).contiguous() for t in lod.slot_index]
    state["num_original"] = int(lod.num_original)
    state["level_sigmas"] = [float(s) for s in lod.level_sigmas]
    state["key"] = str(lod.key)
    state["format_version"] = CACHE_FORMAT_VERSION
    return state


def _from_state(state: dict, device) -> GaussianLOD:
    if state.get("format_version") != CACHE_FORMAT_VERSION:
        raise ValueError(f"cache format {state.get('format_version')}, expected {CACHE_FORMAT_VERSION}")
    tensors = {name: state[name].to(device).contiguous() for name in _TENSOR_FIELDS}
    return GaussianLOD(
        **tensors,
        num_original=int(state["num_original"]),
        slot_index=[t.to(device=device, dtype=torch.int32).contiguous() for t in state["slot_index"]],
        level_sigmas=[float(s) for s in state["level_sigmas"]],
        key=str(state["key"]),
    )


def _to_device(lod: GaussianLOD, device) -> GaussianLOD:
    """``lod`` itself if it is already on ``device``, else a moved copy."""
    moved = {name: getattr(lod, name).to(device) for name in _TENSOR_FIELDS}
    if all(moved[name] is getattr(lod, name) for name in _TENSOR_FIELDS):
        return lod
    return GaussianLOD(
        **moved,
        num_original=lod.num_original,
        slot_index=[t.to(device) for t in lod.slot_index],
        level_sigmas=list(lod.level_sigmas),
        key=lod.key,
    )


def load_or_build(
    key: str, build_fn: Callable[[], GaussianLOD], cache_dir: str = DEFAULT_CACHE_DIR, device=None
) -> GaussianLOD:
    """The ``GaussianLOD`` cached under ``key`` in ``cache_dir``, or ``build_fn()`` saved there.

    ``device`` moves the result there (loaded files are on the CPU; built ones stay where ``build_fn`` put them if
    ``device`` is None). A file that does not load, or whose key does not match, is reported and rebuilt.
    """
    path = cache_path(key, cache_dir)
    if os.path.isfile(path):
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
            if state.get("key") != key:
                raise ValueError(f"file holds key {state.get('key')!r}")
            return _from_state(state, device if device is not None else "cpu")
        except Exception as error:  # a truncated or foreign file must not stop the viewer: rebuild it
            print(f"Gaussian LOD cache file {path} is unreadable ({error}); rebuilding it")
    lod = build_fn()
    if device is not None:
        lod = _to_device(lod, device)
    os.makedirs(cache_dir, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    torch.save(_to_state(lod), tmp)
    os.replace(tmp, path)
    return lod
