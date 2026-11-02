# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cached high-spp jittered HDR reference.

The reference traces the core's jittered rays (D1) through ``AdaptiveRenderer.trace`` in bounded rounds at sample
indices ``[REF_SAMPLE_BASE, REF_SAMPLE_BASE + REF_MAX_SPP]``, disjoint from every test render. Pixels on or next to a
visible mesh (D6, dilated by 1 px) get ``spp`` rays, the rest ``spp_nonmesh``. Besides the mean it stores its own
per-pixel noise (variance of the mean, raw and winsorized), so the metrics can flag pixels where reference noise is
a large share of the measured error.

The on-disk cache is keyed by everything that can change the result: checkpoint path and mtime, the Gaussian
parameters, the render config, the meshes and materials, the kernel settings, the CUDA kernel / host / Python
sources, the camera and the sampling parameters. Files are written as ``<name>.tmp`` + ``os.replace`` with
``meta.json`` last (an existing ``meta.json`` is removed first), so a crash between files never leaves a loadable
half-written entry.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from . import core_api
from .core_api import MESH_VISIBLE_T, SampleStats
from .ldr import ensure_outside_repo, save_png, to_display_ldr
from .sampling import traced_rounds, union_by_pixel, visible_mesh_hits

Tensor = torch.Tensor

REPO_ROOT = Path(__file__).resolve().parents[3]

REF_SAMPLE_BASE = 2**30
REF_CACHE_VERSION = "ref-v4"
REF_MAX_SPP = 2**20

HI_MASK_DILATION = 1

KERNEL_SOURCE_ROOTS = (
    "threedgrut_playground/src/kernels/cuda",
    "threedgrut_playground/include/playground/kernels/cuda",
    "threedgrt_tracer/include/3dgrt/kernels",
)
HOST_SOURCE_FILES = (
    "threedgrut_playground/src/hybridTracer.cpp",
    "threedgrut_playground/src/meshBuffers.cu",
    "threedgrut_playground/include/playground/pipelineParameters.h",
    "threedgrt_tracer/include/3dgrt/particleDensity.h",
    "threedgrut_playground/tracer.py",
)


def _source_file(obj) -> str:
    return Path(inspect.getsourcefile(obj)).resolve().relative_to(REPO_ROOT).as_posix()


# Python sources that shape the reference's samples: the core's ray generation (and its jitter RNG), trace path,
# statistics and luma, plus this harness's sampling, LDR-preview and reference modules.
CODE_FILES = tuple(
    dict.fromkeys(
        [
            _source_file(core_api.rays_for_counts),
            _source_file(core_api.AdaptiveRenderer),
            _source_file(core_api.SampleStats),
            _source_file(core_api.luminance),
            _source_file(inspect.getmodule(core_api.rays_for_counts).rng_torch_low_discrepancy),
            _source_file(traced_rounds),
            _source_file(to_display_ldr),  # ldr.png preview
            Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(),
        ]
    )
)

REF_ARRAYS = ("hdr", "var_mean", "display_var_mean", "mesh_mask", "hi_mask", "count")
_REF_DTYPES = {
    "hdr": np.float32,
    "var_mean": np.float32,
    "display_var_mean": np.float32,
    "mesh_mask": np.bool_,
    "hi_mask": np.bool_,
    "count": np.int32,
}


# ---------------------------------------------------------------------------------------------------------------------
# cache key
# ---------------------------------------------------------------------------------------------------------------------


def _iter_source_files(paths: Iterable, base: Path):
    for p in paths:
        p = Path(p)
        if not p.is_absolute():
            p = base / p
        if p.is_dir():
            for f in p.rglob("*"):
                if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc":
                    yield f
        elif p.is_file():
            yield p
        else:
            raise FileNotFoundError(f"hash_sources: {p} is neither a file nor a directory")


def _rel_posix(path: Path, base: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def hash_sources(paths: Iterable, base) -> str:
    """16 hex of sha256 over sorted ``(relative posix path, bytes)`` of every file under each directory (recursive)
    or of each file in ``paths`` (relative paths resolve against ``base``). A missing path raises."""
    base = Path(base).resolve()
    entries = sorted({_rel_posix(f, base): f for f in _iter_source_files(paths, base)}.items())
    h = hashlib.sha256()
    for rel, f in entries:
        data = f.read_bytes()
        rel_b = rel.encode("utf-8")
        h.update(len(rel_b).to_bytes(8, "little") + rel_b + len(data).to_bytes(8, "little"))
        h.update(data)
    return h.hexdigest()[:16]


def _r6(x: float) -> float:
    return round(float(x), 6) + 0.0  # + 0.0 folds -0.0 into 0.0


def camera_descriptor(camera) -> Dict[str, Any]:
    """Everything of a kaolin pinhole camera that shapes its rays, rounded to 6 decimals."""
    view = camera.view_matrix().detach().reshape(-1, 4, 4)[0].to("cpu", torch.float64)
    return {
        "width": int(camera.width),
        "height": int(camera.height),
        "fov_y_deg": _r6(camera.fov(in_degrees=True).reshape(-1)[0]),
        "x0": _r6(camera.x0.reshape(-1)[0]),
        "y0": _r6(camera.y0.reshape(-1)[0]),
        "view_matrix": [[_r6(v) for v in row] for row in view.tolist()],
    }


def _tensor_sha(h, t: Optional[Tensor]) -> None:
    if t is None:
        h.update(b"none")
        return
    t = t.detach().contiguous().cpu()
    h.update(f"{t.dtype}{tuple(t.shape)}".encode())
    h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")


def _sha_of(tensors: Iterable[Optional[Tensor]]) -> str:
    h = hashlib.sha256()
    for t in tensors:
        _tensor_sha(h, t)
    return h.hexdigest()


def _floats(x) -> Any:
    if x is None:
        return None
    if isinstance(x, Tensor):
        return [float(v) for v in x.detach().reshape(-1).cpu().tolist()]
    return float(x)


def scene_fingerprint(engine, ckpt_path: str) -> Dict[str, Any]:
    """Everything of the engine state (and the code) that can change the reference image."""
    from omegaconf import OmegaConf

    ckpt = Path(ckpt_path).resolve()
    mog = engine.scene_mog
    prims = engine.primitives
    env = engine.environment

    background = getattr(mog, "background", None)
    bg_state = background.state_dict() if isinstance(background, torch.nn.Module) else {}
    gaussians = {
        "n": int(mog.positions.shape[0]),
        "sha": _sha_of(
            [mog.positions, mog.density, mog.rotation, mog.scale, mog.features_albedo, mog.features_specular]
            + [bg_state[k] for k in sorted(bg_state)]
        ),
        "n_active_features": int(getattr(mog, "n_active_features", -1)),
    }

    prims.rebuild_bvh_if_needed()
    sf = prims.stacked_fields
    meshes = (
        "none"
        if sf is None
        else _sha_of(
            [
                sf.vertices,
                sf.triangles,
                sf.material_id,
                sf.primitive_type_tensor,
                sf.vertex_normals,
                sf.vertex_tangents,
                sf.has_tangents,
                sf.material_uv,
                sf.refractive_index_tensor,
            ]
        )
    )

    materials = []
    for m in sorted(prims.registered_materials.values(), key=lambda m: m.material_id):
        materials.append(
            [
                int(m.material_id),
                _floats(m.metallic_factor),
                _floats(m.roughness_factor),
                _floats(m.transmission_factor),
                _floats(m.ior),
                _floats(m.diffuse_factor),
                _floats(m.emissive_factor),
                int(m.alpha_mode),
                _floats(m.alpha_cutoff),
                _sha_of([m.diffuse_map, m.emissive_map, m.metallic_roughness_map, m.normal_map])[:16],
            ]
        )

    envmap = env.get_envmap()
    kernel = {
        "max_pbr_bounces": int(engine.max_pbr_bounces),
        "use_smooth_normals": bool(prims.use_smooth_normals),
        "disable_pbr_textures": bool(prims.disable_pbr_textures),
        "disable_gaussian_tracing": bool(engine.disable_gaussian_tracing),
        "envmap": str(env.current_name),
        "envmap_offset": _floats(env.get_envmap_offset()),
        "envmap_sha": _sha_of([envmap])[:16],
        "camera_type": str(engine.camera_type),
        "primitives_enabled": bool(prims.enabled),
    }

    return {
        "ckpt": {"path": ckpt.as_posix(), "mtime_ns": int(os.stat(ckpt).st_mtime_ns)},
        "gaussians": gaussians,
        "render_config": hashlib.sha256(OmegaConf.to_yaml(mog.conf).encode("utf-8")).hexdigest(),
        "meshes": meshes,
        "materials": materials,
        "kernel": kernel,
        "kernel_sources": hash_sources(KERNEL_SOURCE_ROOTS, REPO_ROOT),
        "host_sources": hash_sources(HOST_SOURCE_FILES, REPO_ROOT),
        "code": hash_sources(CODE_FILES, REPO_ROOT),
    }


def _key_dict(fingerprint: Dict[str, Any], camera_desc: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    return {"version": REF_CACHE_VERSION, "fingerprint": fingerprint, "camera": camera_desc, "params": params}


def cache_key(fingerprint: Dict[str, Any], camera_desc: Dict[str, Any], params: Dict[str, Any]) -> str:
    """16 hex of sha256 of the sorted JSON of ``{version, fingerprint, camera, params}``."""
    blob = json.dumps(_key_dict(fingerprint, camera_desc, params), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------------------------------
# cache I/O
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class ReferenceBundle:
    """A reference image with its own noise estimate (tensors on the engine device).

    Attributes:
        hdr: ``[H, W, 3]`` float32 mean of every finite sample, unclamped HDR.
        var_mean: ``[H, W]`` float32 variance of the mean of raw luma (reporting and reference-noise flags).
        display_var_mean: ``[H, W]`` float32 winsorized variance of the mean (D13), the weights' ``noise_var``.
        mesh_mask: ``[H, W]`` bool D6 visible-mesh union over the pilot samples.
        hi_mask: ``[H, W]`` bool ``mesh_mask`` dilated by 1 px (pixels that got ``spp`` rays).
        count: ``[H, W]`` int32 rays issued per pixel (non-finite samples included).
        key: cache key; meta: ``meta.json`` contents; cache_dir: the entry's directory.
    """

    hdr: Tensor
    var_mean: Tensor
    display_var_mean: Tensor
    mesh_mask: Tensor
    hi_mask: Tensor
    count: Tensor
    key: str
    meta: Dict[str, Any]
    cache_dir: Path


def _fsync_replace(tmp: Path, final: Path) -> None:
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())
    os.replace(tmp, final)


def save_reference(cache_dir, arrays: Dict[str, np.ndarray], meta: Dict[str, Any]) -> None:
    """Write ``<name>.npy`` per array (and ``ldr.png`` from ``hdr``), each as ``<file>.tmp`` + ``os.replace``, then
    ``meta.json`` last. An existing ``meta.json`` is removed first, so an interrupted write never loads."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "meta.json"
    if meta_path.exists():
        meta_path.unlink()
    for name, arr in arrays.items():
        tmp = cache_dir / f"{name}.npy.tmp"
        with open(tmp, "wb") as f:
            np.save(f, np.ascontiguousarray(arr), allow_pickle=False)
        _fsync_replace(tmp, cache_dir / f"{name}.npy")
    if "hdr" in arrays:
        tmp = cache_dir / "ldr.tmp.png"  # PIL picks the format from the extension
        save_png(to_display_ldr(torch.from_numpy(np.asarray(arrays["hdr"]))), tmp)
        _fsync_replace(tmp, cache_dir / "ldr.png")
    tmp = cache_dir / "meta.json.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    _fsync_replace(tmp, meta_path)


def load_reference(cache_dir, key: str) -> Optional[Tuple[Dict[str, np.ndarray], Dict[str, Any]]]:
    """``(arrays, meta)`` of a complete entry for ``key``; ``None`` if ``meta.json`` is missing or unreadable, its
    key differs, or any of :data:`REF_ARRAYS` is missing, unreadable, mis-typed or mis-shaped."""
    cache_dir = Path(cache_dir)
    try:
        with open(cache_dir / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict) or meta.get("key") != key:
        return None
    arrays: Dict[str, np.ndarray] = {}
    for name in REF_ARRAYS:
        try:
            arrays[name] = np.load(cache_dir / f"{name}.npy", allow_pickle=False)
        except (OSError, ValueError, EOFError):
            return None
        if arrays[name].dtype != _REF_DTYPES[name]:
            return None
    hdr = arrays["hdr"]
    if hdr.ndim != 3 or hdr.shape[2] != 3:
        return None
    if any(arrays[n].shape != hdr.shape[:2] for n in REF_ARRAYS[1:]):
        return None
    return arrays, meta


# ---------------------------------------------------------------------------------------------------------------------
# reference rendering
# ---------------------------------------------------------------------------------------------------------------------


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _bundle(arrays: Dict[str, Any], key: str, meta: Dict[str, Any], cache_dir: Path, device) -> ReferenceBundle:
    t = {n: torch.as_tensor(arrays[n]).to(device) for n in REF_ARRAYS}
    return ReferenceBundle(**t, key=key, meta=meta, cache_dir=cache_dir)


def _validate(renderer, spp: int, spp_nonmesh: int, pilot_spp: int) -> None:
    if not 4 <= pilot_spp <= spp_nonmesh <= spp <= REF_MAX_SPP:
        raise ValueError(
            f"reference needs 4 <= pilot_spp <= spp_nonmesh <= spp <= REF_MAX_SPP={REF_MAX_SPP} (its sample indices "
            f"must stay in [{REF_SAMPLE_BASE}, {REF_SAMPLE_BASE + REF_MAX_SPP}]); got pilot_spp={pilot_spp}, "
            f"spp_nonmesh={spp_nonmesh}, spp={spp}"
        )
    if renderer.engine.camera_type != "Pinhole":
        raise ValueError(
            f"reference supports only engine.camera_type == 'Pinhole', got {renderer.engine.camera_type!r}"
        )


@torch.no_grad()
def reference_bundle(
    renderer,
    camera,
    spp: int,
    out_dir,
    *,
    ckpt_path: str,
    spp_nonmesh: int = 256,
    pilot_spp: int = 4,
) -> ReferenceBundle:
    """The cached jittered reference of ``camera``: ``spp`` rays on mesh pixels and their 1-px dilation, ``spp_nonmesh``
    elsewhere, at sample indices ``REF_SAMPLE_BASE + [0, count)`` per pixel. Returns the cached entry under
    ``out_dir/references/<key>`` when one exists for the current engine state, code and arguments."""
    spp, spp_nonmesh, pilot_spp = int(spp), int(spp_nonmesh), int(pilot_spp)
    _validate(renderer, spp, spp_nonmesh, pilot_spp)
    out_dir = ensure_outside_repo(out_dir)
    engine = renderer.engine
    device = torch.device(engine.device)
    camera = camera.to(device)
    height, width = int(camera.height), int(camera.width)
    n_pixels = height * width

    params = {
        "spp": spp,
        "spp_nonmesh": spp_nonmesh,
        "pilot_spp": pilot_spp,
        "MESH_VISIBLE_T": float(MESH_VISIBLE_T),
        "REF_SAMPLE_BASE": REF_SAMPLE_BASE,
        "hi_mask_dilation": HI_MASK_DILATION,
    }
    fingerprint = scene_fingerprint(engine, ckpt_path)
    camera_desc = camera_descriptor(camera)
    key = cache_key(fingerprint, camera_desc, params)
    cache_dir = out_dir / "references" / key
    hit = load_reference(cache_dir, key)
    if hit is not None and hit[0]["hdr"].shape == (height, width, 3):
        return _bundle(hit[0], key, hit[1], cache_dir, device)

    _sync(device)
    t0 = time.perf_counter()
    stats = SampleStats(height, width, device, track_halves=False)
    base = torch.full((n_pixels,), REF_SAMPLE_BASE, dtype=torch.int64, device=device)

    # 1. pilot over all pixels; D6 mesh mask = union of visible mesh hits over these samples
    visible = torch.zeros(n_pixels, dtype=torch.bool, device=device)
    pilot = torch.full((n_pixels,), pilot_spp, dtype=torch.int64, device=device)
    for batch, out in traced_rounds(renderer, camera, pilot, base, kind="reference_pilot"):
        pid = batch.pixel_id.long()
        stats.add(pid, out["rgb"], batch.sample_index.long())
        visible |= union_by_pixel(pid, visible_mesh_hits(out), n_pixels)
    mesh_mask = visible.reshape(height, width)

    # 2. 1-px dilation of the mesh mask
    k = 2 * HI_MASK_DILATION + 1
    hi_mask = F.max_pool2d(mesh_mask.float()[None, None], k, stride=1, padding=HI_MASK_DILATION)[0, 0] > 0

    # 3. fill to spp on hi_mask, spp_nonmesh elsewhere
    target = torch.where(hi_mask.reshape(-1), spp, spp_nonmesh).to(torch.int64)
    for batch, out in traced_rounds(renderer, camera, target - pilot_spp, base, kind="reference"):
        stats.add(batch.pixel_id.long(), out["rgb"], batch.sample_index.long())
    issued = base - REF_SAMPLE_BASE
    if not torch.equal(issued, target) or int(base.max()) > REF_SAMPLE_BASE + REF_MAX_SPP:
        raise RuntimeError(
            f"reference issued sample indices up to {int(base.max()) - 1} (expected per-pixel counts {spp} / "
            f"{spp_nonmesh} inside [{REF_SAMPLE_BASE}, {REF_SAMPLE_BASE + REF_MAX_SPP}])"
        )

    # 4. statistics
    hdr = stats.mean()
    var_mean = stats.variance_of_mean()
    display_var_mean = stats.display_variance_of_mean()
    count = issued.reshape(height, width).to(torch.int32)
    _sync(device)
    seconds = time.perf_counter() - t0

    n_mesh = int(mesh_mask.sum())
    floor = float(var_mean[mesh_mask].double().mean()) if n_mesh else math.nan
    meta = {
        "key": key,
        "key_dict": _key_dict(fingerprint, camera_desc, params),
        "rays": int(issued.sum()),
        "seconds": seconds,
        "n_mesh_px": n_mesh,
        "n_nonfinite": stats.num_nonfinite,
        "ref_noise_floor_mesh": floor if math.isfinite(floor) else None,
    }
    arrays = {
        "hdr": hdr,
        "var_mean": var_mean,
        "display_var_mean": display_var_mean,
        "mesh_mask": mesh_mask,
        "hi_mask": hi_mask,
        "count": count,
    }
    # 5. arrays (tmp + replace), then meta.json last
    save_reference(cache_dir, {n: a.detach().cpu().numpy() for n, a in arrays.items()}, meta)
    return ReferenceBundle(**arrays, key=key, meta=meta, cache_dir=cache_dir)


def render_reference(renderer, camera, spp: int, out_dir, *, ckpt_path: str, **kw) -> Tensor:
    """The reference's ``[H, W, 3]`` float32 HDR mean (see :func:`reference_bundle`)."""
    return reference_bundle(renderer, camera, spp, out_dir, ckpt_path=ckpt_path, **kw).hdr
