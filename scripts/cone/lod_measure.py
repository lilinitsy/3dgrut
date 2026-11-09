# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measurements of the Gaussian LOD for cone tracing on the room roughness ladder (512 px, close camera).

Sections (``--sections``, any subset of ``abcde``; results merge into ``<out>/results.json``):

  a  thin threshold s_0: LOD-on cone frames at cap 0.005 for each s_0 against the LOD-off cone frame at cap 0.005
  b  picture match: 32x32 crops around sphere_2 / sphere_3, LOD on against per-Gaussian widening (LOD off), and each
     against the 16k spp reference
  c  cost: full-frame time, LOD on at caps 0.005..0.35 and LOD off at 0.005 / 0.05; plus the band-edge caps and the
     Gaussians per cut as a diagnostic of where the time goes
  d  camera-ray cost: cone + LOD on with every glossy sphere hidden, against MC 1 spp on the same scene
  e  seams: two frames with the camera moved 1 mm sideways, per-pixel difference on the spheres, LOD on and off

Cones apply to every PBR hit (max roughness 1), frames are unclamped (``configure``), timings are the median of 20
CUDA-synchronized frames after 3 warm-ups. Sections b-e use ``--thin-sigma`` as s_0.

Usage: python scripts/cone/lod_measure.py --sections abcde --out C:/Users/ville/3dgrut_runs/lod_measure
       (``--match-caps 0.05,0.1`` limits section b to those caps)
"""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from threedgrut_playground.adaptive.evaluation.cone_equal_time import (
    ConeSetting,
    _font,
    _interior,
    _pil,
    configure,
    load_ladder,
    luminance,
    object_boxes,
    object_roughness,
    render_frame,
    to_display,
)
from threedgrut_playground.adaptive.evaluation.ldr import ensure_outside_repo, save_png
from threedgrut_playground.adaptive.evaluation.metrics import flip_error_map
from threedgrut_playground.adaptive.evaluation.reflection_rate import LADDER_PRESET, object_id_map
from threedgrut_playground.adaptive.evaluation.sampling import trace_only_renderer
from threedgrut_playground.adaptive.evaluation.scenes import CameraPreset, glossy_layout, scene_frame
from threedgrut_playground.adaptive.evaluation.viewing import ppd_from_camera

RES = 512
WARMUPS = 3
FRAMES = 20
THIN_SIGMAS = (0.0, 0.0005, 0.001, 0.002, 0.005)
MATCH_CAPS = (0.02, 0.05, 0.1)
EXTRA_MATCH_CAPS = (0.2,)  # beyond the brief: per-Gaussian widening of a 32x32 crop at 0.2 still takes ~2 minutes
CROP = 32
SPHERES = ("sphere_2", "sphere_3")
COST_CAPS_LOD = (0.005, 0.05, 0.1, 0.2, 0.35)
COST_CAPS_OFF = (0.005, 0.05)  # LOD off above 0.05 takes minutes per frame
BAND_EDGE_CAPS = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32)  # diagnostic: each adds one more LOD slot
SEAM_STEP = 0.001  # world units, "1 mm"
SEAM_CAPS_LOD = (0.05, 0.2)
SEAM_CAPS_OFF = (0.05,)
ERROR_GAIN = 4.0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", type=Path, default=REPO_ROOT / "runs/room_3dgrt/room-0211_151617/ckpt_last.pt")
    parser.add_argument(
        "--ref",
        type=Path,
        default=Path("C:/Users/ville/3dgrut_runs/cone_equal_time/reference_unclamped_512px_16384spp.pt"),
    )
    parser.add_argument("--out", type=Path, default=Path("C:/Users/ville/3dgrut_runs/lod_measure"))
    parser.add_argument("--sections", default="abcde")
    parser.add_argument("--thin-sigma", type=float, default=None, help="s_0 for sections b-e (default: the engine's)")
    parser.add_argument(
        "--match-caps",
        type=lambda text: tuple(float(c) for c in text.split(",")),
        default=MATCH_CAPS + EXTRA_MATCH_CAPS,
        help="comma-separated caps for section b (default: 0.02,0.05,0.1,0.2; per-Gaussian at 0.2 takes minutes)",
    )
    return parser.parse_args(argv)


def log(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def median_frame_ms(step, warmups: int = WARMUPS, frames: int = FRAMES) -> dict:
    for _ in range(warmups):
        step()
    torch.cuda.synchronize()
    times = []
    for _ in range(frames):
        start = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    return {"median_ms": statistics.median(times), "min_ms": min(times), "max_ms": max(times)}


def window(box, crop, height, width):
    y0, y1, x0, x1 = box
    side = min(crop, y1 - y0)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    top, left = min(max(cy - side // 2, 0), height - side), min(max(cx - side // 2, 0), width - side)
    return top, top + side, left, left + side


def grid(rows, header, path, scale=4):
    """rows: [(row_label, [(caption, image[h,w,3]), ...])]; images upscaled ``scale``x nearest."""
    from PIL import Image, ImageDraw

    font, small = _font(16), _font(13)
    cell_h = max(img.shape[0] for _, cols in rows for _, img in cols) * scale
    cell_w = max(img.shape[1] for _, cols in rows for _, img in cols) * scale
    label_w, head_h, cap_h, pad = 120, 34, 40, 6
    ncol = max(len(cols) for _, cols in rows)
    canvas = Image.new("RGB", (label_w + ncol * (cell_w + pad), head_h + len(rows) * (cell_h + cap_h + pad)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 8), header, fill="black", font=font)
    for r, (label, cols) in enumerate(rows):
        y = head_h + r * (cell_h + cap_h + pad)
        draw.multiline_text((6, y + cell_h // 2 - 16), label, fill="black", font=font)
        for c, (caption, img) in enumerate(cols):
            x = label_w + c * (cell_w + pad)
            tile = _pil(img).resize((img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST)
            canvas.paste(tile, (x, y))
            draw.multiline_text((x + 2, y + cell_h + 2), caption, fill="black", font=small)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def gain(diff: torch.Tensor) -> torch.Tensor:
    return (diff.abs() * ERROR_GAIN).clamp(0.0, 1.0)


class Context:
    def __init__(self, args):
        from threedgrut_playground.engine import Engine3DGRUT

        self.out = ensure_outside_repo(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.engine = engine = Engine3DGRUT(
            gs_object=str(args.ckpt),
            mesh_assets_folder=str(REPO_ROOT / "threedgrut_playground" / "assets"),
            default_config="apps/colmap_3dgrt.yaml",
        )
        self.device = torch.device(engine.device)
        self.preset = load_ladder(engine)
        self.camera = self.preset.to_camera(RES, RES, device=self.device)
        rays = engine.raygen(self.camera, use_spp=False)
        self.rays_o, self.rays_d = rays.rays_ori.contiguous(), rays.rays_dir.contiguous()
        self.ppd = ppd_from_camera(self.camera)
        self.layout = glossy_layout(scene_frame(engine.scene_mog.positions.detach().cpu()), LADDER_PRESET)
        self.names = self.layout.object_names()
        self.object_ids = object_id_map(trace_only_renderer(engine), self.camera, self.layout, self.device)
        self.boxes = object_boxes(self.object_ids, self.names)
        self.roughness = object_roughness(self.names)
        cached = torch.load(args.ref, map_location=self.device)
        self.reference = to_display(engine, (cached["even"] + cached["odd"]) / 2.0)
        self.ref_even = to_display(engine, cached["even"])
        self.ref_odd = to_display(engine, cached["odd"])
        self.default_thin_sigma = engine.lod_thin_sigma
        self.thin_sigma = engine.lod_thin_sigma if args.thin_sigma is None else args.thin_sigma
        self.match_caps = tuple(args.match_caps)
        self.concerns = []

    def set_mode(self, cap, lod, thin_sigma=None, max_roughness=1.0):
        """Cone at ``cap`` (``None``: MC) with LOD on/off; frames unclamped, frame_id 0, full-frame FOV."""
        e = self.engine
        configure(e, None if cap is None else ConeSetting(cap, max_roughness))
        e.use_gaussian_lod = bool(lod)
        e.lod_thin_sigma = self.thin_sigma if thin_sigma is None else thin_sigma
        e.camera_fov = self.preset.fov_deg
        e.frame_id = 0

    def full(self, rays_o=None, rays_d=None):
        self.engine.camera_fov = self.preset.fov_deg
        self.engine.frame_id = 0
        img = to_display(
            self.engine,
            render_frame(
                self.engine, self.rays_o if rays_o is None else rays_o, self.rays_d if rays_d is None else rays_d
            ),
        )
        self.check_finite(img)
        return img

    def time_full(self):
        def step():
            self.engine.frame_id = 0
            render_frame(self.engine, self.rays_o, self.rays_d)

        self.engine.camera_fov = self.preset.fov_deg
        return median_frame_ms(step)

    def check_finite(self, img, label=""):
        if not bool(torch.isfinite(img).all()):
            msg = f"non-finite pixels {label}: {int((~torch.isfinite(img)).sum())}"
            log("CONCERN " + msg)
            self.concerns.append(msg)

    def mask(self, name, ids=None):
        ids = self.object_ids if ids is None else ids
        return ids == self.names.index(name)

    def per_object(self, fn):
        return {n: fn(self.mask(n)) for n in self.names if bool(self.mask(n).any())}


def lum_ratio(img, ref, mask):
    return (luminance(img)[mask].mean() / luminance(ref)[mask].mean().clamp_min(1e-6)).item()


# ---------------------------------------------------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------------------------------------------------


def section_a(ctx: Context) -> dict:
    """s_0 sweep at cap 0.005 against LOD off at 0.005."""
    ctx.set_mode(0.005, lod=False)
    off_time = ctx.time_full()
    off = ctx.full()
    save_png(off, ctx.out / "a_thin" / "lod_off_cap0.005.png")
    log(f"a: LOD off cap 0.005: {off_time['median_ms']:.2f} ms")
    res = {"lod_off_cap0.005": off_time, "thin_sigmas": {}}
    for s0 in THIN_SIGMAS:
        ctx.set_mode(0.005, lod=True, thin_sigma=s0)
        t = ctx.time_full()
        img = ctx.full()
        flip = flip_error_map(img, off, ctx.ppd)
        diff = (img - off).abs()
        entry = dict(
            t,
            mean_abs_diff=diff.mean().item(),
            max_abs_diff=diff.max().item(),
            frac_pixels_diff_gt_1_255=(diff.max(dim=-1).values > 1.0 / 255.0).float().mean().item(),
            flip_mean=flip.mean().item(),
            flip_p99=torch.quantile(flip.flatten(), 0.99).item(),
            flip_per_object=ctx.per_object(lambda m: flip[m].mean().item()),
            flip_background=flip[ctx.object_ids < 0].mean().item(),
        )
        res["thin_sigmas"][f"{s0:g}"] = entry
        save_png(img, ctx.out / "a_thin" / f"lod_s0_{s0:g}.png")
        save_png(gain(img - off), ctx.out / "a_thin" / f"lod_s0_{s0:g}_diff_x{ERROR_GAIN:g}.png")
        log(
            f"a: s_0 {s0:g}: {t['median_ms']:.2f} ms, mean|d| {entry['mean_abs_diff']:.2e}, max|d| "
            f"{entry['max_abs_diff']:.3f}, FLIP {entry['flip_mean']:.4f} (p99 {entry['flip_p99']:.4f})"
        )
    return res


def section_b(ctx: Context) -> dict:
    """Crops: LOD against per-Gaussian widening at the same cap, and each against the reference."""
    e = ctx.engine
    wins = {s: window(ctx.boxes[s], CROP, RES, RES) for s in SPHERES}

    def crop(t, s):
        y0, y1, x0, x1 = wins[s]
        return t[y0:y1, x0:x1]

    def render_crop(s):
        side = wins[s][1] - wins[s][0]
        e.camera_fov = ctx.preset.fov_deg * side / RES  # the engine takes the pixel cone from fov / height
        e.frame_id = 0
        img = to_display(
            e, render_frame(e, crop(ctx.rays_o[0], s)[None].contiguous(), crop(ctx.rays_d[0], s)[None].contiguous())
        )
        ctx.check_finite(img, f"crop {s}")
        return img

    def timed_crop(s):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        img = render_crop(s)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        if ms < 5000.0:  # warm re-render when cheap; slow per-Gaussian caps are timed on their only render
            t0 = time.perf_counter()
            img = render_crop(s)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000.0
        return img, ms

    res = {"crop": CROP, "windows": {s: list(w) for s, w in wins.items()}, "spheres": {}}
    images = {s: [] for s in SPHERES}
    for s in SPHERES:
        m = crop(ctx.mask(s), s)
        ref = crop(ctx.reference, s)
        noise = flip_error_map(crop(ctx.ref_even, s), crop(ctx.ref_odd, s), ctx.ppd)[m].mean().item()
        res["spheres"][s] = {"roughness": ctx.roughness[s], "reference_half_vs_half_flip": noise, "caps": {}}
        images[s].append((f"reference 16k spp\nhalf/half FLIP {noise:.3f}", ref))
    for cap in ctx.match_caps:
        rendered = {}
        for lod in (False, True):
            ctx.set_mode(cap, lod=lod)
            e.camera_fov = ctx.preset.fov_deg / RES
            render_frame(e, ctx.rays_o[:, :1, :1].contiguous(), ctx.rays_d[:, :1, :1].contiguous())  # (re)build BVHs
            torch.cuda.synchronize()
            for s in SPHERES:
                rendered[(lod, s)] = timed_crop(s)
        for s in SPHERES:
            m = crop(ctx.mask(s), s)
            ref = crop(ctx.reference, s)
            (pg, pg_ms), (lo, lo_ms) = rendered[(False, s)], rendered[(True, s)]
            flip_lod_pg = flip_error_map(lo, pg, ctx.ppd)
            entry = {
                "extra": cap in EXTRA_MATCH_CAPS,
                "flip_lod_vs_per_gaussian": flip_lod_pg[m].mean().item(),
                "flip_lod_vs_per_gaussian_interior": flip_lod_pg[_interior(m)].mean().item(),
                "lum_ratio_lod_over_per_gaussian": lum_ratio(lo, pg, m),
                "flip_per_gaussian_vs_ref": flip_error_map(pg, ref, ctx.ppd)[m].mean().item(),
                "flip_lod_vs_ref": flip_error_map(lo, ref, ctx.ppd)[m].mean().item(),
                "lum_ratio_per_gaussian_over_ref": lum_ratio(pg, ref, m),
                "lum_ratio_lod_over_ref": lum_ratio(lo, ref, m),
                "per_gaussian_ms": pg_ms,
                "lod_ms": lo_ms,
            }
            res["spheres"][s]["caps"][f"{cap:g}"] = entry
            save_png(pg, ctx.out / "b_match" / f"{s}_cap{cap:g}_per_gaussian.png")
            save_png(lo, ctx.out / "b_match" / f"{s}_cap{cap:g}_lod.png")
            images[s].append(
                (f"per-Gaussian {cap:g}\n{pg_ms / 1000:.1f} s  FLIP/ref {entry['flip_per_gaussian_vs_ref']:.3f}", pg)
            )
            images[s].append((f"LOD {cap:g}\n{lo_ms:.0f} ms  FLIP/ref {entry['flip_lod_vs_ref']:.3f}", lo))
            caption = (
                f"|LOD-perG| x{ERROR_GAIN:g}\nFLIP {entry['flip_lod_vs_per_gaussian']:.3f} "
                f"lum {entry['lum_ratio_lod_over_per_gaussian']:.3f}"
            )
            images[s].append((caption, gain(lo - pg)))
            log(
                f"b: cap {cap:g} {s}: FLIP LOD/perG {entry['flip_lod_vs_per_gaussian']:.4f}, lum ratio "
                f"{entry['lum_ratio_lod_over_per_gaussian']:.4f}, "
                f"FLIP/ref perG {entry['flip_per_gaussian_vs_ref']:.4f} "
                f"LOD {entry['flip_lod_vs_ref']:.4f}; perG {pg_ms:.0f} ms, LOD {lo_ms:.1f} ms"
            )
    rows = [(f"{s}\nroughness\n{ctx.roughness[s]:g}", images[s]) for s in SPHERES]
    grid(
        rows,
        f"32x32 crops: LOD against per-Gaussian widening (LOD off), s_0 {ctx.thin_sigma:g}",
        ctx.out / "b_picture_match.png",
    )
    return res


def lod_cut_stats(ctx: Context) -> dict:
    """Gaussians per slot and two proxy-overlap figures, relative to slot 0, for each slot inflated by its band's upper
    edge (its inflation at a band-edge cap; the top slot of any other cap is inflated by the cap). The sum of squared
    largest inflated scales (a cone hits about as many proxies as their summed cross-section) cannot see inflation of
    the small axes of thin or flat Gaussians; the summed area proxy (ab + ac + bc of the three inflated scales) can."""
    e = ctx.engine
    lod = e.ensure_gaussian_lod()
    s = lod.level_sigmas
    n = lod.num_original
    slot0 = lod.scales[:n].max(dim=-1).values.square().sum().item()

    def area(sc):
        return (sc[:, 0] * sc[:, 1] + sc[:, 0] * sc[:, 2] + sc[:, 1] * sc[:, 2]).sum().item()

    area0 = area(lod.scales[:n])
    out = {"num_original": n, "level_sigmas": list(map(float, s)), "slots": []}

    def add(slot, band, idx, inflate):
        scales = lod.scales[idx]
        inflated_scales = torch.sqrt(scales * scales + inflate * inflate)
        sc = scales.max(dim=-1).values
        inflated = inflated_scales.max(dim=-1).values
        out["slots"].append(
            {
                "slot": slot,
                "band": band,
                "gaussians": int(idx.numel()),
                "fraction_of_n": idx.numel() / n,
                "inflate": inflate,
                "median_max_scale": sc.median().item(),
                "median_inflated_max_scale": inflated.median().item(),
                "cross_section_vs_slot0": inflated.square().sum().item() / slot0,
                "area_proxy_vs_slot0": area(inflated_scales) / area0,
                "median_min_scale": scales.min(dim=-1).values.median().item(),
                "median_min_scale_growth": (inflated_scales.min(dim=-1).values / scales.min(dim=-1).values)
                .median()
                .item(),
                "min_density": lod.densities[idx].min().item(),
            }
        )

    all_orig = torch.arange(n, device=lod.scales.device)
    add(0, [0.0, ctx.thin_sigma], all_orig, 0.0)
    add(1, [ctx.thin_sigma, float(s[0])], all_orig, float(s[0]))
    for k, idx in enumerate(lod.slot_index, start=1):
        add(1 + k, [float(s[k - 1]), float(s[k])], idx.to(lod.scales.device).long(), float(s[k]))
    return out


def section_c(ctx: Context) -> dict:
    res = {"lod_on": {}, "lod_off": {}, "band_edge_caps_lod_on": {}}
    ctx.set_mode(None, lod=False)
    res["mc_1spp"] = ctx.time_full()
    mc = ctx.full()
    save_png(mc, ctx.out / "c_cost" / "mc_1spp.png")
    log(f"c: MC 1 spp {res['mc_1spp']['median_ms']:.2f} ms")
    images = {}
    for cap in COST_CAPS_OFF:
        ctx.set_mode(cap, lod=False)
        t0 = time.perf_counter()
        res["lod_off"][f"{cap:g}"] = ctx.time_full()
        images[("off", cap)] = ctx.full()
        save_png(images[("off", cap)], ctx.out / "c_cost" / f"lod_off_cap{cap:g}.png")
        ms = res["lod_off"][f"{cap:g}"]["median_ms"]
        log(f"c: LOD off cap {cap:g}: {ms:.2f} ms ({time.perf_counter() - t0:.0f} s)")
    for cap in COST_CAPS_LOD:
        ctx.set_mode(cap, lod=True)
        res["lod_on"][f"{cap:g}"] = ctx.time_full()
        images[("on", cap)] = ctx.full()
        _, num_slots, inflations = ctx.engine._lod_slots
        res["lod_on"][f"{cap:g}"].update(num_slots=num_slots, slot_inflations=list(inflations))
        save_png(images[("on", cap)], ctx.out / "c_cost" / f"lod_on_cap{cap:g}.png")
        log(
            f"c: LOD on cap {cap:g}: {res['lod_on'][f'{cap:g}']['median_ms']:.2f} ms, {num_slots} slots, "
            f"top slot inflated by {inflations[-1] if inflations else 0.0:g}"
        )
    for cap in BAND_EDGE_CAPS:
        ctx.set_mode(cap, lod=True)
        res["band_edge_caps_lod_on"][f"{cap:g}"] = dict(ctx.time_full(), num_slots=ctx.engine._lod_slots[1])
        log(
            f"c: LOD on band-edge cap {cap:g}: {res['band_edge_caps_lod_on'][f'{cap:g}']['median_ms']:.2f} ms, "
            f"{ctx.engine._lod_slots[1]} slots"
        )
    res["cuts"] = lod_cut_stats(ctx)
    for sl in res["cuts"]["slots"]:
        log(
            f"c: slot {sl['slot']} band {sl['band']}: {sl['gaussians']} Gaussians ({sl['fraction_of_n']:.3f} N), "
            f"cross-section {sl['cross_section_vs_slot0']:.2f}x slot 0, area proxy {sl['area_proxy_vs_slot0']:.2f}x, "
            f"median smallest-axis growth {sl['median_min_scale_growth']:.1f}x, min density {sl['min_density']:.2e}"
        )
    # Full-frame image comparisons that come for free: LOD against LOD off at 0.005 and 0.05, and every frame against
    # the reference per object (luminance ratio: darkening at coarse levels)
    res["full_frame_lod_vs_off"] = {}
    for cap in COST_CAPS_OFF:
        flip = flip_error_map(images[("on", cap)], images[("off", cap)], ctx.ppd)
        res["full_frame_lod_vs_off"][f"{cap:g}"] = {
            "flip_per_object": ctx.per_object(lambda m: flip[m].mean().item()),
            "lum_ratio_per_object": ctx.per_object(lambda m: lum_ratio(images[("on", cap)], images[("off", cap)], m)),
            "flip_background": flip[ctx.object_ids < 0].mean().item(),
        }
    res["lum_ratio_vs_reference"] = {
        f"{kind}_{cap:g}": ctx.per_object(lambda m: lum_ratio(img, ctx.reference, m))
        for (kind, cap), img in images.items()
    }
    res["lum_ratio_vs_reference"]["mc_1spp"] = ctx.per_object(lambda m: lum_ratio(mc, ctx.reference, m))
    on = res["lod_on"]
    res["checks"] = {
        "lod_0.35_over_lod_0.005": on["0.35"]["median_ms"] / on["0.005"]["median_ms"],
        "lod_0.005_over_off_0.005": on["0.005"]["median_ms"] / res["lod_off"]["0.005"]["median_ms"],
        "lod_0.05_over_off_0.05": on["0.05"]["median_ms"] / res["lod_off"]["0.05"]["median_ms"],
        "lod_0.005_over_mc": on["0.005"]["median_ms"] / res["mc_1spp"]["median_ms"],
    }
    cols = [("MC 1 spp", mc)] + [
        (f"LOD {'on' if k == 'on' else 'off'} cap {c:g}", img)
        for (k, c), img in sorted(images.items(), key=lambda x: (x[0][1], x[0][0]))
    ]
    grid(
        [("full\nframes", [(f"{c}", img[::2, ::2]) for c, img in cols])],
        "Full frames (half size)",
        ctx.out / "c_full_frames.png",
        scale=1,
    )
    return res


def section_d(ctx: Context) -> dict:
    """Spheres hidden: the frame's cones come only from the floor (rmax 1) or from nothing (rmax 0, floor rougher)."""
    from threedgrut_playground.engine import OptixPrimitiveTypes

    prims = ctx.engine.primitives
    saved = {name: obj.primitive_type for name, obj in prims.objects.items()}
    try:
        for name, obj in prims.objects.items():
            if name.startswith("Sphere"):
                obj.primitive_type = OptixPrimitiveTypes.NONE
        prims.rebuild_bvh_if_needed(force=True, rebuild=True)
        res = {}
        ctx.set_mode(None, lod=False)
        res["mc_1spp"] = ctx.time_full()
        save_png(ctx.full(), ctx.out / "d_camera" / "mc_1spp_no_spheres.png")
        for lod in (True, False):
            for rmax, label in ((0.0, "no_cones"), (1.0, "floor_cones")):
                ctx.set_mode(0.005, lod=lod, max_roughness=rmax)
                key = f"cone0.005_lod_{'on' if lod else 'off'}_{label}"
                res[key] = ctx.time_full()
                save_png(ctx.full(), ctx.out / "d_camera" / f"{key}.png")
                res[key]["over_mc"] = res[key]["median_ms"] / res["mc_1spp"]["median_ms"]
                log(
                    f"d: {key}: {res[key]['median_ms']:.2f} ms ({res[key]['over_mc']:.2f}x MC "
                    f"{res['mc_1spp']['median_ms']:.2f} ms)"
                )
        return res
    finally:
        for name, obj in prims.objects.items():
            obj.primitive_type = saved[name]
        prims.rebuild_bvh_if_needed(force=True, rebuild=True)
        ctx.engine.invalidate_materials_on_gpu()


def section_e(ctx: Context) -> dict:
    e = ctx.engine
    p = ctx.preset
    fwd = p.forward()
    up = torch.tensor(p.up, dtype=torch.float32)
    right = torch.linalg.cross(fwd, up)
    right = right / right.norm()
    shift = (right * SEAM_STEP).tolist()
    moved = CameraPreset(
        p.name + "_moved",
        tuple(a + b for a, b in zip(p.eye, shift)),
        tuple(a + b for a, b in zip(p.at, shift)),
        p.up,
        p.fov_deg,
    )
    cam_b = moved.to_camera(RES, RES, device=ctx.device)
    rays_b = e.raygen(cam_b, use_spp=False)
    rb_o, rb_d = rays_b.rays_ori.contiguous(), rays_b.rays_dir.contiguous()
    ids_b = object_id_map(trace_only_renderer(e), cam_b, ctx.layout, ctx.device)
    radii = {f"sphere_{i + 1}": float(r) for i, (_, _, r) in enumerate(ctx.layout.spheres)}
    res = {"step_world_units": SEAM_STEP, "sphere_radii": radii, "runs": {}}
    frames = {}
    for lod, caps in ((True, SEAM_CAPS_LOD), (False, SEAM_CAPS_OFF)):
        for cap in caps:
            ctx.set_mode(cap, lod=lod)
            a = ctx.full()
            b = ctx.full(rb_o, rb_d)
            a2 = ctx.full()
            key = f"lod_{'on' if lod else 'off'}_cap{cap:g}"
            frames[key] = (a, b)
            diff = (a - b).abs().max(dim=-1).values
            stats = {"same_camera_rerender_max_diff": (a - a2).abs().max().item(), "objects": {}}
            for n in ctx.names:
                m = _interior(ctx.mask(n) & (ids_b == ctx.names.index(n)))
                if not bool(m.any()):
                    continue
                d = diff[m]
                stats["objects"][n] = {
                    "mean_abs_diff": d.mean().item(),
                    "p99_abs_diff": torch.quantile(d, 0.99).item(),
                    "max_abs_diff": d.max().item(),
                }
            res["runs"][key] = stats
            log(
                f"e: {key}: "
                + ", ".join(
                    f"{n} {v['mean_abs_diff']:.4f}/{v['max_abs_diff']:.3f}" for n, v in stats["objects"].items()
                )
                + f"; rerender max {stats['same_camera_rerender_max_diff']:.2e}"
            )
            save_png(gain(a - b), ctx.out / "e_seams" / f"{key}_diff_x{ERROR_GAIN:g}.png")
            save_png(a, ctx.out / "e_seams" / f"{key}_a.png")
    s = "sphere_3"
    y0, y1, x0, x1 = ctx.boxes[s]
    c = lambda t: t[y0:y1, x0:x1]
    a, b = frames["lod_on_cap0.2"]
    a5, b5 = frames["lod_on_cap0.05"]
    o5, _ = frames["lod_off_cap0.05"]
    cols = [
        ("reference 16k spp", c(ctx.reference)),
        ("LOD cap 0.2", c(a)),
        ("LOD cap 0.2, moved 1 mm", c(b)),
        (f"|diff| x{ERROR_GAIN:g}", c(gain(a - b))),
        ("LOD cap 0.05", c(a5)),
        (f"|diff| 0.05 x{ERROR_GAIN:g}", c(gain(a5 - b5))),
        ("LOD off cap 0.05", c(o5)),
    ]
    grid(
        [(f"{s}\nroughness\n{ctx.roughness[s]:g}", cols)],
        "Seams: sphere_3, 4x nearest crops",
        ctx.out / "e_seams_sphere3_cap0.2.png",
    )
    return res


def main(argv=None) -> int:
    args = parse_args(argv)
    ctx = Context(args)
    path = ctx.out / "results.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    results["setup"] = {
        "ckpt": str(args.ckpt),
        "resolution": RES,
        "warmups": WARMUPS,
        "frames": FRAMES,
        "camera": {"eye": ctx.preset.eye, "at": ctx.preset.at, "fov_deg": ctx.preset.fov_deg},
        "lod_base_sigma": ctx.engine.lod_base_sigma,
        "lod_num_levels": ctx.engine.lod_num_levels,
        "engine_default_thin_sigma": ctx.default_thin_sigma,
        "cone_roughness_scale": ctx.engine.cone_roughness_scale,
        "cone_max_roughness": 1.0,
        "gpu": torch.cuda.get_device_name(),
    }
    sections = {"a": section_a, "b": section_b, "c": section_c, "d": section_d, "e": section_e}
    for key in args.sections:
        start = time.perf_counter()
        log(f"section {key} (s_0 {ctx.thin_sigma:g})")
        try:
            out = sections[key](ctx)
        except Exception as exc:  # record and go on: a crash is a finding, not something to fix here
            import traceback

            traceback.print_exc()
            out = {"error": repr(exc)}
            ctx.concerns.append(f"section {key}: {exc!r}")
        out["thin_sigma_used"] = ctx.thin_sigma if key != "a" else None
        out["seconds"] = time.perf_counter() - start
        results[key] = out
        results["concerns"] = ctx.concerns
        path.write_text(json.dumps(results, indent=2, default=str))
        log(f"section {key} done in {out['seconds']:.0f} s")
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
