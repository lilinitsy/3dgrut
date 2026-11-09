# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Equal-time comparison of cone-traced glossy reflections against Monte Carlo (MC) on the roughness ladder.

Scene: the four metal spheres of the reflection-rate test (roughness 0.05, 0.2, 0.4, 0.7) on a brushed copper floor,
seen from the ``close`` camera, with the procedural sky. Every image is ONE displayed frame from the viewer's render
path (``Engine3DGRUT._render_playground_hybrid``) with pixel-centre rays: no accumulation, no anti-aliasing jitter, and
no denoiser unless the column says so. Each column answers one question:

- ``Reference``: MC averaged over ``--ref-spp`` frames, plain BVH. What a converged frame looks like.
- ``MC 1 spp``: what the viewer shows on the first frame after the camera moves.
- ``MC equal time``: as many MC frames averaged as fit in the time of one cone frame (``N = round(cone ms / MC ms)``).
- ``MC eq. time + denoiser``: the same, minus the denoiser's own time, then the OptiX denoiser.
- ``Cone``: one cone-traced frame.

Each cone setting gets its own equal-time MC, because a wider cap inflates the BVH and costs more per frame. Error
maps show ``|image - reference|`` (luminance, x4). Per object it reports FLIP, the mean CVVDP heatmap (0.1 = one JOD
lost locally) and a detail ratio: the RMS Laplacian of luminance relative to the reference inside the object (1 = as
much fine detail as the reference; above 1 = noise for MC, a too-sharp reflection for the cone; below 1 = blurred).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .ldr import ensure_outside_repo, save_png
from .metrics import cvvdp_available, cvvdp_jod, flip_error_map
from .reflection_rate import (
    FLOOR_ROUGHNESS,
    LADDER_PRESET,
    ROUGHNESS_LADDER,
    cvvdp_heatmap,
    object_id_map,
    procedural_sky,
    register_ladder_materials,
)
from .sampling import trace_only_renderer
from .scenes import CameraPreset, glossy_layout, scene_frame, setup_glossy_scene
from .viewing import ppd_from_camera

Tensor = torch.Tensor

REPO_ROOT = Path(__file__).resolve().parents[3]
REF_FRAME_ID = 1 << 24  # reference frames use other random numbers than the test images
WARMUPS = 3
TIMED_FRAMES = 15
SLOW_MS = 500.0
SLOW_FRAMES = 3
# Engine settings ``run`` changes and puts back (the scene and environment it replaces are the caller's to restore)
RESTORED_SETTINGS = (
    "use_cone_tracing",
    "cone_max_sigma",
    "cone_max_roughness",
    "cone_split",
    "clamp_frame_output",
    "frame_id",
    "camera_fov",
)
ERROR_GAIN = 4.0
CROP = 176
NOISY = 1.25  # detail ratio above this: noise (MC) or too sharp (cone)
BLURRED = 0.8  # detail ratio below this: blurred


@dataclass(frozen=True)
class ConeSetting:
    """A cone configuration: the cap on the cone std (world units) and the roughest surface that still uses cones."""

    max_sigma: float
    max_roughness: float

    @property
    def name(self) -> str:
        return f"cone_sigma{self.max_sigma:g}_rmax{self.max_roughness:g}"


def parse_cone_setting(text: str) -> ConeSetting:
    """``"0.005:0.2"`` -> ``ConeSetting(0.005, 0.2)``."""
    sigma, roughness = text.split(":")
    return ConeSetting(float(sigma), float(roughness))


def equal_time_spp(cone_ms: float, mc_ms: float, overhead_ms: float = 0.0) -> int:
    """MC frames that fit in one cone frame after ``overhead_ms``: ``round((cone - overhead) / mc)``, at least 1."""
    if mc_ms <= 0.0:
        raise ValueError("mc_ms must be positive")
    return max(1, int(math.floor((cone_ms - overhead_ms) / mc_ms + 0.5)))


def load_ladder(engine) -> CameraPreset:
    """Replace the engine's objects with the roughness ladder and the procedural sky; return the ``close`` camera.

    Also sets ``engine.camera_fov`` to the camera's FOV, which the camera rays' cone width is computed from.
    """
    register_ladder_materials(engine)
    engine.environment.envmap = procedural_sky(torch.device(engine.device))
    engine.environment.current_name = "procedural_sky"
    preset = setup_glossy_scene(engine, LADDER_PRESET)["close"]
    engine.camera_fov = preset.fov_deg
    engine.invalidate_materials_on_gpu()
    return preset


def configure(engine, cone: Optional[ConeSetting]) -> None:
    """Cone tracing off (``None``) or on with ``cone`` over the whole frame; frames are not clamped (see ``run``)."""
    engine.clamp_frame_output = False
    engine.use_cone_tracing = cone is not None
    engine.cone_split = 0.0
    if cone is not None:
        engine.cone_max_sigma = cone.max_sigma
        engine.cone_max_roughness = cone.max_roughness


def render_frame(engine, rays_o: Tensor, rays_d: Tensor) -> Tensor:
    """One frame ``[H, W, 3]``, tonemapped and before the display gamma (unclamped after ``configure``)."""
    out = engine._render_playground_hybrid(rays_o.clone(), rays_d.clone())
    return engine.environment.tonemap(out["pred_features"][0].clone())


def mean_frames(engine, rays_o: Tensor, rays_d: Tensor, count: int, first_frame_id: int) -> Tuple[Tensor, Tensor]:
    """Means of the even and of the odd frames among ``count`` frames starting at ``first_frame_id``."""
    engine.frame_id = first_frame_id
    sums = [None, None]
    for index in range(count):
        frame = render_frame(engine, rays_o, rays_d)
        slot = index % 2
        sums[slot] = frame if sums[slot] is None else sums[slot] + frame
    if sums[1] is None:
        return sums[0], sums[0]
    return sums[0] / ((count + 1) // 2), sums[1] / (count // 2)


def average(engine, rays_o: Tensor, rays_d: Tensor, count: int, first_frame_id: int) -> Tensor:
    """Mean of ``count`` frames starting at ``first_frame_id``."""
    even, odd = mean_frames(engine, rays_o, rays_d, count, first_frame_id)
    n_even, n_odd = (count + 1) // 2, count // 2
    return even if n_odd == 0 else (even * n_even + odd * n_odd) / count


def to_display(engine, linear: Tensor) -> Tensor:
    """Accumulation space to the displayed image, as ``Engine3DGRUT.render_pass`` does (gamma, clamp)."""
    return torch.pow(linear.clamp(0.0, 1.0), 1.0 / engine.gamma_correction)


def denoise(engine, display: Tensor) -> Tensor:
    """The viewer's OptiX denoiser on a displayed image ``[H, W, 3]``."""
    return engine.tracer.denoise(display[None].contiguous())[0].clamp(0.0, 1.0)


def median_ms(step: Callable[[], object], warmups: int = WARMUPS, frames: int = TIMED_FRAMES) -> float:
    """Median wall time of ``step`` in ms, CUDA synchronized, after ``warmups`` untimed calls.

    A step slower than ``SLOW_MS`` (the wide cone caps take seconds) is timed on ``SLOW_FRAMES`` frames after one warm-up.
    """
    step()  # first warm-up; also rebuilds the BVH after a cone setting change
    torch.cuda.synchronize()
    start = time.perf_counter()
    step()
    torch.cuda.synchronize()
    if (time.perf_counter() - start) * 1000.0 > SLOW_MS:
        frames = SLOW_FRAMES
    else:
        for _ in range(warmups - 2):
            step()
    times = []
    for _ in range(frames):
        torch.cuda.synchronize()
        start = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times)


def luminance(rgb: Tensor) -> Tensor:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _interior(mask: Tensor, margin: int = 2) -> Tensor:
    """``mask`` eroded by ``margin`` pixels, so silhouette edges do not count as detail."""
    eroded = -torch.nn.functional.max_pool2d(-mask.float()[None, None], 2 * margin + 1, 1, margin)[0, 0]
    return eroded > 0.5


def _laplacian(gray: Tensor) -> Tensor:
    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=gray.device)
    return torch.nn.functional.conv2d(gray[None, None], kernel[None, None], padding=1)[0, 0]


def detail_ratio(image: Tensor, reference: Tensor, mask: Tensor, reference_noise: Optional[Tensor] = None) -> float:
    """RMS Laplacian of luminance of ``image`` over that of ``reference``, inside ``mask`` minus a 2 px rim.

    ``reference_noise`` is an image with the reference's own noise and no signal (half the difference of two independent
    half references); its Laplacian energy is subtracted from the reference's, so leftover reference noise does not
    count as detail the others lack.
    """
    inner = _interior(mask)
    if not bool(inner.any()):
        return float("nan")
    test = _laplacian(luminance(image))[inner].square().mean()
    ref = _laplacian(luminance(reference))[inner].square().mean()
    if reference_noise is not None:
        ref = ref - _laplacian(luminance(reference_noise))[inner].square().mean()
    return float(torch.sqrt(test / ref.clamp_min(1e-12)))


def verdict(method: str, ratio: float) -> str:
    """Plain-words reading of a detail ratio."""
    if ratio != ratio:
        return "-"
    if ratio > NOISY:
        return "too sharp" if method == "cone" else "noisy"
    if ratio < BLURRED:
        return "blurred"
    return "detail ok"


def object_boxes(object_ids: Tensor, names: Sequence[str], pad: int = 4) -> Dict[str, Tuple[int, int, int, int]]:
    """Square ``(y0, y1, x0, x1)`` box around every object but the floor that is on screen."""
    height, width = object_ids.shape
    boxes = {}
    for index, name in enumerate(names):
        if name == "floor":
            continue
        ys, xs = torch.nonzero(object_ids == index, as_tuple=True)
        if ys.numel() == 0:
            continue
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        side = min(max(y1 - y0, x1 - x0) + 2 * pad, height, width)
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        top = min(max(cy - side // 2, 0), height - side)
        left = min(max(cx - side // 2, 0), width - side)
        boxes[name] = (top, top + side, left, left + side)
    return boxes


def object_roughness(names: Sequence[str]) -> Dict[str, float]:
    """Roughness of each ladder object: the floor, then ``sphere_k`` with the k-th ladder roughness."""
    return {n: FLOOR_ROUGHNESS if n == "floor" else ROUGHNESS_LADDER[int(n.split("_")[-1]) - 1] for n in names}


def error_image(image: Tensor, reference: Tensor) -> Tensor:
    """``|lum(image) - lum(reference)| x ERROR_GAIN`` on a black-red-yellow-white ramp ``[H, W, 3]``."""
    e = (luminance(image) - luminance(reference)).abs() * ERROR_GAIN
    return torch.stack([(3.0 * e).clamp(0, 1), (3.0 * e - 1.0).clamp(0, 1), (3.0 * e - 2.0).clamp(0, 1)], dim=-1)


# ---------------------------------------------------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------------------------------------------------


def _pil(image: Tensor):
    from PIL import Image

    arr = (image.detach().float().clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def _font(size: int):
    from PIL import ImageFont

    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def crops_figure(
    columns: Sequence[Tuple[str, Tensor, bool]],
    boxes: Dict[str, Tuple[int, int, int, int]],
    captions: Dict[Tuple[str, str], str],
    row_labels: Dict[str, str],
    title: str,
    path: Path,
) -> None:
    """One row per box, one column per ``(label, image, is_error_map)``; ``captions[(label, object)]`` under a cell."""
    from PIL import Image, ImageDraw

    label_w, header_h, title_h, caption_h, gap = 120, 34, 30, 34, 6
    cell_w, cell_h = CROP + gap, CROP + caption_h + gap
    sheet = Image.new("RGB", (label_w + cell_w * len(columns), title_h + header_h + cell_h * len(boxes)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    font, small = _font(14), _font(12)
    draw.text((8, 6), title, fill=(255, 255, 255), font=_font(16))
    for col, (label, _, is_error) in enumerate(columns):
        x = label_w + col * cell_w
        draw.multiline_text((x, title_h + 1), label, fill=(255, 170, 120) if is_error else (230, 230, 230), font=small)
    for row, (sphere, (y0, y1, x0, x1)) in enumerate(boxes.items()):
        y = title_h + header_h + row * cell_h
        draw.multiline_text((8, y + CROP // 2 - 16), row_labels[sphere], fill=(230, 230, 230), font=font)
        for col, (label, image, _) in enumerate(columns):
            x = label_w + col * cell_w
            sheet.paste(_pil(image[y0:y1, x0:x1]).resize((CROP, CROP), Image.NEAREST), (x, y))
            caption = captions.get((label, sphere))
            if caption:
                draw.multiline_text((x, y + CROP + 2), caption, fill=(200, 200, 200), font=small, spacing=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def full_figure(columns: Sequence[Tuple[str, Tensor]], path: Path) -> None:
    """Full frames side by side with a label above each."""
    from PIL import Image, ImageDraw

    height, width = int(columns[0][1].shape[0]), int(columns[0][1].shape[1])
    header, gap = 26, 6
    sheet = Image.new("RGB", ((width + gap) * len(columns), height + header), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for col, (label, image) in enumerate(columns):
        x = col * (width + gap)
        draw.text((x + 4, 4), label, fill=(230, 230, 230), font=_font(15))
        sheet.paste(_pil(image), (x, header))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


# ---------------------------------------------------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="Gaussian checkpoint")
    parser.add_argument("--out-dir", required=True, help="output folder, outside the repository")
    parser.add_argument("--resolution", type=int, default=512, help="square image size")
    parser.add_argument("--ref-spp", type=int, default=16384, help="MC frames averaged for the reference")
    parser.add_argument(
        "--cones",
        default="0.005:0.2,0.05:1.0",
        help="cone settings max_sigma:max_roughness, comma separated (default: the viewer's, and a wide one)",
    )
    return parser.parse_args(argv)


def _load_or_render_reference(engine, rays_o, rays_d, spp: int, path: Path) -> Tuple[Tensor, Tensor]:
    if path.exists():
        cached = torch.load(path, map_location=rays_o.device)
        return cached["even"], cached["odd"]
    configure(engine, None)
    start = time.perf_counter()
    even, odd = mean_frames(engine, rays_o, rays_d, spp, REF_FRAME_ID)
    print(f"reference: {spp} MC frames in {time.perf_counter() - start:.0f} s", flush=True)
    torch.save({"even": even, "odd": odd}, path)
    return even, odd


def run(args: argparse.Namespace, engine=None) -> Dict[str, object]:
    """Render, score and draw every column; returns the summary also written to ``summary.json``.

    Frames are averaged unclamped and clamped once for display. The viewer clamps every frame before accumulating,
    which darkens Monte Carlo (a sample of 5 counts as 1) but not the noise-free cone; that would make the comparison
    about the clamp, not the estimators.
    """
    if engine is None:
        from threedgrut_playground.engine import Engine3DGRUT

        engine = Engine3DGRUT(
            gs_object=str(args.ckpt),
            mesh_assets_folder=str(REPO_ROOT / "threedgrut_playground" / "assets"),
            default_config="apps/colmap_3dgrt.yaml",
        )
    saved = {name: getattr(engine, name) for name in RESTORED_SETTINGS}
    try:
        return _run(args, engine)
    finally:
        for name, value in saved.items():  # the next render rebuilds the plain BVH once cone tracing is off again
            setattr(engine, name, value)


def _run(args: argparse.Namespace, engine) -> Dict[str, object]:
    out_dir = ensure_outside_repo(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cones = [parse_cone_setting(text) for text in args.cones.split(",")]
    device = torch.device(engine.device)
    preset = load_ladder(engine)
    camera = preset.to_camera(args.resolution, args.resolution, device=device)
    rays = engine.raygen(camera, use_spp=False)
    rays_o, rays_d = rays.rays_ori.contiguous(), rays.rays_dir.contiguous()
    ppd = ppd_from_camera(camera)

    layout = glossy_layout(scene_frame(engine.scene_mog.positions.detach().cpu()), LADDER_PRESET)
    names = layout.object_names()
    object_ids = object_id_map(trace_only_renderer(engine), camera, layout, device)
    boxes = object_boxes(object_ids, names)
    roughness = object_roughness(names)

    even, odd = _load_or_render_reference(
        engine, rays_o, rays_d, args.ref_spp, out_dir / f"reference_unclamped_{args.resolution}px_{args.ref_spp}spp.pt"
    )
    reference = to_display(engine, (even + odd) / 2.0)
    reference_noise = (to_display(engine, even) - to_display(engine, odd)) / 2.0
    save_png(reference, out_dir / "reference.png")

    # Timing. MC frames use the plain BVH; each cone setting rebuilds the inflated one during its warm-ups.
    configure(engine, None)
    mc_ms = median_ms(lambda: render_frame(engine, rays_o, rays_d))
    sample = to_display(engine, render_frame(engine, rays_o, rays_d))
    denoise_ms = median_ms(lambda: denoise(engine, sample))
    cone_ms = {}
    for cone in cones:
        configure(engine, cone)
        cone_ms[cone.name] = median_ms(lambda: render_frame(engine, rays_o, rays_d))
    print(f"MC frame {mc_ms:.2f} ms, denoiser {denoise_ms:.2f} ms", flush=True)

    rows: List[Dict[str, object]] = []
    jods: Dict[str, Dict[str, float]] = {}

    def score(setting: str, method: str, label: str, spp: int, ms: float, image: Tensor, max_roughness: float = 1.0):
        """Append one row per object and return the caption text per object.

        For ``method == "cone"``, objects rougher than ``max_roughness`` are scored as ``"mc fallback"``.
        """
        flip = flip_error_map(image, reference, ppd)
        heat = cvvdp_heatmap(image, reference, ppd) if cvvdp_available() else None
        jod = cvvdp_jod(image, reference, ppd)
        jods.setdefault(setting, {})[label] = float("nan") if jod is None else jod
        captions = {}
        for index, name in enumerate(names):
            mask = object_ids == index
            if not bool(mask.any()):
                continue
            ratio = detail_ratio(image, reference, mask, reference_noise)
            object_method = "mc fallback" if method == "cone" and roughness[name] > max_roughness else method
            row = {
                "setting": setting,
                "method": object_method,
                "column": label,
                "spp": spp,
                "frame_ms": round(ms, 3),
                "object": name,
                "flip": flip[mask].mean().item(),
                "cvvdp_heatmap": float("nan") if heat is None else heat[mask].mean().item(),
                "detail_ratio": ratio,
                "verdict": verdict(object_method, ratio),
            }
            rows.append(row)
            captions[name] = f"FLIP {row['flip']:.3f}  vis {row['cvvdp_heatmap']:.3f}\ndetail {ratio:.2f}  {row['verdict']}"
        return captions

    # Noise left in the reference: one half of its frames against the other half (an upper bound for the full mean).
    half_flip = flip_error_map(to_display(engine, even), to_display(engine, odd), ppd)
    noise_floor = {
        name: half_flip[object_ids == index].mean().item()
        for index, name in enumerate(names)
        if bool((object_ids == index).any())
    }

    configure(engine, None)
    mc1 = to_display(engine, average(engine, rays_o, rays_d, 1, 0))
    summary_settings = {}
    for cone in cones:
        n_eq = equal_time_spp(cone_ms[cone.name], mc_ms)
        n_dn = equal_time_spp(cone_ms[cone.name], mc_ms, denoise_ms)
        configure(engine, None)
        mc_eq = to_display(engine, average(engine, rays_o, rays_d, n_eq, 0))
        mc_dn = denoise(engine, to_display(engine, average(engine, rays_o, rays_d, n_dn, 0)))
        configure(engine, cone)
        engine.frame_id = 0
        cone_image = to_display(engine, render_frame(engine, rays_o, rays_d))

        columns_spec = [  # (key, label, method, spp, frame ms, image)
            ("mc1", "MC 1 spp", "mc", 1, mc_ms, mc1),
            ("eq", f"MC equal time ({n_eq} spp)", "mc", n_eq, n_eq * mc_ms, mc_eq),
            ("dn", f"MC {n_dn} spp + denoiser", "mc+denoiser", n_dn, n_dn * mc_ms + denoise_ms, mc_dn),
            ("cone", f"Cone, cap {cone.max_sigma:g}", "cone", 1, cone_ms[cone.name], cone_image),
        ]
        columns = [("Reference\n", reference, False)]
        captions = {}
        for key, label, method, spp, ms, image in columns_spec:
            header = f"{label}\n{ms:.1f} ms"
            columns.append((header, image, False))
            for name, text in score(cone.name, method, label, spp, ms, image, cone.max_roughness).items():
                captions[(header, name)] = text
        columns += [
            (f"error x{ERROR_GAIN:g}\n{label}", error_image(image, reference), True)
            for key, label, method, spp, ms, image in columns_spec
        ]
        title = (
            f"{cone.name}: single frames, pixel-centre rays; reference = {args.ref_spp} MC frames.  "
            f"Columns 2-4 cost about the same time per frame as the cone (column 5), except MC 1 spp."
        )
        row_labels = {
            name: f"roughness {roughness[name]:g}\n" + ("cone" if roughness[name] <= cone.max_roughness else "MC fallback")
            for name in boxes
        }
        crops_figure(columns, boxes, captions, row_labels, title, out_dir / f"crops_{cone.name}.png")
        full_figure(
            [("Reference", reference)] + [(spec[1], spec[5]) for spec in columns_spec],
            out_dir / f"full_{cone.name}.png",
        )
        for key, label, method, spp, ms, image in columns_spec:
            save_png(image, out_dir / cone.name / f"{key}.png")
        summary_settings[cone.name] = {
            "max_sigma": cone.max_sigma,
            "max_roughness": cone.max_roughness,
            "cone_ms": cone_ms[cone.name],
            "mc_equal_time_spp": n_eq,
            "mc_denoised_spp": n_dn,
            "jod": jods[cone.name],
        }

    with open(out_dir / "results.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "ckpt": str(args.ckpt),
        "resolution": args.resolution,
        "ref_spp": args.ref_spp,
        "ppd": ppd,
        "mc_ms": mc_ms,
        "denoise_ms": denoise_ms,
        "roughness_ladder": list(ROUGHNESS_LADDER),
        "reference_noise_floor": noise_floor,
        "settings": summary_settings,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print_table(rows, summary)
    return summary


def print_table(rows: Sequence[Dict[str, object]], summary: Dict[str, object]) -> None:
    print(f"\nMC frame {summary['mc_ms']:.2f} ms, denoiser {summary['denoise_ms']:.2f} ms, ppd {summary['ppd']:.2f}")
    for name, setting in summary["settings"].items():
        print(f"\n== {name}: cone frame {setting['cone_ms']:.2f} ms -> MC equal time {setting['mc_equal_time_spp']} spp,"
              f" with denoiser {setting['mc_denoised_spp']} spp")
        print("   whole-image CVVDP JOD (10 = identical): "
              + ", ".join(f"{label} {jod:.2f}" for label, jod in setting["jod"].items()))
        print(f"   {'object':<15}{'column':<28}{'FLIP':>7}{'vis':>7}{'detail':>8}  verdict")
        for row in rows:
            if row["setting"] == name:
                print(f"   {row['object']:<15}{row['column']:<28}{row['flip']:7.3f}{row['cvvdp_heatmap']:7.3f}"
                      f"{row['detail_ratio']:8.2f}  {row['verdict']}")
    print("\nreference noise (FLIP of the even against the odd reference frames; the full mean is less noisy):")
    for name, flip in summary["reference_noise_floor"].items():
        print(f"   {name:<15}FLIP {flip:.4f}")


def main(argv=None) -> int:
    run(parse_args(argv))
    return 0
