# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Primary-ray cones through the Gaussian LOD: prefiltered images at the cost of one ray?

Gaussians only (the room checkpoint, every mesh removed), one camera with a fixed 60 degree field of view, rendered at
several output resolutions R. A lower R means a wider pixel footprint, as in the periphery. Per R:

- ``plain 1 spp``: one thin ray through each pixel centre (cone off): cheap, aliased.
- ``plain s x s``: rendered at R*s and box-averaged to R: the brute-force prefilter, s^2 rays per pixel.
- ``cone``: one cone per pixel with the pixel's footprint (std = pixel width / sqrt 12, growing with distance), traced
  through the Gaussian LOD (cone on, LOD on, every Gaussian segment a cone).
- ``reference``: rendered at 4096 px (in row chunks) and box-averaged to R.

Times are CUDA-synchronized medians of one-launch frames. Quality is CVVDP JOD and FLIP against the reference with the
display geometry R px over 60 degrees (ppd = R / 60): each resolution is judged as shown at its own size.

Usage: python scripts/cone/primary_footprint.py --out C:/Users/ville/3dgrut_runs/primary_footprint
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from threedgrut_playground.adaptive.evaluation.cone_equal_time import (
    ConeSetting,
    configure,
    load_ladder,
    render_frame,
    to_display,
)
from threedgrut_playground.adaptive.evaluation.ldr import ensure_outside_repo, save_png
from threedgrut_playground.adaptive.evaluation.metrics import cvvdp_jod, flip_error_map

FOV = 60.0
RESOLUTIONS = (512, 256, 128, 64)
SUPERSAMPLE = (2, 4, 8)
REF_RES = 4096
CHUNK_ROWS = 512
CONE_CAP = 0.64  # the top LOD band: no camera footprint here needs more


def log(text):
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", type=Path, default=REPO_ROOT / "runs/room_3dgrt/room-0211_151617/ckpt_last.pt")
    p.add_argument("--out", type=Path, default=Path("C:/Users/ville/3dgrut_runs/primary_footprint"))
    args = p.parse_args(argv)
    out = ensure_outside_repo(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from threedgrut_playground.engine import Engine3DGRUT

    e = Engine3DGRUT(
        gs_object=str(args.ckpt),
        mesh_assets_folder=str(REPO_ROOT / "threedgrut_playground" / "assets"),
        default_config="apps/colmap_3dgrt.yaml",
    )
    dev = torch.device(e.device)
    preset = load_ladder(e)
    for name in list(e.primitives.objects):
        e.primitives.remove_primitive(name)

    def rays(res):
        r = e.raygen(preset.to_camera(res, res, device=dev), use_spp=False)
        return r.rays_ori.contiguous(), r.rays_dir.contiguous()

    def set_mode(cone):
        configure(e, ConeSetting(CONE_CAP, 1.0) if cone else None)
        e.use_gaussian_lod = bool(cone)
        e.lod_widen = cone != "level"
        e.cone_camera_rays = True
        e.camera_fov = preset.fov_deg

    def render(res, cone, timed=True):
        """(ms, linear image [res, res, 3]); renders in row chunks above CHUNK_ROWS rows. The cone footprint comes from
        camera_fov / rows, so chunks get the fov scaled to their row count."""
        set_mode(cone)
        o, d = rays(res)
        chunks = [(y, min(y + CHUNK_ROWS, res)) for y in range(0, res, CHUNK_ROWS)]

        def frame():
            parts = []
            for y0, y1 in chunks:
                e.camera_fov = preset.fov_deg * (y1 - y0) / res
                e.frame_id = 0
                parts.append(render_frame(e, o[:, y0:y1].contiguous(), d[:, y0:y1].contiguous()))
            e.camera_fov = preset.fov_deg
            return torch.cat(parts, dim=0)

        img = frame()
        if not timed:
            return float("nan"), img
        for _ in range(2):
            frame()
        torch.cuda.synchronize()
        times = []
        for _ in range(7 if res * res <= 1024 * 1024 else 3):
            t0 = time.perf_counter()
            frame()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
        return statistics.median(times), img

    def box(img, f):
        return F.avg_pool2d(img.permute(2, 0, 1)[None], f)[0].permute(1, 2, 0) if f > 1 else img

    log(f"reference: {REF_RES} px plain, box-averaged to each resolution")
    _, ref_full = render(REF_RES, False, timed=False)
    results, images = {}, {}
    for res in RESOLUTIONS:
        ppd = res / FOV
        ref = to_display(e, box(ref_full, REF_RES // res))
        rows = []

        def score(label, ms, img, rays_per_pixel):
            disp = to_display(e, img)
            rows.append(
                {
                    "method": label,
                    "ms": ms,
                    "rays_per_pixel": rays_per_pixel,
                    "jod": cvvdp_jod(disp, ref, ppd),
                    "flip": flip_error_map(disp, ref, ppd).mean().item(),
                    "psnr": (-10.0 * torch.log10(((disp - ref) ** 2).mean())).item(),
                }
            )
            images[(res, label)] = disp
            r = rows[-1]
            log(
                f"{res:>4} px  {label:<14} {ms:8.2f} ms  JOD {r['jod']:.2f}  FLIP {r['flip']:.3f}  PSNR {r['psnr']:.2f}"
            )

        ms, img = render(res, False)
        score("plain 1 spp", ms, img, 1)
        for s in SUPERSAMPLE:
            if res * s > REF_RES // 2:
                continue
            ms, img = render(res * s, False)
            score(f"plain {s * s} spp", ms, box(img, s), s * s)
        ms, img = render(res, True)
        score("cone (LOD)", ms, img, 1)
        ms, img = render(res, "level")
        score("LOD level only", ms, img, 1)
        results[res] = {"ppd": ppd, "rows": rows}
        images[(res, "reference")] = ref
    (out / "results.json").write_text(json.dumps(results, indent=2))

    # Figure: for each resolution, reference / plain 1 / plain 16 / cone, all upscaled to 512 (nearest)
    from PIL import Image, ImageDraw, ImageFont

    def font(size):
        try:
            return ImageFont.truetype("arial.ttf", size)
        except OSError:
            return ImageFont.load_default()

    cols = ["reference", "plain 1 spp", "plain 4 spp", "plain 16 spp", "cone (LOD)", "LOD level only"]
    cell, pad, lab_w, cap_h = 384, 6, 120, 40
    canvas = Image.new("RGB", (lab_w + len(cols) * (cell + pad), len(RESOLUTIONS) * (cell + cap_h + pad)), "white")
    draw = ImageDraw.Draw(canvas)
    for r_i, res in enumerate(RESOLUTIONS):
        y = r_i * (cell + cap_h + pad)
        draw.multiline_text((6, y + cell // 2), f"{res} px\n{res / FOV:.1f} ppd", fill="black", font=font(18))
        for c_i, label in enumerate(cols):
            img = images.get((res, label))
            if img is None:
                continue
            row = next((r for r in results[res]["rows"] if r["method"] == label), None)
            cap = label if row is None else f"{label}: {row['ms']:.1f} ms, JOD {row['jod']:.2f}"
            x = lab_w + c_i * (cell + pad)
            draw.text((x + 2, y + 4), cap, fill="black", font=font(14))
            pil = Image.fromarray((img.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype("uint8"))
            # centre crop: the middle half of the frame, upscaled to the cell
            h = pil.size[0]
            pil = pil.crop((h // 4, h // 4, 3 * h // 4, 3 * h // 4)).resize((cell, cell), Image.NEAREST)
            canvas.paste(pil, (x, y + cap_h - 6))
    canvas.save(out / "footprint_grid.png")
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
