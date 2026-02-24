import cv2
import numpy as np
from PIL import Image
from histolab.filters.image_filters import Compose, OtsuThreshold, RgbToGrayscale
import slideio
from pathlib import Path

from src.utils.focus_metrics import compute_focus_score
from src.utils.io_utils import suppress_stderr


def detect_tissue_mask(thumbnail_rgb: np.ndarray) -> np.ndarray:
    pipeline = Compose([RgbToGrayscale(), OtsuThreshold()])
    bool_mask = pipeline(Image.fromarray(thumbnail_rgb))
    return (bool_mask * 255).astype(np.uint8)


def generate_tissue_patches(
    width: int, height: int, params: dict, mask: np.ndarray
) -> list[tuple[int, int, float]]:
    patch_size_raw = params["patch_size"] * params["downsample"]
    stride = params["stride"]
    mask_downscale = 8  # hardcoded as per user request for simplicity

    xs = range(0, width - patch_size_raw + 1, stride)
    ys = range(0, height - patch_size_raw + 1, stride)

    mw, mh = patch_size_raw // mask_downscale, patch_size_raw // mask_downscale
    binary = mask > 0

    results = []
    for y in ys:
        for x in xs:
            patch = binary[
                y // mask_downscale : y // mask_downscale + mh,
                x // mask_downscale : x // mask_downscale + mw,
            ]
            if patch.size > 0:
                coverage = float(patch.mean())
                if coverage >= params["cov"]:
                    results.append((x, y, coverage))
    return results


def process_vsi_slide(
    slide_path: str, params: dict, dry_run: bool = False
) -> list[dict]:
    patch_size = params["patch_size"]
    downsample = params["downsample"]
    extent = patch_size * downsample
    mask_downscale = 8

    with suppress_stderr():
        slide = slideio.open_slide(str(slide_path), "VSI")
    scene = slide.get_scene(0)
    width, height = scene.size
    num_z = scene.num_z_slices

    z_res = getattr(scene, "z_resolution", 0)
    if not z_res:
        raise RuntimeError(f"Slide {slide_path} has no Z-res.")
    z_res_microns = z_res * 1e6

    mid_z = num_z // 2
    d_w, d_h = width // mask_downscale, height // mask_downscale
    with suppress_stderr():
        thumb_raw = scene.read_block(
            rect=(0, 0, width, height), size=(d_w, d_h), slices=(mid_z, mid_z + 1)
        )
    mask = detect_tissue_mask(cv2.cvtColor(thumb_raw, cv2.COLOR_BGR2RGB))
    candidates = generate_tissue_patches(width, height, params, mask)

    if not candidates:
        return []
    if dry_run:
        import random

        random.shuffle(candidates)
        candidates = candidates[:20]

    all_scores, best_zs = [], np.zeros(len(candidates), dtype=np.int32)
    slide_name = Path(slide_path).name

    for i, (x, y, _cov) in enumerate(candidates):
        if i % 100 == 0:
            print(f"[{slide_name}] focus {i}/{len(candidates)}")
        with suppress_stderr():
            stack = scene.read_block(
                rect=(x, y, extent, extent),
                size=(patch_size, patch_size),
                slices=(0, num_z),
            )
        if stack.ndim == 3:
            stack = stack[np.newaxis]
        scores = [float(compute_focus_score(stack[z])) for z in range(num_z)]
        all_scores.append(scores)
        best_zs[i] = int(np.argmax(scores))

    return [
        {
            "slide_name": slide_name,
            "x": x,
            "y": y,
            "z_level": z,
            "optimal_z": int(best_z),
            "num_z": num_z,
            "z_res_microns": z_res_microns,
            "z_offset_microns": (int(best_z) - z) * z_res_microns,
            "focus_score": scores[z],
            "max_focus_score": max(scores),
            "tissue_coverage": cov,
        }
        for (x, y, cov), best_z, scores in zip(candidates, best_zs, all_scores)
        for z in range(num_z)
    ]
