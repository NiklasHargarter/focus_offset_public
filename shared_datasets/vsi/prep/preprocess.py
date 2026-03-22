import cv2
import random

import numpy as np

import slideio
from pathlib import Path

from focus_offset.utils.focus_metrics import compute_focus_score
from focus_offset.utils.io_utils import suppress_stderr

# Tissue detection constants
MASK_DOWNSCALE = 8
GAUSSIAN_BLUR_KERNEL = (5, 5)
MORPH_KERNEL_SIZE = (7, 7)
MORPH_ITERATIONS = 3

# Conversion and fallback constants
MICRONS_CONVERSION = 1e6

# Dry-run limits
DRY_RUN_MAX_PATCHES = 20


def get_tissue_patches(
    scene,
    patch_size: int,
    downsample: int,
    min_coverage: float,
    target_downscale: int,
    return_dropped: bool,
) -> (
    list[tuple[int, int, float]]
    | tuple[list[tuple[int, int, float]], list[tuple[int, int, float]]]
):
    """Generates an HSV-Otsu tissue mask from a thumbnail and returns spatial patching coordinates."""
    patch_size_raw = patch_size * downsample
    width, height = scene.size
    num_z = scene.num_z_slices
    mid_z = num_z // 2

    # Calculate target dimensions for the mask (approx target_downscale)
    d_w, d_h = width // target_downscale, height // target_downscale

    with suppress_stderr():
        thumb_raw = scene.read_block(
            rect=(0, 0, width, height), size=(d_w, d_h), slices=(mid_z, mid_z + 1)
        )
    thumbnail_rgb = cv2.cvtColor(thumb_raw, cv2.COLOR_BGR2RGB)

    # 1. Convert to grayscale
    gray = cv2.cvtColor(thumbnail_rgb, cv2.COLOR_RGB2GRAY)

    # 2. Blur slightly to ignore dust/scratches
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL, 0)

    # 3. Invert: Glass (255) becomes 0, Tissue (darker) becomes bright
    inverted = cv2.bitwise_not(blurred)

    # 4. Otsu on the inverted "brightness"
    # This captures the faint blue counterstain of IHC much better than Saturation
    _, mask = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. Morphological Closing to connect fragmented IHC pieces
    kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERATIONS)

    mask_scale = width / mask.shape[1]

    mask_patch_size = int(patch_size_raw / mask_scale)

    results = []
    dropped = []
    # Iterate through the grid at Level 0
    for y in range(0, height - patch_size_raw + 1, patch_size_raw):
        for x in range(0, width - patch_size_raw + 1, patch_size_raw):
            # Map Level 0 coords to Mask coords
            mx, my = int(x / mask_scale), int(y / mask_scale)
            mask_roi = mask[my : my + mask_patch_size, mx : mx + mask_patch_size]

            # 1. Pragmatic Tissue Filter (our mask is 0/255 from otsu)
            if mask_roi.size > 0:
                cov = float(np.mean(mask_roi) / 255.0)
                if cov >= min_coverage:
                    results.append((x, y, cov))
                else:
                    dropped.append((x, y, cov))

    if return_dropped:
        return results, dropped
    return results


def process_vsi_slide(
    slide_path: str,
    patch_size: int,
    downsample: int,
    min_coverage: float,
    dry_run: bool,
) -> list[dict]:
    with suppress_stderr():
        slide = slideio.open_slide(str(slide_path), "VSI")
    scene = slide.get_scene(0)
    width, height = scene.size
    num_z = scene.num_z_slices

    z_res = getattr(scene, "z_resolution", 0)
    if not z_res:
        raise RuntimeError(f"Slide {slide_path} has no Z-res.")
    z_res_microns = z_res * MICRONS_CONVERSION

    candidates = get_tissue_patches(
        scene=scene,
        patch_size=patch_size,
        downsample=downsample,
        min_coverage=min_coverage,
        target_downscale=MASK_DOWNSCALE,
        return_dropped=False,
    )

    if not candidates:
        return []
    if dry_run:
        random.shuffle(candidates)
        candidates = candidates[:DRY_RUN_MAX_PATCHES]

    slide_name = Path(slide_path).name
    results = []

    for x, y, cov in candidates:
        with suppress_stderr():
            stack = scene.read_block(
                rect=(
                    x,
                    y,
                    patch_size * downsample,
                    patch_size * downsample,
                ),
                size=(patch_size, patch_size),
                slices=(0, num_z),
            )

        if stack.ndim != 4:
            raise ValueError(
                f"Expected Z-stack with 4 dimensions for {slide_name}, got {stack.ndim}"
            )

        scores = [compute_focus_score(stack[z]) for z in range(num_z)]

        best_z = int(
            np.argmax(scores)
        )  # leaving int cast to prevent np.int64 types leaking
        max_score = scores[best_z]

        for z in range(num_z):
            results.append(
                {
                    "slide_name": slide_name,
                    "x": x,
                    "y": y,
                    "z_level": z,
                    "optimal_z": best_z,
                    "num_z": num_z,
                    "z_res_microns": z_res_microns,
                    "z_offset_microns": (best_z - z) * z_res_microns,
                    "focus_score": scores[z],
                    "max_focus_score": max_score,
                    "tissue_coverage": cov,
                }
            )

    return results
