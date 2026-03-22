import random
from pathlib import Path

import numpy as np
import slideio

from focus_offset.utils.focus_metrics import compute_focus_score
from focus_offset.utils.io_utils import suppress_stderr

from shared_datasets.vsi.prep.preprocess import get_tissue_patches, DRY_RUN_MAX_PATCHES


def process_vsi_slide_synthetic(
    slide_path: str,
    patch_size: int,
    downsample: int,
    min_coverage: float,
    dry_run: bool,
) -> list[dict]:
    with suppress_stderr():
        slide = slideio.open_slide(str(slide_path), "VSI")
    scene = slide.get_scene(0)
    num_z = scene.num_z_slices
    slide_name = Path(slide_path).name

    print(f"[{slide_name}] Starting preprocessing...", flush=True)

    candidates = get_tissue_patches(
        scene=scene,
        patch_size=patch_size,
        downsample=downsample,
        min_coverage=min_coverage,
        target_downscale=8,
        return_dropped=False,
    )

    if not candidates:
        return []

    if dry_run:
        random.shuffle(candidates)
        candidates = candidates[:DRY_RUN_MAX_PATCHES]

    results = []
    for x, y, cov in candidates:
        with suppress_stderr():
            stack = scene.read_block(
                rect=(x, y, patch_size * downsample, patch_size * downsample),
                size=(patch_size, patch_size),
                slices=(0, num_z),
            )

        scores = [compute_focus_score(stack[z]) for z in range(num_z)]
        best_z = int(np.argmax(scores))

        results.append(
            {
                "slide_name": slide_name,
                "x": x,
                "y": y,
                "optimal_z": best_z,
                "num_z": num_z,
                "max_focus_score": scores[best_z],
                "tissue_coverage": cov,
            }
        )

    print(f"[{slide_name}] Finished. Generated {len(results)} patches.", flush=True)
    return results
