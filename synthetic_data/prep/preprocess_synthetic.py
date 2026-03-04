import random
import numpy as np
import slideio
from pathlib import Path

from focus_offset.utils.focus_metrics import compute_focus_score
from focus_offset.utils.io_utils import suppress_stderr

from shared_datasets.vsi.prep.preprocess import get_tissue_patches


def process_vsi_slide_synthetic(
    slide_path: str,
    patch_size_n: int,
    downsample: int,
    stride: int,
    min_coverage: float,
    dry_run: bool = False,
) -> list[dict]:
    extent = patch_size_n * downsample

    with suppress_stderr():
        slide = slideio.open_slide(str(slide_path), "VSI")
    scene = slide.get_scene(0)
    width, height = scene.size
    num_z = scene.num_z_slices

    z_res = getattr(scene, "z_resolution", 0)
    if not z_res:
        raise RuntimeError(f"Slide {slide_path} has no Z-res.")

    candidates = get_tissue_patches(
        scene=scene,
        patch_size_raw=extent,
        stride=stride,
        min_coverage=min_coverage,
    )

    if not candidates:
        return []

    if dry_run:
        random.shuffle(candidates)
        candidates = candidates[:20]

    slide_name = Path(slide_path).name
    results = []

    for x, y, cov in candidates:
        with suppress_stderr():
            # For focus calculation, we can use a smaller size or the full N.
            # Using patch_size_n itself to match the focus.
            stack = scene.read_block(
                rect=(x, y, extent, extent),
                size=(patch_size_n, patch_size_n),
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

        # We ONLY yield one row per spatial location.
        # Properties like num_z and z_res_microns are easily fetched dynamically inside the Dataset.
        results.append(
            {
                "slide_name": slide_name,
                "x": x,
                "y": y,
                "optimal_z": best_z,
                "max_focus_score": max_score,
                "tissue_coverage": cov,
            }
        )

    return results
