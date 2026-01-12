import os
import argparse
import multiprocessing
import pickle
from functools import partial
from pathlib import Path
from skimage.filters import threshold_otsu
from typing import Any
import cv2
import numpy as np
import slideio
from tqdm import tqdm

import config
from src.dataset.vsi_types import SlideMetadata, Patch, PreprocessConfig, MasterIndex
from src.utils.focus_metrics import compute_brenner_gradient
from src.utils.io_utils import suppress_stderr


def detect_tissue(scene: Any, downscale: int) -> tuple[int, np.ndarray]:
    """
    Generate a binary tissue mask for the entire slide.
    1. Finds the physically sharpest slice using Brenner Gradient.
    2. Performs Otsu thresholding on that slice to create the mask.
    """
    width, height = scene.size
    num_z = scene.num_z_slices
    d_w, d_h = width // downscale, height // downscale

    best_score, best_img, best_z = -1.0, None, 0
    for z in range(num_z):
        img = scene.read_block(rect=(0, 0, width, height), size=(d_w, d_h), slices=(z, z + 1))
        # Use our standard focus metric to find the best slice for masking
        score = compute_brenner_gradient(img)
        if score > best_score:
            best_score, best_img, best_z = score, img, z

    if best_img is None:
         gray = np.zeros((d_h, d_w), dtype=np.uint8)
    else:
         gray = cv2.cvtColor(best_img, cv2.COLOR_BGR2GRAY)

    # Otsu thresholding to distinguish tissue from background
    thresh = threshold_otsu(gray)
    mask = ((gray <= thresh) * 255).astype(np.uint8)
    
    return best_z, mask


def create_focal_map(scene: Any, focus_patch_size: int, downscale: int) -> np.ndarray:
    """
    Map the focal plane of the slide using large Super-Patches.
    Returns a coarse grid of best Z-levels.
    """
    width, height = scene.size
    num_z = scene.num_z_slices
    
    grid_w = (width + focus_patch_size - 1) // focus_patch_size
    grid_h = (height + focus_patch_size - 1) // focus_patch_size
    
    score_grid = np.zeros((grid_h, grid_w, num_z), dtype=np.float32)

    d_w, d_h = width // downscale, height // downscale
    for z in range(num_z):
        img_z = scene.read_block(rect=(0, 0, width, height), size=(d_w, d_h), slices=(z, z + 1))
        for gy in range(grid_h):
            y, ye = int(gy * focus_patch_size / downscale), int((gy+1) * focus_patch_size / downscale)
            for gx in range(grid_w):
                x, xe = int(gx * focus_patch_size / downscale), int((gx+1) * focus_patch_size / downscale)
                roi = img_z[y:min(ye, d_h), x:min(xe, d_w)]
                if roi.size > 0:
                    score_grid[gy, gx, z] = compute_brenner_gradient(roi)
    
    return np.argmax(score_grid, axis=2).astype(np.int32)


def generate_training_patches(
    mask: np.ndarray, 
    focus_map: np.ndarray, 
    width: int, 
    height: int, 
    patch_size: int, 
    binning_factor: int,
    focus_patch_size: int,
    downscale: int, 
    min_cov: float
) -> list[Patch]:
    """Iterate through slide, filter by mask, and assign Z from focus map."""
    patches = []
    m_h, m_w = mask.shape
    
    # Area covered by one training patch on the slide
    stride = patch_size * binning_factor
    
    for y in range(0, height - stride + 1, stride):
        my, mye = int(y / downscale), int((y + stride) / downscale)
        for x in range(0, width - stride + 1, stride):
            mx, mxe = int(x / downscale), int((x + stride) / downscale)

            mask_patch = mask[my:min(mye, m_h), mx:min(mxe, m_w)]
            if mask_patch.size > 0 and (np.count_nonzero(mask_patch) / mask_patch.size) >= min_cov:
                # Look up Z from the coarse grid (using center of the patch area)
                center_x, center_y = x + stride // 2, y + stride // 2
                gx, gy = center_x // focus_patch_size, center_y // focus_patch_size
                z = int(focus_map[min(gy, focus_map.shape[0]-1), min(gx, focus_map.shape[1]-1)])
                patches.append(Patch(x=x, y=y, z=z))
    
    return patches


class SlidePreprocessor:
    def __init__(self, config: PreprocessConfig):
        self.cfg = config

    def process(self, vsi_path: Path) -> SlideMetadata:
        with suppress_stderr():
            slide = slideio.open_slide(str(vsi_path), "VSI")
        scene = slide.get_scene(0)
        width, height = scene.size
        
        # STAGE 1: Create a correct Focus Map (Stable, Global)
        # We use a 4x downscale and large Super-Patches as the foundation of truth.
        print(f"[{vsi_path.name}] Stage 1: Mapping Focus Plane...")
        focus_map = create_focal_map(scene, self.cfg.focus_patch_size, self.cfg.downscale_factor)
        
        # STAGE 2: Generate Training Patches (Filtered, Binned)
        # We use the Best-Z from Stage 1 as the reference for Stage 2.
        print(f"[{vsi_path.name}] Stage 2: Extracting training targets...")
        _, mask = detect_tissue(scene, self.cfg.downscale_factor)
        
        final_patches = generate_training_patches(
            mask, focus_map, width, height,
            self.cfg.patch_size, self.cfg.binning_factor,
            self.cfg.focus_patch_size, self.cfg.downscale_factor,
            self.cfg.min_tissue_coverage
        )

        print(f"Processed {vsi_path.name}: {len(final_patches)} patches (binning={self.cfg.binning_factor})")
        return SlideMetadata(vsi_path.name, vsi_path, width, height, scene.num_z_slices, final_patches)


def preprocess_dataset(
    dataset_name: str,
    patch_size: int,
    binning_factor: int,
    downscale_factor: int,
    min_tissue_coverage: float,
    focus_patch_size: int | None = None,
    workers: int | None = None,
    force: bool = False,
) -> None:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    workers = workers or os.cpu_count()
    focus_patch_size = focus_patch_size or (patch_size * binning_factor * 2)
    
    current_config = PreprocessConfig(
        patch_size=patch_size,
        binning_factor=binning_factor,
        focus_patch_size=focus_patch_size,
        downscale_factor=downscale_factor,
        min_tissue_coverage=min_tissue_coverage,
        dataset_name=dataset_name,
    )

    output_path = config.get_master_index_path(dataset_name, patch_size)

    if not force and output_path.exists():
        try:
            with open(output_path, "rb") as f:
                existing = pickle.load(f)
            if existing.config_state == current_config:
                print(f"Index for {dataset_name} matches. Skipping.")
                return
        except Exception:
            pass

    from src.utils.exact_utils import get_exact_image_list
    images_meta = get_exact_image_list(dataset_name=dataset_name)
    raw_dir = config.get_vsi_raw_dir(dataset_name)
    files = sorted([raw_dir / img["name"] for img in images_meta if (raw_dir / img["name"]).exists()])

    if not files: print("No files found. Aborting."); return

    print(f"\n=== Preprocessing Dataset: {dataset_name} ===")
    with multiprocessing.Pool(workers) as pool:
        process_func = partial(process_slide_wrapper, config=current_config)
        results = pool.map(process_func, files)

    results = [r for r in results if r is not None]
    final_index = MasterIndex(file_registry=results, patch_size=patch_size, config_state=current_config)

    with open(output_path, "wb") as f:
        pickle.dump(final_index, f)
    print(f"Final Count: {len(results)} slides, {final_index.total_samples} patches")


def process_slide_wrapper(vsi_path: Path, config: PreprocessConfig):
    return SlidePreprocessor(config).process(vsi_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--patch_size", type=int, required=True)
    parser.add_argument("--binning_factor", type=int, default=1)
    parser.add_argument("--downscale_factor", type=int, default=4)
    parser.add_argument("--min_tissue_coverage", type=float, required=True)
    parser.add_argument("--focus_patch_size", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    preprocess_dataset(
        args.dataset, args.patch_size, args.binning_factor, args.downscale_factor,
        args.min_tissue_coverage, args.focus_patch_size, force=args.force
    )
