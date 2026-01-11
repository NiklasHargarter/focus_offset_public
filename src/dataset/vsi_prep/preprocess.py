import os
import argparse
from pathlib import Path
from typing import Any, Optional
import pickle
import numpy as np
import cv2
import slideio
import multiprocessing
from functools import partial
from skimage.filters import threshold_otsu

import config
from src.utils.io_utils import suppress_stderr
from src.dataset.vsi_types import SlideMetadata, MasterIndex, PreprocessConfig, Patch


class SlidePreprocessor:
    def __init__(self, preprocess_config: PreprocessConfig):
        self.cfg = preprocess_config
        self.patch_size = preprocess_config.patch_size
        self.focus_patch_size = preprocess_config.focus_patch_size
        self.downscale = preprocess_config.downscale_factor
        self.dataset_raw_dir = config.get_vsi_raw_dir(preprocess_config.dataset_name)

    def _compute_focus_score(self, image: np.ndarray) -> float:
        """Compute focus score using Brenner Gradient."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.int32)
        shifted = np.roll(gray, -2, axis=1)
        return float(np.sum((gray - shifted) ** 2))

    def _find_best_global_z(
        self, scene: Any, d_w: int, d_h: int, num_z: int
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """Find the best global Z-slice and return it along with the tissue mask."""
        best_score = -1.0
        best_img = None
        best_z = 0

        # Global read at 16x (or configured downscale)
        for z in range(num_z):
            img = scene.read_block(
                rect=(0, 0, scene.size[0], scene.size[1]),
                size=(d_w, d_h),
                slices=(z, z + 1),
            )
            # Use brenner for global selection as verified default
            # We use a temp calculation here just for the mask source
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Simple score for mask selection
            score = cv2.Laplacian(gray, cv2.CV_64F).var() 
            
            if score > best_score:
                best_score = score
                best_img = gray
                best_z = z

        if best_img is None:
             # Fallback
             best_img = np.zeros((d_h, d_w), dtype=np.uint8)

        # Generate Mask
        thresh = threshold_otsu(best_img)
        mask = ((best_img <= thresh) * 255).astype(np.uint8)
        
        return best_z, best_img, mask

    def _get_valid_indices(self, mask: np.ndarray, d_w: int, d_h: int) -> list[tuple[int, int]]:
        """Identify valid patch coordinates (x, y) based on tissue coverage."""
        valid_patches = []
        ps = self.patch_size
        
        # Grid loop on original coordinates
        for y in range(0, self.height - ps + 1, ps):
            # Map to downscaled mask coordinates
            ds_y = int(y / self.downscale)
            ds_y_end = int((y + ps) / self.downscale)
            if ds_y >= d_h: continue
            ds_y_end = min(ds_y_end, d_h)

            for x in range(0, self.width - ps + 1, ps):
                ds_x = int(x / self.downscale)
                ds_x_end = int((x + ps) / self.downscale)
                if ds_x >= d_w: continue
                ds_x_end = min(ds_x_end, d_w)

                mask_patch = mask[ds_y:ds_y_end, ds_x:ds_x_end]
                if mask_patch.size == 0: continue

                tissue_ratio = np.count_nonzero(mask_patch) / mask_patch.size
                if tissue_ratio >= self.cfg.min_tissue_coverage:
                    valid_patches.append((x, y))
        
        return valid_patches

    def _resolve_z_grid(self, scene: Any, d_w: int, d_h: int, num_z: int) -> np.ndarray:
        """
        Build a 2D grid of optimal Z-levels using Super-Patches.
        Returns: z_grid of shape (grid_h, grid_w)
        """
        fps = self.focus_patch_size
        # Calculate grid dimensions based on CEILING division to cover edges
        grid_w = (self.width + fps - 1) // fps
        grid_h = (self.height + fps - 1) // fps
        
        # Score grid: [grid_h, grid_w, num_z]
        score_grid = np.zeros((grid_h, grid_w, num_z), dtype=np.float32)

        for z in range(num_z):
            # Read full downscaled slide for this Z
            img_z = scene.read_block(
                rect=(0, 0, self.width, self.height),
                size=(d_w, d_h),
                slices=(z, z + 1)
            )
            
            # Vectorized/Grid processing for Super-Patches
            for gy in range(grid_h):
                y = gy * fps
                ds_y = int(y / self.downscale)
                ds_y_end = int((y + fps) / self.downscale)
                ds_y_end = min(ds_y_end, d_h)
                
                if ds_y >= d_h: continue

                for gx in range(grid_w):
                    x = gx * fps
                    ds_x = int(x / self.downscale)
                    ds_x_end = int((x + fps) / self.downscale)
                    ds_x_end = min(ds_x_end, d_w)
                    
                    if ds_x >= d_w: continue
                    
                    # Extract super-patch ROI
                    roi = img_z[ds_y:ds_y_end, ds_x:ds_x_end]
                    if roi.size > 0:
                        score_grid[gy, gx, z] = self._compute_focus_score(roi)
        
        # Find best Z for each grid cell
        best_z_grid = np.argmax(score_grid, axis=2).astype(np.int32)
        return best_z_grid

    def process(self, vsi_path: Path) -> SlideMetadata:
        with suppress_stderr():
            slide = slideio.open_slide(str(vsi_path), "VSI")
        scene = slide.get_scene(0)
        self.width, self.height = scene.size
        num_z = scene.num_z_slices
        
        d_w = self.width // self.downscale
        d_h = self.height // self.downscale

        # 1. Generate Mask (Global Read)
        _, _, mask = self._find_best_global_z(scene, d_w, d_h, num_z)
        
        # 2. Find Valid Data Patches
        valid_coords = self._get_valid_indices(mask, d_w, d_h)
        if not valid_coords:
             print(f"Warning: No tissue found in {vsi_path.name}")
             return SlideMetadata(vsi_path.name, vsi_path, self.width, self.height, num_z, [])

        # 3. Resolve Z-levels via Grid Optimization
        z_grid = self._resolve_z_grid(scene, d_w, d_h, num_z)
        
        # 4. Map Data Patches to Z-Grid
        final_patches = []
        fps = self.focus_patch_size
        
        # Optimize mapping: Direct lookup in 2D grid
        grid_h, grid_w = z_grid.shape
        
        for px, py in valid_coords:
            gx = px // fps
            gy = py // fps
            
            # Boundary safety (should be covered by grid matching)
            gx = min(gx, grid_w - 1)
            gy = min(gy, grid_h - 1)
            
            best_z = int(z_grid[gy, gx])
            final_patches.append(Patch(x=px, y=py, z=best_z))

        print(f"Processed {vsi_path.name}: {len(final_patches)} valid patches")
        return SlideMetadata(
            name=vsi_path.name,
            path=vsi_path,
            width=self.width,
            height=self.height,
            num_z=num_z,
            patches=final_patches,
        )


# --- Worker & Utilities ---

def resolve_file_list(filenames: list[str], dataset_name: str) -> list[Path]:
    valid_files = []
    # Using config to get raw dir
    raw_dir = config.get_vsi_raw_dir(dataset_name)
    for filename in filenames:
        path = raw_dir / filename
        if path.exists():
            valid_files.append(path)
    return sorted(valid_files)

def process_slide_wrapper(vsi_path: Path, config: PreprocessConfig):
    """Wrapper for multiprocessing to instantiate the processor."""
    processor = SlidePreprocessor(config)
    return processor.process(vsi_path)

def preprocess_dataset(
    dataset_name: str,
    patch_size: int,
    downscale_factor: int,
    min_tissue_coverage: float,
    focus_patch_size: int | None = None,
    workers: int | None = None,
    force: bool = False,
) -> None:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if workers is None:
        workers = os.cpu_count()
        
    if focus_patch_size is None:
        focus_patch_size = patch_size * 10

    from src.utils.exact_utils import get_exact_image_list

    print(f"Fetching image list for {dataset_name} from EXACT...")
    images_meta = get_exact_image_list(dataset_name=dataset_name)
    filenames = sorted([img["name"] for img in images_meta])

    current_config = PreprocessConfig(
        patch_size=patch_size,
        focus_patch_size=focus_patch_size,
        downscale_factor=downscale_factor,
        min_tissue_coverage=min_tissue_coverage,
        dataset_name=dataset_name,
    )

    output_path = config.get_master_index_path(
        dataset_name=dataset_name, patch_size=patch_size
    )

    should_generate = force
    if not output_path.exists():
        should_generate = True
    elif not force:
        try:
            with open(output_path, "rb") as f:
                existing_data = pickle.load(f)
            
            if isinstance(existing_data, MasterIndex):
                existing_config = existing_data.config_state
            else: # Legacy dict support
                cfg_dict = existing_data.get("config_state", {})
                existing_config = PreprocessConfig(**cfg_dict)

            if existing_config == current_config:
                print(f"Master index for {dataset_name} exists and matches. Skipping.")
                should_generate = False
            else:
                 print(f"Master index mismatch (Config changed). Regenerating.")
                 should_generate = True
        except Exception:
            should_generate = True

    if not should_generate:
        return

    print(f"\n=== Generating Master Index for {dataset_name} ===")
    files = resolve_file_list(filenames, dataset_name=dataset_name)
    if not files:
        print("No files found on disk. Aborting.")
        return

    # Use wrapper for pool
    process_func = partial(process_slide_wrapper, config=current_config)

    with multiprocessing.Pool(workers) as pool:
        results = pool.map(process_func, files)

    results = [r for r in results if r is not None]

    final_index = MasterIndex(
        file_registry=results,
        patch_size=patch_size,
        config_state=current_config,
    )

    with open(output_path, "wb") as f:
        pickle.dump(final_index, f)

    print(
        f"Saved master index to {output_path} ({len(results)} slides, {final_index.total_samples} patches)"
    )


def calculate_stability_metrics(patches: list, width: int, height: int, patch_size: int):
    """Calculate Total Variation and Outlier Ratio for a Z-map."""
    if not patches:
        return 0.0, 0.0

    xs = sorted(list(set(p.x for p in patches)))
    ys = sorted(list(set(p.y for p in patches)))
    x_map = {x: i for i, x in enumerate(xs)}
    y_map = {y: j for j, y in enumerate(ys)}

    z_grid = np.full((len(ys), len(xs)), np.nan)
    for p in patches:
        z_grid[y_map[p.y], x_map[p.x]] = p.z

    diff_h = np.abs(np.diff(z_grid, axis=1))
    diff_v = np.abs(np.diff(z_grid, axis=0))

    tv = ((np.nanmean(diff_h) + np.nanmean(diff_v)) / 2 if diff_h.size > 0 and diff_v.size > 0 else 0.0)
    
    outliers = np.count_nonzero(diff_h > 2) + np.count_nonzero(diff_v > 2)
    total_diffs = diff_h.size + diff_v.size
    outlier_ratio = outliers / total_diffs if total_diffs > 0 else 0.0

    return float(tv), float(outlier_ratio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--patch_size", type=int, required=True)
    parser.add_argument("--downscale_factor", type=int, required=True)
    parser.add_argument("--min_tissue_coverage", type=float, required=True)
    parser.add_argument("--focus_patch_size", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    preprocess_dataset(
        dataset_name=args.dataset,
        patch_size=args.patch_size,
        downscale_factor=args.downscale_factor,
        min_tissue_coverage=args.min_tissue_coverage,
        focus_patch_size=args.focus_patch_size,
        force=args.force,
    )
