import os
import argparse
import multiprocessing
import pickle
import json
from functools import partial
from pathlib import Path
from typing import Any, Tuple, List
import cv2
import numpy as np
import tifffile
from dataclasses import asdict

from src import config
from src.dataset.vsi_types import SlideMetadata, PreprocessConfig, MasterIndex
from src.utils.focus_metrics import compute_brenner_gradient

MASK_DOWNSCALE = 16

class OMEImageReader:
    """Uses tifffile to read multi-series OME-TIFFs as Z-stacks."""
    def __init__(self, path: Path):
        self.path = path
        self._tif = tifffile.TiffFile(path)
        self.num_z = len(self._tif.series)
        # Assume all series have the same shape for now (AgNor case)
        first_series = self._tif.series[0]
        self.height, self.width = first_series.shape[:2]
        self.size = (self.width, self.height)

    def read_block(self, rect: Tuple[int, int, int, int], slices: Tuple[int, int] = None) -> np.ndarray:
        """
        Reads a block from the multi-series image.
        rect: (x, y, w, h)
        slices: (z_start, z_end)
        Returns: (Z, H, W, C)
        """
        x, y, w, h = rect
        z_start, z_end = slices if slices else (0, self.num_z)
        
        stack = []
        for z in range(z_start, z_end):
            # Read the series and crop
            data = self._tif.series[z].asarray()
            # data is (H, W, C) or (H, W)
            if data.ndim == 2:
                data = np.expand_dims(data, axis=-1)
            
            crop = data[y:y+h, x:x+w]
            stack.append(crop)
            
        return np.stack(stack)

    def close(self):
        self._tif.close()

def detect_tissue(reader: OMEImageReader) -> Tuple[int, np.ndarray]:
    """Find the sharpest slice and generate a tissue mask."""
    width, height = reader.size
    num_z = reader.num_z
    d_w, d_h = width // MASK_DOWNSCALE, height // MASK_DOWNSCALE

    print(f"  [DEBUG] Scanning slices for sharpest thumbnail...")
    best_score, best_img, best_z = -1.0, None, 0
    
    # Sample slices
    for z in range(0, num_z):
        # Read full image downsampled if possible, but for these 2k x 2k, just read and resize
        full_img = reader.read_block((0, 0, width, height), (z, z + 1))[0]
        img_small = cv2.resize(full_img, (d_w, d_h))
        score = compute_brenner_gradient(img_small)
        if score > best_score:
            best_score, best_img, best_z = score, img_small, z

    if best_img is None:
        return 0, np.zeros((d_h, d_w), dtype=np.uint8)

    gray = cv2.cvtColor(best_img, cv2.COLOR_BGR2GRAY)
    # Since these are often FOVs, Otsu might be tricky if background is nearly uniform
    # but we'll try it.
    try:
        from skimage.filters import threshold_otsu
        thresh = threshold_otsu(gray)
        mask = ((gray <= thresh) * 255).astype(np.uint8)
    except:
        # Fallback to simple threshold if Otsu fails
        mask = ((gray < 200) * 255).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return best_z, mask

def generate_patch_candidates(
    mask: np.ndarray,
    width_raw: int,
    height_raw: int,
    patch_size_raw: int,
    stride_raw: int,
    min_cov: float,
) -> List[Tuple[int, int]]:
    candidates = []
    for y in range(0, height_raw - patch_size_raw + 1, stride_raw):
        for x in range(0, width_raw - patch_size_raw + 1, stride_raw):
            mx, my = x // MASK_DOWNSCALE, y // MASK_DOWNSCALE
            mw, mh = patch_size_raw // MASK_DOWNSCALE, patch_size_raw // MASK_DOWNSCALE
            mask_patch = mask[my : my + mh, mx : mx + mw]
            if mask_patch.size == 0: continue
            if np.mean(mask_patch > 0) >= min_cov:
                candidates.append((x, y))
    return candidates

class OMEPreprocessor:
    def __init__(self, config: PreprocessConfig):
        self.cfg = config
        self.ds = self.cfg.downsample_factor

    def process(self, img_path: Path) -> SlideMetadata:
        reader = OMEImageReader(img_path)
        width_raw, height_raw = reader.size
        num_z = reader.num_z

        print(f"[{img_path.name}] Stage 1: Tissue Masking...")
        _, mask = detect_tissue(reader)

        print(f"[{img_path.name}] Stage 2: Grid Generation...")
        raw_patch_size = self.cfg.patch_size * self.ds
        raw_stride = self.cfg.stride * self.ds

        candidates = generate_patch_candidates(
            mask, width_raw, height_raw, raw_patch_size, raw_stride, self.cfg.min_tissue_coverage
        )

        total_patches = len(candidates)
        print(f"[{img_path.name}] Stage 3: Focus Search ({total_patches} patches)...")

        if total_patches == 0:
            reader.close()
            return SlideMetadata(
                name=img_path.name, width=width_raw, height=height_raw, num_z=num_z,
                patches=np.array([], dtype=np.int32).reshape(0, 3)
            )

        best_zs = np.zeros(total_patches, dtype=np.int32)
        for i, (ox, oy) in enumerate(candidates):
            z_stack = reader.read_block(
                rect=(ox, oy, raw_patch_size, raw_patch_size),
                slices=(0, num_z)
            )
            # z_stack is (Z, H_raw, W_raw, C)
            patch_scores = []
            for z in range(num_z):
                # Resize to target patch size for focus metric consistent with training
                patch_img = cv2.resize(z_stack[z], (self.cfg.patch_size, self.cfg.patch_size))
                score = compute_brenner_gradient(patch_img)
                patch_scores.append(score)
            best_zs[i] = np.argmax(patch_scores)

        reader.close()
        c_arr = np.array(candidates)
        final_patches = np.column_stack([c_arr[:, 0], c_arr[:, 1], best_zs]).astype(np.int32)

        return SlideMetadata(
            name=img_path.name, width=width_raw, height=height_raw, num_z=num_z, patches=final_patches
        )

def process_image_wrapper(img_path: Path, config: PreprocessConfig):
    try:
        return OMEPreprocessor(config).process(img_path)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def preprocess_dataset(
    dataset_name: str,
    patch_size: int,
    stride: int,
    downsample_factor: int,
    min_tissue_coverage: float,
    limit: int | None = None,
    workers: int | None = None,
    force: bool = False,
) -> None:
    # Use DATA_ROOT override if present, same as VSI
    raw_dir = config.get_vsi_raw_dir(dataset_name)
    manifest_path = config.get_master_index_path(dataset_name, patch_size)
    indices_dir = config.get_slide_index_dir(dataset_name, patch_size)
    indices_dir.mkdir(parents=True, exist_ok=True)

    workers = workers or os.cpu_count() or 1
    current_config = PreprocessConfig(
        patch_size=patch_size, stride=stride, downsample_factor=downsample_factor,
        min_tissue_coverage=min_tissue_coverage, dataset_name=dataset_name
    )

    existing_results = []
    processed_names = set()

    if manifest_path.exists() and not force:
        with open(manifest_path, "rb") as f:
            manifest_data = pickle.load(f)
        if manifest_data.config_state == current_config:
            for pkl_path in sorted(indices_dir.glob("*.pkl")):
                with open(pkl_path, "rb") as f:
                    res = pickle.load(f)
                existing_results.append(res)
                processed_names.add(res.name)

    all_files = sorted(list(raw_dir.glob("*.ome.tiff")) + list(raw_dir.glob("*.ome.tif")))
    if limit:
        all_files = all_files[:limit]

    files_to_process = [f for f in all_files if f.name not in processed_names]
    print(f"Preprocessing OME-TIFF {dataset_name} (Total: {len(all_files)}, Remaining: {len(files_to_process)})")

    if files_to_process:
        with multiprocessing.Pool(workers) as pool:
            process_func = partial(process_image_wrapper, config=current_config)
            for result in pool.imap_unordered(process_func, files_to_process):
                if result is not None:
                    slide_pkl_path = indices_dir / f"{result.name}.pkl"
                    with open(slide_pkl_path, "wb") as f:
                        pickle.dump(result, f)
                    existing_results.append(result)

    if existing_results:
        master_index = MasterIndex(
            file_registry=sorted(existing_results, key=lambda x: x.name),
            patch_size=patch_size,
            config_state=current_config
        )
        with open(manifest_path, "wb") as f:
            pickle.dump(master_index, f)
        print(f"Master index saved to {manifest_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=448) # For 2k x 2k FOVs, stride can be larger or same
    parser.add_argument("--downsample_factor", type=int, default=1) # AgNor might not need downsampling?
    parser.add_argument("--min_tissue_coverage", type=float, default=0.01)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    preprocess_dataset(
        args.dataset, args.patch_size, args.stride, args.downsample_factor,
        args.min_tissue_coverage, limit=args.limit, workers=args.workers, force=args.force
    )
