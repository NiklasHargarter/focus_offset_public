import os
import argparse
import multiprocessing
import pickle
from functools import partial
from pathlib import Path
from skimage.filters import threshold_otsu
from typing import Any, Tuple, List
import cv2
import numpy as np
import slideio
import json
from dataclasses import asdict

from src.utils.io_utils import suppress_stderr
from src import config
from src.dataset.vsi_types import SlideMetadata, PreprocessConfig, MasterIndex
from src.utils.focus_metrics import compute_brenner_gradient

MASK_DOWNSCALE = 16


def detect_tissue(scene: Any) -> Tuple[int, np.ndarray]:
    """Find the sharpest slice and generate a tissue mask for sparse filtering."""
    width, height = scene.size
    num_z = scene.num_z_slices
    d_w, d_h = width // MASK_DOWNSCALE, height // MASK_DOWNSCALE

    print(f"  [DEBUG] Scanning Z-slices for sharpest thumbnail...")
    best_score, best_img, best_z = -1.0, None, 0
    # Sample every 3rd slice to find a good reference image for the mask
    for z in range(0, num_z, 3):
        img = scene.read_block(
            rect=(0, 0, width, height), size=(d_w, d_h), slices=(z, z + 1)
        )
        score = compute_brenner_gradient(img)
        if score > best_score:
            best_score, best_img, best_z = score, img, z

    if best_img is None:
        return 0, np.zeros((d_h, d_w), dtype=np.uint8)

    gray = cv2.cvtColor(best_img, cv2.COLOR_BGR2GRAY)
    thresh = threshold_otsu(gray)

    # Tissue is usually darker than the slide background
    mask = ((gray <= thresh) * 255).astype(np.uint8)

    # Basic Morphological Opening to remove small noise (dust)
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
    """Return top-left coordinates for grid patches that contain enough tissue."""
    candidates = []
    m_h, m_w = mask.shape

    for y in range(0, height_raw - patch_size_raw + 1, stride_raw):
        for x in range(0, width_raw - patch_size_raw + 1, stride_raw):
            # Map raw coordinates to mask coordinates
            mx = int(x / MASK_DOWNSCALE)
            my = int(y / MASK_DOWNSCALE)
            mw = int(patch_size_raw / MASK_DOWNSCALE)
            mh = int(patch_size_raw / MASK_DOWNSCALE)

            # Crop mask region
            mask_patch = mask[my : my + mh, mx : mx + mw]
            if mask_patch.size == 0:
                continue

            coverage = np.mean(mask_patch > 0)
            if coverage >= min_cov:
                candidates.append((x, y))

    return candidates


class SlidePreprocessor:
    def __init__(self, config: PreprocessConfig):
        self.cfg = config
        self.ds = self.cfg.downsample_factor

    def process(self, vsi_path: Path) -> SlideMetadata:
        with suppress_stderr():
            slide = slideio.open_slide(str(vsi_path), "VSI")
        scene = slide.get_scene(0)
        width_raw, height_raw = scene.size
        num_z = scene.num_z_slices

        print(f"[{vsi_path.name}] Stage 1: Tissue Masking...")
        _, mask = detect_tissue(scene)

        print(f"[{vsi_path.name}] Stage 2: Grid Generation...")
        raw_patch_size = self.cfg.patch_size * self.ds
        raw_stride = self.cfg.stride * self.ds

        candidates = generate_patch_candidates(
            mask,
            width_raw,
            height_raw,
            raw_patch_size,
            raw_stride,
            self.cfg.min_tissue_coverage,
        )

        total_patches = len(candidates)
        print(f"[{vsi_path.name}] Stage 3: Focus Search ({total_patches} patches)...")

        if total_patches == 0:
            return SlideMetadata(
                name=vsi_path.name,
                width=width_raw,
                height=height_raw,
                num_z=num_z,
                patches=np.array([], dtype=np.int32).reshape(0, 3),
            )

        best_zs = np.zeros(total_patches, dtype=np.int32)

        # Read each patch as a full Z-stack (S, H, W, C)
        # This is much faster than reading full slides or large ROIs in Slideio
        for i, (ox, oy) in enumerate(candidates):
            if i % 100 == 0:
                print(f"  [{vsi_path.name}] Processing patch {i}/{total_patches}...")

            with suppress_stderr():
                # Read the full Z-stack for this patch
                z_stack = scene.read_block(
                    rect=(ox, oy, raw_patch_size, raw_patch_size),
                    size=(self.cfg.patch_size, self.cfg.patch_size),
                    slices=(0, num_z),
                )

            # Handle slideio behavior: single slice returns 3D, multiple returns 4D
            if z_stack.ndim == 3:
                z_stack = np.expand_dims(z_stack, axis=0)

            # Compute Brenner focus metric for each slice in the stack
            patch_scores = []
            for z in range(num_z):
                # compute_brenner_gradient internally handles BGR -> Gray conversion
                score = compute_brenner_gradient(z_stack[z])
                patch_scores.append(score)

            best_zs[i] = np.argmax(patch_scores)
            del z_stack

        # Construct final patch array: [raw_x, raw_y, best_z]
        c_arr = np.array(candidates)
        final_patches = np.column_stack([c_arr[:, 0], c_arr[:, 1], best_zs]).astype(
            np.int32
        )

        return SlideMetadata(
            name=vsi_path.name,
            width=width_raw,
            height=height_raw,
            num_z=num_z,
            patches=final_patches,
        )


def load_master_index(dataset_name: str, patch_size: int) -> MasterIndex | None:
    manifest_path = config.get_master_index_path(dataset_name, patch_size)
    indices_dir = config.get_slide_index_dir(dataset_name, patch_size)

    if not manifest_path.exists():
        return None

    try:
        with open(manifest_path, "rb") as f:
            manifest_data = pickle.load(f)

        if isinstance(manifest_data, MasterIndex):
            return manifest_data

        slide_metadatas = []
        for pkl_path in sorted(indices_dir.glob("*.pkl")):
            try:
                with open(pkl_path, "rb") as f:
                    slide_metadatas.append(pickle.load(f))
            except Exception:
                continue

        return MasterIndex(
            file_registry=slide_metadatas,
            patch_size=patch_size,
            config_state=manifest_data["config_state"],
        )
    except Exception as e:
        print(f"Error loading indices: {e}")
        return None


def save_slide_json(result: SlideMetadata, pkl_path: Path):
    json_path = pkl_path.with_suffix(".json")
    data = {
        "name": result.name,
        "width": result.width,
        "height": result.height,
        "num_z": result.num_z,
        "patch_count": result.patch_count,
        "patches": result.patches.tolist(),
    }
    with open(json_path, "w") as f:
        json.dump(data, f)


def save_master_index_json(
    master_index: MasterIndex, dataset_name: str, patch_size: int
) -> None:
    json_path = config.get_master_index_path(dataset_name, patch_size).with_suffix(
        ".json"
    )
    data = {
        "dataset_name": dataset_name,
        "patch_size": patch_size,
        "config": asdict(master_index.config_state),
        "slides": [],
    }

    for slide in master_index.file_registry:
        data["slides"].append(
            {
                "name": slide.name,
                "width": slide.width,
                "height": slide.height,
                "num_z": slide.num_z,
                "patch_count": slide.patch_count,
                "patches": slide.patches.tolist(),
            }
        )

    with open(json_path, "w") as f:
        json.dump(data, f)
    print(f"  [EXPORT] Consolidated JSON index saved to {json_path}")


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
    raw_dir = config.get_vsi_raw_dir(dataset_name)
    manifest_path = config.get_master_index_path(dataset_name, patch_size)
    indices_dir = config.get_slide_index_dir(dataset_name, patch_size)

    workers = workers or os.cpu_count() or 1

    current_config = PreprocessConfig(
        patch_size=patch_size,
        stride=stride,
        downsample_factor=downsample_factor,
        min_tissue_coverage=min_tissue_coverage,
        dataset_name=dataset_name,
    )

    existing_results = []
    processed_names = set()

    if manifest_path.exists() and not force:
        with open(manifest_path, "rb") as f:
            manifest_data = pickle.load(f)

        if manifest_data["config_state"] != current_config:
            print(f"Config mismatch! Use --force to reprocess.")
            return

        for pkl_path in sorted(indices_dir.glob("*.pkl")):
            try:
                with open(pkl_path, "rb") as f:
                    res = pickle.load(f)
                existing_results.append(res)
                processed_names.add(res.name)
            except Exception:
                continue

    all_files = sorted(list(raw_dir.glob("*.vsi")))
    if limit:
        all_files = all_files[:limit]

    files_to_process = [f for f in all_files if f.name not in processed_names]

    print(f"Preprocessing {dataset_name} (Stride={stride}, Patch={patch_size})")
    print(f"Total: {len(all_files)} | Remaining: {len(files_to_process)}")

    if not manifest_path.exists() or force:
        with open(manifest_path, "wb") as f:
            pickle.dump({"config_state": current_config}, f)

    if files_to_process:
        with multiprocessing.Pool(workers) as pool:
            process_func = partial(process_slide_wrapper, config=current_config)

            for result in pool.imap_unordered(process_func, files_to_process):
                if result is not None:
                    slide_pkl_path = indices_dir / f"{result.name}.pkl"
                    with open(slide_pkl_path, "wb") as f:
                        pickle.dump(result, f)
                    save_slide_json(result, slide_pkl_path)
                    existing_results.append(result)
                    print(f"  [SAVE] {result.name} saved")

    if existing_results:
        master_index = MasterIndex(
            file_registry=sorted(existing_results, key=lambda x: x.name),
            patch_size=patch_size,
            config_state=current_config,
        )
        with open(manifest_path, "wb") as f:
            pickle.dump(master_index, f)
        save_master_index_json(master_index, dataset_name, patch_size)


def process_slide_wrapper(vsi_path: Path, config: PreprocessConfig):
    return SlidePreprocessor(config).process(vsi_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline VSI Preprocessor.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=448)
    parser.add_argument("--downsample_factor", type=int, default=2)
    parser.add_argument("--min_tissue_coverage", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    preprocess_dataset(
        args.dataset,
        args.patch_size,
        args.stride,
        args.downsample_factor,
        args.min_tissue_coverage,
        limit=args.limit,
        workers=args.workers,
        force=args.force,
    )
