import os
import json
from pathlib import Path
from typing import Any

import pickle
import numpy as np
import cv2
import slideio
import multiprocessing
from functools import partial
from skimage.filters import threshold_otsu

import config
from src.utils.io_utils import suppress_stderr


def compute_brenner_gradient(image: np.ndarray) -> int:
    """Compute Brenner Gradient focus measure (sum of squared differences)."""
    # Image is BGR; convert to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Cast to int32 to prevent overflow
    gray = gray.astype(np.int32)
    shifted = np.roll(gray, -2, axis=1)
    return int(np.sum((gray - shifted) ** 2))


def select_best_focus_slice(
    scene: Any, width: int, height: int, num_z: int, down_w: int, down_h: int
) -> np.ndarray:
    """Iterate all Z-slices to find the global best focus slice (grayscale)."""
    best_focus_score = -1
    best_img = None

    for z in range(num_z):
        img = scene.read_block(
            rect=(0, 0, width, height), size=(down_w, down_h), slices=(z, z + 1)
        )
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        score = compute_brenner_gradient(img)

        if score > best_focus_score:
            best_focus_score = score
            best_img = gray

    if best_img is None:
        raise ValueError("No focus slice found (image set might be empty).")

    return best_img


def generate_tissue_mask(image_gray: np.ndarray) -> np.ndarray:
    """Generate binary tissue mask via Otsu's method."""
    thresh = threshold_otsu(image_gray)
    # Tissue is darker than background in Brightfield
    return ((image_gray <= thresh) * 255).astype(np.uint8)


def find_valid_patches(
    mask: np.ndarray,
    width: int,
    height: int,
    patch_size: int,
    stride: int,
    downscale_factor: int,
    down_w: int,
    down_h: int,
) -> list[tuple[int, int]]:
    """Scan mask/slide area to find valid patches based on tissue coverage."""
    valid_patches = []
    min_tissue_coverage = config.MIN_TISSUE_COVERAGE

    # Iterate in Full Resolution Coordinates
    for y in range(0, height - patch_size + 1, stride):
        ds_y = int(y / downscale_factor)
        ds_y_end = int((y + patch_size) / downscale_factor)
        ds_y_end = min(ds_y_end, down_h)
        if ds_y >= down_h:
            continue

        for x in range(0, width - patch_size + 1, stride):
            ds_x = int(x / downscale_factor)
            ds_x_end = int((x + patch_size) / downscale_factor)
            ds_x_end = min(ds_x_end, down_w)
            if ds_x >= down_w:
                continue

            mask_patch = mask[ds_y:ds_y_end, ds_x:ds_x_end]
            if mask_patch.size == 0:
                continue

            tissue_ratio = np.count_nonzero(mask_patch) / mask_patch.size

            if tissue_ratio >= min_tissue_coverage:
                valid_patches.append((x, y))

    return valid_patches


def resolve_patch_z_levels(
    scene: Any,
    valid_patches: list[tuple[int, int]],
    width: int,
    height: int,
    num_z: int,
    down_w: int,
    down_h: int,
    patch_size: int,
    downscale_factor: int,
) -> list[tuple[int, int, int]]:
    """Determine the best Z-level for each identified patch."""
    num_patches = len(valid_patches)
    if num_patches == 0:
        return []

    # Shape: [num_patches, num_z]
    patch_scores = np.zeros((num_patches, num_z), dtype=np.float32)

    for z in range(num_z):
        # Read full downsampled frame for this Z
        img_z = scene.read_block(
            rect=(0, 0, width, height), size=(down_w, down_h), slices=(z, z + 1)
        )

        for i, (vx, vy) in enumerate(valid_patches):
            # Map valid patch coords to downsampled coords
            dx = int(vx / downscale_factor)
            dy = int(vy / downscale_factor)
            dx_end = int((vx + patch_size) / downscale_factor)
            dy_end = int((vy + patch_size) / downscale_factor)

            dx_end = min(dx_end, down_w)
            dy_end = min(dy_end, down_h)

            if dx >= down_w or dy >= down_h:
                continue

            patch_roi = img_z[dy:dy_end, dx:dx_end]
            if patch_roi.size > 0:
                patch_scores[i, z] = compute_brenner_gradient(patch_roi)

    # Select best Z
    best_z_indices = np.argmax(patch_scores, axis=1)

    final_patches = []
    for i in range(num_patches):
        vx, vy = valid_patches[i]
        final_patches.append((vx, vy, int(best_z_indices[i])))

    return final_patches


def save_mask_visualization(
    path: Path,
    mask: np.ndarray,
    suffix: str = "_mask.png",
) -> None:
    """Save a binary mask visualization."""
    config.VIS_DIR.mkdir(parents=True, exist_ok=True)
    basename = path.stem
    out_path = config.VIS_DIR / f"{basename}{suffix}"

    cv2.imwrite(str(out_path), mask)
    print(f"Saved visualization to {out_path}")


def save_patch_visualization(
    path: Path,
    reference_shape: tuple[int, int],
    patches: list[tuple[int, int]],
    patch_size: int,
    downscale_factor: int,
    suffix: str = "_patch_mask.png",
) -> None:
    """Save a visualization of selected patches on a black canvas."""
    config.VIS_DIR.mkdir(parents=True, exist_ok=True)
    basename = path.stem
    out_path = config.VIS_DIR / f"{basename}{suffix}"

    h, w = reference_shape
    vis_img = np.zeros((h, w), dtype=np.uint8)

    for vx, vy in patches:
        dx = int(vx / downscale_factor)
        dy = int(vy / downscale_factor)
        dx_end = int((vx + patch_size) / downscale_factor)
        dy_end = int((vy + patch_size) / downscale_factor)
        cv2.rectangle(vis_img, (dx, dy), (dx_end, dy_end), 255, -1)

    cv2.imwrite(str(out_path), vis_img)
    print(f"Saved visualization to {out_path}")


def process_slide(
    vsi_path: Path,
    patch_size: int = config.PATCH_SIZE,
    stride: int = config.STRIDE,
    downscale_factor: int = config.DOWNSCALE_FACTOR,
    otsu_threshold_rel: float | None = None,
) -> dict[str, Any]:
    with suppress_stderr():
        slide = slideio.open_slide(str(vsi_path), "VSI")
    scene = slide.get_scene(0)
    width, height = scene.size
    num_z = scene.num_z_slices

    # Dimensions for downsampled operations
    down_w = width // downscale_factor
    down_h = height // downscale_factor

    # 1. Select Best Focus Slice
    # This now raises ValueError if it fails, which will be caught by the worker wrapper or crash intentionally
    best_gray_img = select_best_focus_slice(scene, width, height, num_z, down_w, down_h)

    # 2. Generate Mask
    mask = generate_tissue_mask(best_gray_img)
    save_mask_visualization(vsi_path, mask, suffix="_mask.png")

    # 3. Find Valid Patches
    valid_patches_spatial = find_valid_patches(
        mask, width, height, patch_size, stride, downscale_factor, down_w, down_h
    )

    # Visualise Patches
    save_patch_visualization(
        vsi_path,
        reference_shape=mask.shape[:2],
        patches=valid_patches_spatial,
        patch_size=patch_size,
        downscale_factor=downscale_factor,
        suffix="_patch_mask.png",
    )

    # 4. Resolve Z-levels
    final_patches = resolve_patch_z_levels(
        scene,
        valid_patches_spatial,
        width,
        height,
        num_z,
        down_w,
        down_h,
        patch_size,
        downscale_factor,
    )  # Phase 2

    print(f"Processed {vsi_path.name}: {len(final_patches)} valid patches")
    return {
        "path": vsi_path,
        "width": width,
        "height": height,
        "num_z": num_z,
        "patches": final_patches,
    }


def resolve_file_list(split_files: list[str]) -> list[Path]:
    """
    Validates existence of files listed in the split.
    """
    valid_files = []
    missing_files = []

    for filename in split_files:
        path = config.VSI_RAW_DIR / filename
        if path.exists():
            valid_files.append(path)
        else:
            missing_files.append(path)

    if missing_files:
        print(f"Warning: {len(missing_files)} files from split not found on disk.")
        for m in missing_files[:5]:
            print(f"  Missing: {m}")

    return sorted(valid_files)


def preprocess_dataset(
    patch_size: int = config.PATCH_SIZE,
    stride: int = config.STRIDE,
    workers: int | None = os.cpu_count(),
    force: bool = False,
) -> None:
    # Ensure cache directory exists
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not config.SPLIT_FILE.exists():
        print(
            f"Error: Split file {config.SPLIT_FILE} not found. Run create_split.py first."
        )
        return

    print(f"Loading splits from {config.SPLIT_FILE}...")
    with open(config.SPLIT_FILE, "r") as f:
        splits = json.load(f)

    # Process each split independently
    for split_name, split_files in splits.items():
        if split_name in ["seed", "total"]:
            continue  # Skip metadata keys

        output_path = config.CACHE_DIR / f"{config.INDEX_PREFIX}_{split_name}.pkl"

        if output_path.exists() and not force:
            print(f"Index for {split_name} already exists at {output_path}. Skipping.")
            continue

        print(f"\n=== Generating Index for Split: {split_name.upper()} ===")

        # 1. Resolve actual files for this split
        files = resolve_file_list(split_files)

        if not files:
            print(f"No valid files found for split {split_name}. Skipping.")
            continue

        output_path = config.CACHE_DIR / f"{config.INDEX_PREFIX}_{split_name}.pkl"

        print(f"Processing {len(files)} files for {split_name}...")

        # 2. Process Files (Reuse process_slide logic but feed specific file list)
        process_func = partial(process_slide, patch_size=patch_size, stride=stride)

        with multiprocessing.Pool(workers) as pool:
            results = pool.map(process_func, files)

        # 3. Aggregate
        results = [r for r in results if r is not None]

        file_registry = []
        cumulative_count = 0
        cumulative_indices = []
        total_valid_patches = 0
        total_z_slices = 0

        for res in results:
            path = res["path"]
            patches = res["patches"]
            num_z = res["num_z"]
            samples_in_file = len(patches) * num_z

            cumulative_count += samples_in_file
            cumulative_indices.append(cumulative_count)
            file_registry.append({"path": path, "patches": patches, "num_z": num_z})

            total_valid_patches += len(patches)
            total_z_slices += samples_in_file

        final_index = {
            "file_registry": file_registry,
            "cumulative_indices": np.array(cumulative_indices, dtype=np.int64),
            "total_samples": cumulative_count,
            "patch_size": patch_size,
        }

        with open(output_path, "wb") as f:
            pickle.dump(final_index, f)

        print(f"Saved {split_name} index to {output_path}")
        print(f"  - Slides: {len(file_registry)}")
        print(f"  - Patches: {total_valid_patches}")
        print(f"  - Total Samples: {cumulative_count}")


def main():
    preprocess_dataset()


if __name__ == "__main__":
    main()
