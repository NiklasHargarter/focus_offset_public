import os
import argparse
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
    """Compute Brenner Gradient focus score."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.int32)
    shifted = np.roll(gray, -2, axis=1)
    return int(np.sum((gray - shifted) ** 2))


def select_best_focus_slice(
    scene: Any, width: int, height: int, num_z: int, down_w: int, down_h: int
) -> np.ndarray:
    """Find best focus slice across Z-levels."""
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
        raise ValueError("No focus slice found.")

    return best_img


def generate_tissue_mask(image_gray: np.ndarray) -> np.ndarray:
    """Generate binary tissue mask."""
    thresh = threshold_otsu(image_gray)
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
    """Find patches with sufficient tissue coverage."""
    valid_patches = []
    min_tissue_coverage = config.MIN_TISSUE_COVERAGE

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
    """Determine optimal Z-level for each patch."""
    num_patches = len(valid_patches)
    if num_patches == 0:
        return []

    patch_scores = np.zeros((num_patches, num_z), dtype=np.float32)

    for z in range(num_z):
        img_z = scene.read_block(
            rect=(0, 0, width, height), size=(down_w, down_h), slices=(z, z + 1)
        )

        for i, (vx, vy) in enumerate(valid_patches):
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

    best_z_indices = np.argmax(patch_scores, axis=1)
    return [
        (valid_patches[i][0], valid_patches[i][1], int(best_z_indices[i]))
        for i in range(num_patches)
    ]


def save_mask_visualization(
    path: Path, mask: np.ndarray, suffix: str = "_mask.png"
) -> None:
    config.VIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.VIS_DIR / f"{path.stem}{suffix}"
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
    config.VIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.VIS_DIR / f"{path.stem}{suffix}"

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
) -> dict[str, Any]:
    with suppress_stderr():
        slide = slideio.open_slide(str(vsi_path), "VSI")
    scene = slide.get_scene(0)
    width, height = scene.size
    num_z = scene.num_z_slices

    down_w, down_h = width // downscale_factor, height // downscale_factor

    best_gray_img = select_best_focus_slice(scene, width, height, num_z, down_w, down_h)
    mask = generate_tissue_mask(best_gray_img)

    if config.GENERATE_VISUALIZATIONS:
        save_mask_visualization(vsi_path, mask, suffix="_mask.png")

    valid_patches_spatial = find_valid_patches(
        mask, width, height, patch_size, stride, downscale_factor, down_w, down_h
    )

    if config.GENERATE_VISUALIZATIONS:
        save_patch_visualization(
            vsi_path,
            mask.shape[:2],
            valid_patches_spatial,
            patch_size,
            downscale_factor,
        )

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
    )

    print(f"Processed {vsi_path.name}: {len(final_patches)} valid patches")
    return {
        "path": vsi_path,
        "width": width,
        "height": height,
        "num_z": num_z,
        "patches": final_patches,
    }


def resolve_file_list(
    split_files: list[str], dataset_name: str = config.DATASET_NAME
) -> list[Path]:
    valid_files = []
    missing_files = []

    raw_dir = config.get_vsi_raw_dir(dataset_name)
    for filename in split_files:
        path = raw_dir / filename
        if path.exists():
            valid_files.append(path)
        else:
            missing_files.append(path)

    if missing_files:
        print(
            f"Warning: {len(missing_files)} files from split not found on disk for {dataset_name}."
        )
    return sorted(valid_files)


def get_config_state(dataset_name: str = config.DATASET_NAME) -> dict[str, Any]:
    import hashlib

    split_file = config.get_split_path(dataset_name)

    split_hash = "none"
    if split_file.exists():
        with open(split_file, "rb") as f:
            split_hash = hashlib.sha256(f.read()).hexdigest()

    return {
        "PATCH_SIZE": config.PATCH_SIZE,
        "STRIDE": config.STRIDE,
        "DOWNSCALE_FACTOR": config.DOWNSCALE_FACTOR,
        "MIN_TISSUE_COVERAGE": config.MIN_TISSUE_COVERAGE,
        "SPLIT_HASH": split_hash,
        "DATASET_NAME": dataset_name,
    }


def preprocess_dataset(
    dataset_name: str = config.DATASET_NAME,
    patch_size: int = config.PATCH_SIZE,
    stride: int = config.STRIDE,
    workers: int | None = None,
    force: bool = False,
    mode: str = None,
) -> None:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if workers is None:
        workers = os.cpu_count()

    split_file = config.get_split_path(dataset_name)

    if not split_file.exists():
        print(f"Error: Split file {split_file} not found.")
        return

    print(f"Loading splits from {split_file} for {dataset_name}...")
    with open(split_file, "r") as f:
        splits = json.load(f)

    current_config = get_config_state(dataset_name=dataset_name)

    for split_name, split_files in splits.items():
        if split_name in ["seed", "total"]:
            continue
        if mode and split_name != mode:
            continue

        output_path = config.get_index_path(split_name, dataset_name=dataset_name)

        should_generate = force
        if not output_path.exists():
            should_generate = True
        elif not force:
            try:
                with open(output_path, "rb") as f:
                    existing_data = pickle.load(f)
                if existing_data.get("config_state", {}) == current_config:
                    print(
                        f"Index for {dataset_name}/{split_name} exists and matches. Skipping."
                    )
                    should_generate = False
                else:
                    print(
                        f"Index for {dataset_name}/{split_name} mismatch. Regenerating."
                    )
                    should_generate = True
            except Exception:
                should_generate = True

        if not should_generate:
            continue

        print(
            f"\n=== Generating Index for {dataset_name} Split: {split_name.upper()} ==="
        )
        files = resolve_file_list(split_files, dataset_name=dataset_name)
        if not files:
            continue

        process_func = partial(process_slide, patch_size=patch_size, stride=stride)
        with multiprocessing.Pool(workers) as pool:
            results = pool.map(process_func, files)

        results = [r for r in results if r is not None]
        file_registry, cumulative_indices, cumulative_count = [], [], 0

        for res in results:
            samples_in_file = len(res["patches"]) * res["num_z"]
            cumulative_count += samples_in_file
            cumulative_indices.append(cumulative_count)
            file_registry.append(
                {"path": res["path"], "patches": res["patches"], "num_z": res["num_z"]}
            )

        final_index = {
            "file_registry": file_registry,
            "cumulative_indices": np.array(cumulative_indices, dtype=np.int64),
            "total_samples": cumulative_count,
            "patch_size": patch_size,
            "config_state": current_config,
        }

        with open(output_path, "wb") as f:
            pickle.dump(final_index, f)

        print(
            f"Saved {dataset_name}/{split_name} index to {output_path} ({cumulative_count} samples)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=config.DATASET_NAME)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    preprocess_dataset(dataset_name=args.dataset, force=args.force)
