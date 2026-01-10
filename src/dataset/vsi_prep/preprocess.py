import os
import argparse
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
from src.dataset.vsi_types import SlideMetadata, MasterIndex, PreprocessConfig, Patch


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
    min_tissue_coverage: float = 0.05,
) -> list[tuple[int, int]]:
    """Find patches with sufficient tissue coverage."""
    valid_patches = []

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
        Patch(x=valid_patches[i][0], y=valid_patches[i][1], z=int(best_z_indices[i]))
        for i in range(num_patches)
    ]


def process_slide(
    vsi_path: Path,
    patch_size: int,
    stride: int,
    downscale_factor: int,
    min_tissue_coverage: float = 0.05,
) -> SlideMetadata:
    with suppress_stderr():
        slide = slideio.open_slide(str(vsi_path), "VSI")
    scene = slide.get_scene(0)
    width, height = scene.size
    num_z = scene.num_z_slices

    down_w, down_h = width // downscale_factor, height // downscale_factor

    best_gray_img = select_best_focus_slice(scene, width, height, num_z, down_w, down_h)
    mask = generate_tissue_mask(best_gray_img)

    valid_patches_spatial = find_valid_patches(
        mask,
        width,
        height,
        patch_size,
        stride,
        downscale_factor,
        down_w,
        down_h,
        min_tissue_coverage=min_tissue_coverage,
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
    return SlideMetadata(
        name=vsi_path.name,
        path=vsi_path,
        width=width,
        height=height,
        num_z=num_z,
        patches=final_patches,
    )


def resolve_file_list(filenames: list[str], dataset_name: str) -> list[Path]:
    valid_files = []
    missing_files = []

    raw_dir = config.get_vsi_raw_dir(dataset_name)
    for filename in filenames:
        path = raw_dir / filename
        if path.exists():
            valid_files.append(path)
        else:
            missing_files.append(path)

    if missing_files:
        print(
            f"Warning: {len(missing_files)} files not found on disk for {dataset_name}."
        )
    return sorted(valid_files)


def get_config_state(
    dataset_name: str,
    patch_size: int,
    stride: int,
    downscale_factor: int,
    min_tissue_coverage: float,
) -> dict[str, Any]:
    import hashlib

    split_file = config.get_split_path(dataset_name)

    split_hash = "none"
    if split_file.exists():
        with open(split_file, "rb") as f:
            split_hash = hashlib.sha256(f.read()).hexdigest()

    return {
        "PATCH_SIZE": patch_size,
        "STRIDE": stride,
        "DOWNSCALE_FACTOR": downscale_factor,
        "MIN_TISSUE_COVERAGE": min_tissue_coverage,
        "SPLIT_HASH": split_hash,
        "DATASET_NAME": dataset_name,
    }


def preprocess_dataset(
    dataset_name: str,
    patch_size: int = 224,
    stride: int = 224,
    downscale_factor: int = 8,
    min_tissue_coverage: float = 0.05,
    workers: int | None = None,
    force: bool = False,
) -> None:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if workers is None:
        workers = os.cpu_count()

    from src.utils.exact_utils import get_exact_image_list

    print(f"Fetching image list for {dataset_name} from EXACT...")
    images_meta = get_exact_image_list(dataset_name=dataset_name)
    filenames = sorted([img["name"] for img in images_meta])

    current_config = PreprocessConfig(
        patch_size=patch_size,
        stride=stride,
        downscale_factor=downscale_factor,
        min_tissue_coverage=min_tissue_coverage,
        dataset_name=dataset_name,
    )

    output_path = config.get_master_index_path(dataset_name=dataset_name)

    should_generate = force
    if not output_path.exists():
        should_generate = True
    elif not force:
        try:
            with open(output_path, "rb") as f:
                existing_data = pickle.load(f)
            # Handle both MasterIndex objects and old dicts for transition
            if isinstance(existing_data, MasterIndex):
                existing_config = existing_data.config_state
            else:
                cfg_dict = existing_data.get("config_state", {})
                existing_config = PreprocessConfig(**cfg_dict)

            if existing_config == current_config:
                print(f"Master index for {dataset_name} exists and matches. Skipping.")
                should_generate = False
            else:
                print(f"Master index for {dataset_name} mismatch. Regenerating.")
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

    process_func = partial(
        process_slide,
        patch_size=patch_size,
        stride=stride,
        downscale_factor=downscale_factor,
        min_tissue_coverage=min_tissue_coverage,
    )
    with multiprocessing.Pool(workers) as pool:
        results = pool.map(process_func, files)

    results = [r for r in results if r is not None]

    # In the master index, we store the full registry.
    # We don't bother with cumulative_indices here yet, as these depend on the split.
    # Actually, let's keep the structure similar but just containing everything.

    final_index = MasterIndex(
        file_registry=results,
        patch_size=patch_size,
        config_state=current_config,
    )

    with open(output_path, "wb") as f:
        pickle.dump(final_index, f)

    print(
        f"Saved master index for {dataset_name} to {output_path} ({len(results)} slides, {final_index.total_samples} total patches)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ZStack_HE")
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=224)
    parser.add_argument("--downscale_factor", type=int, default=8)
    parser.add_argument("--min_tissue_coverage", type=float, default=0.05)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    preprocess_dataset(
        dataset_name=args.dataset,
        patch_size=args.patch_size,
        stride=args.stride,
        downscale_factor=args.downscale_factor,
        min_tissue_coverage=args.min_tissue_coverage,
        force=args.force,
    )
