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

from src.utils.io_utils import suppress_stderr

from src import config
from src.dataset.vsi_types import SlideMetadata, Patch, PreprocessConfig, MasterIndex
from src.utils.focus_metrics import compute_brenner_gradient
from src.utils.exact_utils import get_exact_image_list

MASK_DOWNSCALE = 16


def detect_tissue(scene: Any) -> Tuple[int, np.ndarray]:
    """Find the sharpest slice and generate a tissue mask for sparse filtering."""
    width, height = scene.size
    num_z = scene.num_z_slices
    d_w, d_h = width // MASK_DOWNSCALE, height // MASK_DOWNSCALE

    best_score, best_img, best_z = -1.0, None, 0
    # Sample every 3rd slice for speed - mask just needs a reasonably sharp image
    for z in range(0, num_z, 3):
        img = scene.read_block(
            rect=(0, 0, width, height), size=(d_w, d_h), slices=(z, z + 1)
        )
        score = compute_brenner_gradient(img)
        if score > best_score:
            best_score, best_img, best_z = score, img, z

    if best_img is None:
        gray = np.zeros((d_h, d_w), dtype=np.uint8)
    else:
        gray = cv2.cvtColor(best_img, cv2.COLOR_BGR2GRAY)

    thresh = threshold_otsu(gray)
    mask = ((gray <= thresh) * 255).astype(np.uint8)
    return best_z, mask


def generate_patch_candidates(
    mask: np.ndarray,
    width: int,
    height: int,
    patch_size: int,
    stride: int,
    min_cov: float,
) -> List[Tuple[int, int]]:
    """Return top-left coordinates for grid patches based on masking."""
    m_h, m_w = mask.shape
    candidates = []

    for y in range(0, height - patch_size + 1, stride):
        my, mye = int(y / MASK_DOWNSCALE), int((y + patch_size) / MASK_DOWNSCALE)
        for x in range(0, width - patch_size + 1, stride):
            mx, mxe = int(x / MASK_DOWNSCALE), int((x + patch_size) / MASK_DOWNSCALE)

            mask_patch = mask[my : min(mye, m_h), mx : min(mxe, m_w)]
            if (
                mask_patch.size > 0
                and (np.count_nonzero(mask_patch) / mask_patch.size) >= min_cov
            ):
                # Returns top-left (ox, oy)
                candidates.append((x, y))
    return candidates


class SlidePreprocessor:
    def __init__(self, config: PreprocessConfig):
        self.cfg = config
        # Optimal Master Block Size (Number of patches per side)
        # 8x8 is the standard high-throughput strategy for 1x magnification
        self.block_n = 8
        self.patch_px = self.cfg.patch_size
        self.block_px = self.patch_px * self.block_n

    def process_block(
        self,
        scene: Any,
        brx: int,
        bry: int,
        brw: int,
        brh: int,
        patches: List[Tuple[int, int]],
    ) -> List[Patch]:
        """Process a large block of Z-slices to find best Z for all patches inside."""
        num_z = scene.num_z_slices
        results = []

        # Read the entire Z-stack for the larger block (Master Block strategy)
        # to avoid redundant I/O requests for overlapping or adjacent patches.
        with suppress_stderr():
            # stack: (Z, H, W, C)
            stack = scene.read_block(
                rect=(brx, bry, brw, brh), size=(brw, brh), slices=(0, num_z)
            )

        # For each patch in this block, find the sharpest Z-level
        for ox, oy in patches:
            # Map global slide coordinates to local block coordinates
            lx = ox - brx
            ly = oy - bry

            # Extract patch stack from the pre-read block stack (zero-copy view)
            patch_stack = stack[
                :, ly : ly + self.cfg.patch_size, lx : lx + self.cfg.patch_size
            ]

            best_s, best_z = -1.0, 0
            for z in range(num_z):
                # Calculate focus score for each Z-slice
                s = compute_brenner_gradient(patch_stack[z])
                if s > best_s:
                    best_s, best_z = s, z
            results.append(Patch(x=ox, y=oy, z=best_z))

        return results

    def process(self, vsi_path: Path) -> SlideMetadata:
        with suppress_stderr():
            slide = slideio.open_slide(str(vsi_path), "VSI")
        scene = slide.get_scene(0)
        width, height = scene.size

        print(f"[{vsi_path.name}] Stage 1: Tissue Masking...")
        _, mask = detect_tissue(scene)

        print(
            f"[{vsi_path.name}] Stage 2: Grid Generation (Stride={self.cfg.stride})..."
        )
        candidates = generate_patch_candidates(
            mask,
            width,
            height,
            self.cfg.patch_size,
            self.cfg.stride,
            self.cfg.min_tissue_coverage,
        )

        # Group candidates into blocks
        blocks = {}
        for ox, oy in candidates:
            bx, by = ox // self.block_px, oy // self.block_px
            if (bx, by) not in blocks:
                blocks[(bx, by)] = []
            blocks[(bx, by)].append((ox, oy))

        total_blocks = len(blocks)
        print(
            f"[{vsi_path.name}] Stage 3: Master Block Focus Search ({total_blocks} blocks)..."
        )
        processed_blocks = 0
        final_patches = []
        for (bx, by), patches in sorted(blocks.items()):
            brx, bry = bx * self.block_px, by * self.block_px
            max_ox = max(p[0] for p in patches)
            max_oy = max(p[1] for p in patches)

            # Distance from block start to the end of the last patch in that block
            # We clip to width/height to ensure valid slideio requests
            brw = min((max_ox - brx) + self.patch_px, width - brx)
            brh = min((max_oy - bry) + self.patch_px, height - bry)

            block_results = self.process_block(scene, brx, bry, brw, brh, patches)
            final_patches.extend(block_results)

            processed_blocks += 1
            if processed_blocks % 10 == 0 or processed_blocks == total_blocks:
                print(
                    f"  [{vsi_path.name}] Stage 3: {processed_blocks}/{total_blocks} blocks..."
                )

        return SlideMetadata(
            vsi_path.name, vsi_path, width, height, scene.num_z_slices, final_patches
        )


def preprocess_dataset(
    dataset_name: str,
    patch_size: int,
    stride: int,
    min_tissue_coverage: float,
    limit: int | None = None,
    workers: int | None = None,
    force: bool = False,
) -> None:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    workers = workers or os.cpu_count()

    current_config = PreprocessConfig(
        patch_size=patch_size,
        stride=stride,
        min_tissue_coverage=min_tissue_coverage,
        dataset_name=dataset_name,
    )

    output_path = config.get_master_index_path(dataset_name, patch_size)

    existing_results = []
    if output_path.exists() and not force:
        try:
            with open(output_path, "rb") as f:
                existing: MasterIndex = pickle.load(f)

            # Check if config matches
            if existing.config_state == current_config:
                existing_results = existing.file_registry
                print(
                    f"Found existing index for {dataset_name} with {len(existing_results)} slides."
                )
            else:
                print(
                    f"Configuration changed for {dataset_name}. Full re-preprocess required."
                )
        except Exception as e:
            print(f"Failed to load existing index: {e}. Starting fresh.")

    images_meta = get_exact_image_list(dataset_name=dataset_name, force=force)
    raw_dir = config.get_vsi_raw_dir(dataset_name)
    all_files = sorted(
        [
            raw_dir / img["name"]
            for img in images_meta
            if (raw_dir / img["name"]).exists()
        ]
    )

    # Filter out files already in existing_results
    processed_names = {r.name for r in existing_results}
    files_to_process = [f for f in all_files if f.name not in processed_names]

    if limit is not None:
        files_to_process = files_to_process[:limit]

    if not files_to_process:
        if existing_results:
            print(f"All {len(all_files)} files already processed. Nothing to do.")
            return
        else:
            print("No files found to process. Aborting.")
            return

    print(f"\n=== Preprocessing: {dataset_name} ===")
    print(f"  Configuration: Stride={stride}px, Patch={patch_size}px")
    print("  Strategy: Optimal Master Blocks (1x Resolution)")
    print(
        f"  Status: {len(existing_results)} already processed, {len(files_to_process)} new files to process."
    )

    new_results = []
    if files_to_process:
        with multiprocessing.Pool(workers) as pool:
            process_func = partial(process_slide_wrapper, config=current_config)
            new_results = pool.map(process_func, files_to_process)

    new_results = [r for r in new_results if r is not None]

    # Combine old and new
    combined_results = existing_results + new_results

    # Sort by name for consistency
    combined_results.sort(key=lambda x: x.name)

    final_index = MasterIndex(
        file_registry=combined_results,
        patch_size=patch_size,
        config_state=current_config,
    )

    with open(output_path, "wb") as f:
        pickle.dump(final_index, f)
    print(
        f"Final Count: {len(combined_results)} slides, {final_index.total_samples} patches"
    )


def process_slide_wrapper(vsi_path: Path, config: PreprocessConfig):
    return SlidePreprocessor(config).process(vsi_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-Throughput VSI Preprocessor.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=224)
    parser.add_argument("--min_tissue_coverage", type=float, default=0.05)
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of slides to process"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    preprocess_dataset(
        args.dataset,
        args.patch_size,
        args.stride,
        args.min_tissue_coverage,
        limit=args.limit,
        force=args.force,
    )
