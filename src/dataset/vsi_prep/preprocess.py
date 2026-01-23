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
import json
from dataclasses import asdict
from src.utils.focus_metrics import compute_brenner_gradient

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


def load_master_index(dataset_name: str, patch_size: int) -> MasterIndex | None:
    """Helper to load all individual slide indices and the manifest configuration."""
    manifest_path = config.get_master_index_path(dataset_name, patch_size)
    indices_dir = config.get_slide_index_dir(dataset_name, patch_size)

    if not manifest_path.exists():
        return None

    try:
        with open(manifest_path, "rb") as f:
            manifest_data = pickle.load(f)

        # Load all slide pickles from the indices directory
        slide_metadatas = []
        for pkl_path in sorted(indices_dir.glob("*.pkl")):
            with open(pkl_path, "rb") as f:
                slide_metadatas.append(pickle.load(f))

        return MasterIndex(
            file_registry=slide_metadatas,
            patch_size=patch_size,
            config_state=manifest_data["config_state"],
        )
    except Exception as e:
        print(f"Error loading indices: {e}")
        return None


def save_slide_json(result: SlideMetadata, pkl_path: Path):
    """Save a compact JSON sidecar for a slide index."""
    json_path = pkl_path.with_suffix(".json")
    data = {
        "name": result.name,
        "width": result.width,
        "height": result.height,
        "num_z": result.num_z,
        "patch_count": result.patch_count,
        "patches": [[p.x, p.y, p.z] for p in result.patches],
    }
    with open(json_path, "w") as f:
        json.dump(data, f)


def save_master_index_json(
    master_index: MasterIndex, dataset_name: str, patch_size: int
) -> None:
    """Export the MasterIndex to a JSON file for shareability."""
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
                "patches": [[p.x, p.y, p.z] for p in slide.patches],
            }
        )

    with open(json_path, "w") as f:
        json.dump(data, f)
    print(f"  [EXPORT] Consolidated JSON index saved to {json_path}")


def preprocess_dataset(
    dataset_name: str,
    patch_size: int,
    stride: int,
    min_tissue_coverage: float,
    limit: int | None = None,
    workers: int | None = None,
    force: bool = False,
) -> None:
    """Preprocess all slides in a dataset into a MasterIndex."""
    raw_dir = config.get_vsi_raw_dir(dataset_name)
    manifest_path = config.get_master_index_path(dataset_name, patch_size)
    indices_dir = config.get_slide_index_dir(dataset_name, patch_size)
    workers = workers or os.cpu_count()

    current_config = PreprocessConfig(
        patch_size=patch_size,
        stride=stride,
        min_tissue_coverage=min_tissue_coverage,
        dataset_name=dataset_name,
    )

    # 1. Check for existing indices
    existing_results = []
    processed_names = set()

    # Load manifest config
    if manifest_path.exists() and not force:
        with open(manifest_path, "rb") as f:
            manifest_data = pickle.load(f)

        if manifest_data["config_state"] != current_config:
            print(f"Config mismatch! Existing config: {manifest_data['config_state']}")
            print("Delete the index directory or use --force to reprocess.")
            return

        # Load all individual slide pickles
        for pkl_path in sorted(indices_dir.glob("*.pkl")):
            with open(pkl_path, "rb") as f:
                res = pickle.load(f)
                existing_results.append(res)
                processed_names.add(res.name)

        print(f"Found {len(existing_results)} already processed slides.")

    # 2. Identify remaining work
    all_files = sorted(list(raw_dir.glob("*.vsi")))
    if limit:
        all_files = all_files[:limit]

    files_to_process = [f for f in all_files if f.name not in processed_names]

    print(f"Preprocessing {dataset_name} (Stride={stride}, Patch={patch_size})")
    print(
        f"Total: {len(all_files)} | Skipping: {len(processed_names)} | Remaining: {len(files_to_process)}"
    )

    # Save initial manifest to mark the start/config
    with open(manifest_path, "wb") as f:
        pickle.dump({"config_state": current_config}, f)

    if files_to_process:
        # Use a Pool to process slides
        with multiprocessing.Pool(workers) as pool:
            process_func = partial(process_slide_wrapper, config=current_config)

            # Use imap to save results as they come in (atomic saving)
            for result in pool.imap_unordered(process_func, files_to_process):
                if result is not None:
                    # Save individual slide index (Pickle for perf, JSON for Git)
                    slide_pkl_path = indices_dir / f"{result.name}.pkl"
                    with open(slide_pkl_path, "wb") as f:
                        pickle.dump(result, f)

                    save_slide_json(result, slide_pkl_path)

                    existing_results.append(result)
                    print(f"  [SAVE] {result.name} saved (.pkl + .json)")

    # 3. Final Consolidation & Master JSON Export
    if existing_results:
        master_index = MasterIndex(
            file_registry=sorted(existing_results, key=lambda x: x.name),
            patch_size=patch_size,
            config_state=current_config,
        )
        print(f"Completed! Total Patches: {master_index.total_samples // 27:,}")
        save_master_index_json(master_index, dataset_name, patch_size)
    else:
        print("No slides were processed or found.")


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
