import multiprocessing
import os
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import tifffile

from src.datasets.agnor_ome.config import AgNorOMEConfig
from src.utils.grid import filter_by_tissue_coverage, generate_grid
from src.utils.tissue_mask import detect_tissue_mask
from src.utils.focus_metrics import compute_brenner_gradient

import pandas as pd


class PreprocessConfig:
    def __init__(
        self, patch_size, stride, downsample_factor, min_tissue_coverage, dataset_name
    ):
        self.patch_size = patch_size
        self.stride = stride
        self.downsample_factor = downsample_factor
        self.min_tissue_coverage = min_tissue_coverage
        self.dataset_name = dataset_name


class OMEImageReader:
    """Uses tifffile to read multi-series OME-TIFFs as Z-stacks."""

    def __init__(self, path: Path):
        self.path = path
        self._tif = tifffile.TiffFile(path)
        self.num_z = len(self._tif.series)
        first_series = self._tif.series[0]
        self.height, self.width = first_series.shape[:2]
        self.size = (self.width, self.height)

    def read_block(
        self, rect: tuple[int, int, int, int], slices: tuple[int, int] | None = None
    ) -> np.ndarray:
        x, y, w, h = rect
        z_start, z_end = slices if slices else (0, self.num_z)

        stack = []
        for z in range(z_start, z_end):
            data = self._tif.series[z].asarray()
            if data.ndim == 2:
                data = np.expand_dims(data, axis=-1)
            stack.append(data[y : y + h, x : x + w])

        return np.stack(stack)

    def close(self):
        self._tif.close()


# ---------------------------------------------------------------------------
# Single-image processing
# ---------------------------------------------------------------------------


def read_thumbnail_rgb(
    reader: OMEImageReader, width: int, height: int, mask_downscale: int
) -> np.ndarray:
    """Read a small RGB thumbnail from the first z-slice."""
    d_w = width // mask_downscale
    d_h = height // mask_downscale
    full_img = reader.read_block((0, 0, width, height), (0, 1))[0]
    small = cv2.resize(full_img, (d_w, d_h))
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)


def find_best_z_per_patch(reader, candidates, raw_patch_size, patch_size, num_z):
    """For each candidate patch, read the full z-stack and pick the sharpest slice."""
    best_zs = np.zeros(len(candidates), dtype=np.int32)

    for i, (x, y) in enumerate(candidates):
        z_stack = reader.read_block(
            rect=(x, y, raw_patch_size, raw_patch_size), slices=(0, num_z)
        )
        scores = [
            compute_brenner_gradient(cv2.resize(z_stack[z], (patch_size, patch_size)))
            for z in range(num_z)
        ]
        best_zs[i] = int(np.argmax(scores))

    return best_zs


def build_patch_index(candidates, best_zs) -> np.ndarray:
    """Combine (x, y) candidates with best-z into an (N, 3) int32 array."""
    coords = np.array(candidates, dtype=np.int32)
    return np.column_stack([coords, best_zs])


def process_image(
    img_path: Path, cfg: PreprocessConfig, mask_downscale: int
) -> list[dict]:
    """Full single-image pipeline: thumbnail → mask → grid → focus → index."""
    raw_extent = cfg.patch_size * cfg.downsample_factor

    reader = OMEImageReader(img_path)
    width, height = reader.size
    num_z = reader.num_z

    print(f"[{img_path.name}] tissue mask")
    thumbnail = read_thumbnail_rgb(reader, width, height, mask_downscale)
    mask = detect_tissue_mask(thumbnail_rgb=thumbnail)

    print(f"[{img_path.name}] grid + tissue filter")
    grid = generate_grid(width, height, raw_extent, cfg.stride)
    candidates = filter_by_tissue_coverage(
        grid,
        mask,
        raw_extent,
        cfg.min_tissue_coverage,
        mask_downscale=mask_downscale,
    )

    if not candidates:
        reader.close()
        return []

    print(f"[{img_path.name}] focus search ({len(candidates)} patches)")
    best_zs = find_best_z_per_patch(
        reader, candidates, raw_extent, cfg.patch_size, num_z
    )
    reader.close()

    rows = []
    for (x, y), best_z in zip(candidates, best_zs):
        for z in range(num_z):
            rows.append(
                {
                    "slide_name": img_path.name,
                    "x": int(x),
                    "y": int(y),
                    "z_level": int(z),
                    "optimal_z": int(best_z),
                    "num_z": int(num_z),
                }
            )
    return rows


def _process_image_worker(img_path: Path, cfg: PreprocessConfig, mask_downscale: int):
    try:
        return process_image(img_path, cfg, mask_downscale)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


def preprocess_dataset(
    dataset_name: str = "AgNor_OME",
    workers: int | None = None,
    dry_run: bool = False,
) -> None:
    dataset_cfg = AgNorOMEConfig()

    raw_dir = dataset_cfg.raw_dir
    index_path = dataset_cfg.get_index_path()

    workers = workers or os.cpu_count() or 1

    current_config = PreprocessConfig(
        patch_size=dataset_cfg.prep.patch_size,
        stride=dataset_cfg.prep.stride,
        downsample_factor=dataset_cfg.prep.downsample_factor,
        min_tissue_coverage=dataset_cfg.prep.min_tissue_coverage,
        dataset_name=dataset_name,
    )

    all_files = sorted(
        list(raw_dir.glob("*.ome.tiff")) + list(raw_dir.glob("*.ome.tif"))
    )
    if dry_run:
        all_files = all_files[:10]

    print(f"Preprocessing OME-TIFF {dataset_name} (Total: {len(all_files)})")

    if all_files:
        with multiprocessing.Pool(workers) as pool:
            process_func = partial(
                _process_image_worker,
                cfg=current_config,
                mask_downscale=dataset_cfg.prep.mask_downscale,
            )
            all_rows = []
            for result_rows in pool.imap_unordered(process_func, all_files):
                if result_rows:
                    all_rows.extend(result_rows)

            if all_rows:
                df = pd.DataFrame(all_rows)
                df.to_parquet(index_path)
                print(f"  [EXPORT] Consolidated parquet index saved to {index_path}")
