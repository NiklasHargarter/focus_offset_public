import bisect
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

from src import config
from src.dataset.vsi_types import ProcessedIndex, SlideMetadata


class OMEDataset(Dataset):
    """
    Worker-safe Dataset for OME-TIFF patches.
    Uses tifffile for reading multi-series Z-stacks.
    """

    def __init__(
        self,
        index_data: ProcessedIndex,
        transform: Callable[[Any], torch.Tensor] | None = None,
        z_res_microns: float = 1.0,  # Default fallback if metadata is missing
    ):
        self.index = index_data
        self.transform = transform
        self.z_res_microns = z_res_microns

        self.file_registry = self.index.file_registry
        self.cumulative_indices = self.index.cumulative_indices
        self.total_samples = self.index.total_samples
        self.patch_size = self.index.patch_size
        self.downsample_factor = self.index.downsample_factor
        self.dataset_name = self.index.dataset_name

        self._readers = None
        self._owner_pid = None

    def _get_reader(self, img_path_str: str):
        current_pid = os.getpid()
        if self._readers is None or self._owner_pid != current_pid:
            self._readers = {}
            self._owner_pid = current_pid

        if img_path_str not in self._readers:
            actual_path = Path(img_path_str)
            if not actual_path.exists() and self.dataset_name:
                raw_dir = config.get_vsi_raw_dir(self.dataset_name)
                alt_path = raw_dir / actual_path.name
                if alt_path.exists():
                    actual_path = alt_path

            try:
                # Store the TiffFile object
                reader = tifffile.TiffFile(actual_path)
                self._readers[img_path_str] = reader
            except Exception as e:
                raise RuntimeError(
                    f"Error opening OME-TIFF in process {current_pid}: {actual_path}. Error: {e}"
                )

        return self._readers[img_path_str]

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        file_idx = bisect.bisect_right(self.cumulative_indices, idx)
        if file_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_indices[file_idx - 1]

        file_meta: SlideMetadata = self.file_registry[file_idx]
        img_name = file_meta.name
        num_z = file_meta.num_z

        patch_idx = local_idx // num_z
        z_level = local_idx % num_z

        x, y, best_z = file_meta.patches[patch_idx]

        reader = self._get_reader(img_name)

        # Calculate focus offset
        # TODO: Attempt to extract z_res_microns from OME-XML if possible
        z_offset = float(best_z - z_level) * self.z_res_microns

        # Read the patch from the specific series (z_level)
        # In multi-series OME-TIFF, series corresponds to Z
        series = reader.series[z_level]

        # Crop raw patch, then resize to ViT output resolution
        raw_extent = self.patch_size * self.downsample_factor
        try:
            full_plane = series.asarray()
            block = full_plane[int(y) : int(y + raw_extent), int(x) : int(x + raw_extent)]

            # Grayscale to RGB if needed
            if block.ndim == 2:
                block = np.stack([block] * 3, axis=-1)
            elif block.shape[-1] == 1:
                block = np.concatenate([block] * 3, axis=-1)

            # Resize to patch_size (identity when ds=1)
            if block.shape[0] != self.patch_size or block.shape[1] != self.patch_size:
                block = cv2.resize(block, (self.patch_size, self.patch_size))

            if self.transform:
                image = self.transform(image=block)["image"]
            else:
                image = torch.from_numpy(block).permute(2, 0, 1).float() / 255.0

            metadata = {
                "filename": img_name,
                "x": int(x),
                "y": int(y),
                "z_level": int(z_level),
                "optimal_z": int(best_z),
            }

            return {
                "image": image,
                "target": torch.tensor(z_offset, dtype=torch.float32),
                "metadata": metadata,
            }

        except Exception as e:
            print(f"Error reading {img_name} at ({x}, {y}) z={z_level}: {e}")
            raise
