import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

from src.datasets.agnor_ome.config import AgNorOMEConfig
import pandas as pd


class OMEDataset(Dataset):
    """
    Worker-safe Dataset for OME-TIFF patches.
    Uses tifffile for reading multi-series Z-stacks.
    """

    def __init__(
        self,
        index_df: pd.DataFrame,
        transform: Any | None = None,
        dataset_name: str = "AgNor_OME",
    ):
        self.df = index_df
        self.transform = transform
        self.dataset_name = dataset_name
        self.patch_size = 224
        self.downsample_factor = 1

        self._readers: dict[str, Any] | None = None
        self._owner_pid: int | None = None

    def __len__(self) -> int:
        return len(self.df)

    def _get_reader(self, img_path_str: str):
        current_pid = os.getpid()
        if self._readers is None or self._owner_pid != current_pid:
            self._readers = {}
            self._owner_pid = current_pid

        if img_path_str not in self._readers:
            actual_path = Path(img_path_str)
            if not actual_path.exists() and self.dataset_name:
                raw_dir = AgNorOMEConfig().raw_dir
                actual_path = raw_dir / actual_path.name

            try:
                # Store the TiffFile object
                reader = tifffile.TiffFile(actual_path)
                self._readers[img_path_str] = reader
            except Exception as e:
                raise RuntimeError(
                    f"Error opening OME-TIFF in process {current_pid}: {actual_path}. Error: {e}"
                )

        return self._readers[img_path_str]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        try:
            row = self.df.iloc[idx]
            img_name = row["slide_name"]
            x = row["x"]
            y = row["y"]
            z_level = row["z_level"]
            best_z = row["optimal_z"]
            num_z = row["num_z"]

            reader = self._get_reader(img_name)

            # Calculate focus offset dynamically from OME metadata
            try:
                ome_metadata = reader.ome_metadata
                import xml.etree.ElementTree as ET

                root = ET.fromstring(ome_metadata)
                ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
                pixels = root.find(".//ome:Pixels", ns)
                if pixels is not None and "PhysicalSizeZ" in pixels.attrib:
                    z_res_microns = float(pixels.attrib["PhysicalSizeZ"])
                else:
                    z_res_microns = 1.0  # Fallback
            except Exception:
                z_res_microns = 1.0

            z_offset = float(best_z - z_level) * z_res_microns

            # Read the patch from the specific series (z_level)
            # In multi-series OME-TIFF, series corresponds to Z
            series = reader.series[z_level]

            # Crop raw patch, then resize to ViT output resolution
            raw_extent = self.patch_size * self.downsample_factor

            full_plane = series.asarray()
            block = full_plane[
                int(y) : int(y + raw_extent), int(x) : int(x + raw_extent)
            ]

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
                "num_z": int(num_z),
            }

            return {
                "image": image,
                "target": torch.tensor(z_offset, dtype=torch.float32),
                "metadata": metadata,
            }

        except Exception as e:
            # We print error but re-raise to be safe
            # x, y, z_level might not be bound if exception happens early
            print(f"Error reading index {idx}: {e}")
            raise
