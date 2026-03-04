import os
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
import slideio
import torch
from torch.utils.data import Dataset

from focus_offset.utils.io_utils import suppress_stderr


class SyntheticVSIMetadata(TypedDict):
    filename: str
    x: int
    y: int
    optimal_z: int
    offset_z: int


class SyntheticVSIDataset(Dataset):
    def __init__(
        self,
        index_df: pd.DataFrame,
        slide_dir: Path,
        patch_size_n: int = 1024,
        patch_size_k: int = 256,
        z_offset_steps: int = 5,
        downsample: int = 2,
        transform: Any | None = None,
    ):
        """
        Synthetic Dataset for Focus Offset parsing.

        Args:
            index_df: DataFrame containing the prep_synthetic index
                      (needs 'slide_name', 'x', 'y', 'optimal_z', 'num_z')
            slide_dir: Base directory containing .vsi files
            patch_size_n: Size of the N*N patch at the optimal focal plane
            patch_size_k: Size of the K*K patch at the offset plane (k < N)
            z_offset_steps: Distance in focal planes from optimal Z for the K*K patch
            downsample: Extraction downsample level
            transform: Transformations applied to both patches (must handle dict format or independent)
        """
        if patch_size_k > patch_size_n:
            raise ValueError("K must be less than or equal to N")

        self.df = index_df
        self.slide_dir = Path(slide_dir)
        self.patch_size_n = patch_size_n
        self.patch_size_k = patch_size_k
        self.z_offset_steps = z_offset_steps
        self.downsample = downsample
        self.transform = transform

        self._slides: dict[str, Any] | None = None
        self._owner_pid: int | None = None

    def __len__(self) -> int:
        return len(self.df)

    def _get_scene(self, vsi_filename: str):
        current_pid = os.getpid()
        if self._slides is None or self._owner_pid != current_pid:
            self._slides = {}
            self._owner_pid = current_pid

        if vsi_filename not in self._slides:
            actual_path = self.slide_dir / vsi_filename
            if not actual_path.exists():
                raise FileNotFoundError(f"Slide not found at {actual_path}")
            with suppress_stderr():
                slide = slideio.open_slide(str(actual_path), "VSI")
                scene = slide.get_scene(0)
            self._slides[vsi_filename] = (slide, scene)
        return self._slides[vsi_filename][1]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        vsi_name = row["slide_name"]

        x = int(row["x"])
        y = int(row["y"])
        optimal_z = int(row["optimal_z"])

        scene = self._get_scene(vsi_name)
        num_z = scene.num_z_slices

        # Determine the target z-slice for the K*K patch directly relative to optimal_z
        offset_z = optimal_z + self.z_offset_steps
        if offset_z < 0 or offset_z >= num_z:
            raise ValueError(
                f"Requested Z offset_z={offset_z} is out of bounds for slide "
                f"{vsi_name} with {num_z} slices (optimal_z={optimal_z})."
            )

        # 1. Read the N * N Patch at optimal Z
        extent_n = self.patch_size_n * self.downsample

        block_n = scene.read_block(
            rect=(x, y, extent_n, extent_n),
            size=(self.patch_size_n, self.patch_size_n),
            slices=(optimal_z, optimal_z + 1),
        )

        # 2. Read the K * K Patch at offset Z
        extent_k = self.patch_size_k * self.downsample

        # Calculate exactly the center coordinates of K inside N at the slide level
        center_x = x + (extent_n // 2)
        center_y = y + (extent_n // 2)

        k_x = center_x - (extent_k // 2)
        k_y = center_y - (extent_k // 2)

        block_k = scene.read_block(
            rect=(k_x, k_y, extent_k, extent_k),
            size=(self.patch_size_k, self.patch_size_k),
            slices=(offset_z, offset_z + 1),
        )

        if self.transform:
            pass

        # Convert HWC numpy -> CHW tensor (normalizing to [0, 1])
        tensor_n = torch.from_numpy(block_n).permute(2, 0, 1).float() / 255.0
        tensor_k = torch.from_numpy(block_k).permute(2, 0, 1).float() / 255.0

        # Read z resolution dynamically from scene
        z_res = getattr(scene, "z_resolution", 0)
        if not z_res:
            raise ValueError(f"Slide {vsi_name} has no Z-resolution.")
        z_res_microns = z_res * 1e6

        z_offset_actual = offset_z - optimal_z
        z_offset_microns = z_offset_actual * z_res_microns

        metadata: SyntheticVSIMetadata = {
            "filename": vsi_name,
            "x": x,
            "y": y,
            "optimal_z": optimal_z,
            "offset_z": offset_z,
            "z_offset_microns": z_offset_microns,
            "z_res_microns": z_res_microns,
        }

        return {
            "optimal_patch": tensor_n,
            "offset_patch": tensor_k,
            "z_offset_actual": z_offset_actual,
            "z_offset_microns": z_offset_microns,
            "metadata": metadata,
        }
