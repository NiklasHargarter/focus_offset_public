import os
from pathlib import Path
from typing import Any

import pandas as pd
import slideio
import torch
from torch.utils.data import Dataset

from src.datasets.zstack_he.config import ZStackHEConfig, PrepConfig
from src.utils.io_utils import suppress_stderr


class VSIDataset(Dataset):
    """
    Dataset for VSI patches.
    Uses worker-safe lazy initialization with PID tracking to ensure slide handles
    are never shared across processes.
    """

    def __init__(
        self,
        index_df: pd.DataFrame,
        transform: Any | None = None,
        dataset_name: str = "ZStack_HE",
        prep_cfg: PrepConfig | None = None,
    ):
        self.df = index_df
        self.transform = transform
        self.dataset_name = dataset_name
        prep_cfg = prep_cfg or PrepConfig()
        self.patch_size = prep_cfg.patch_size
        self.downsample_factor = prep_cfg.downsample_factor

        self._slides: dict[str, Any] | None = None
        self._owner_pid: int | None = None

    def __len__(self) -> int:
        return len(self.df)

    def _get_scene(self, vsi_path: str):
        """
        Lazy initialization of slide handles, unique to each worker process.
        """
        current_pid = os.getpid()

        if self._slides is None or self._owner_pid != current_pid:
            self._slides = {}
            self._owner_pid = current_pid

        if vsi_path not in self._slides:
            actual_path = Path(vsi_path)
            if not actual_path.exists() and self.dataset_name:
                raw_dir = ZStackHEConfig().raw_dir
                actual_path = raw_dir / actual_path.name

            try:
                with suppress_stderr():
                    slide = slideio.open_slide(str(actual_path), "VSI")
                    scene = slide.get_scene(0)
                self._slides[vsi_path] = (slide, scene)
            except Exception as e:
                raise RuntimeError(
                    f"Error opening slide in process {current_pid}: {actual_path}. Error: {e}"
                )

        return self._slides[vsi_path][1]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Maps a global linear index to a specific (Slide, Patch, Z-slice) triplet.
        """
        try:
            row = self.df.iloc[idx]
            vsi_name = row["slide_name"]
            x = row["x"]
            y = row["y"]
            z_level = row["z_level"]
            best_z = row["optimal_z"]
            num_z = row["num_z"]
            z_offset = float(row["z_offset_microns"])

            scene = self._get_scene(vsi_name)

            raw_extent = self.patch_size * self.downsample_factor
            rect = (
                int(x),
                int(y),
                raw_extent,
                raw_extent,
            )

            # Efficient single-slice read from VSI, with downscaling
            block = scene.read_block(
                rect=rect,
                size=(self.patch_size, self.patch_size),
                slices=(int(z_level), int(z_level) + 1),
            )

            if self.transform:
                # Albumentations standard call
                image = self.transform(image=block)["image"]
            else:
                image = torch.from_numpy(block).permute(2, 0, 1).float() / 255.0

            metadata = {
                "filename": vsi_name,
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
            # We don't have vsi_name/rect/z_level if _get_grid_info fails
            # But usually it fails inside the try block after resolving
            print(f"Error reading index {idx}: {e}")
            raise
