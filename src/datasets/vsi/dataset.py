import os
from pathlib import Path
from typing import Any

import pandas as pd
import slideio
import torch
from torch.utils.data import Dataset

from src.utils.io_utils import suppress_stderr


class VSIDataset(Dataset):
    def __init__(
        self,
        index_df: pd.DataFrame,
        slide_dir: Path,
        patch_size: int = 224,
        downsample: int = 2,
        transform: Any | None = None,
    ):
        self.df = index_df
        self.slide_dir = Path(slide_dir)
        self.patch_size = patch_size
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
        x, y, z_level, z_offset = (
            row["x"],
            row["y"],
            row["z_level"],
            float(row["z_offset_microns"]),
        )

        scene = self._get_scene(vsi_name)
        extent = self.patch_size * self.downsample

        block = scene.read_block(
            rect=(int(x), int(y), extent, extent),
            size=(self.patch_size, self.patch_size),
            slices=(int(z_level), int(z_level) + 1),
        )

        if self.transform:
            image = self.transform(image=block)["image"]
        else:
            image = torch.from_numpy(block).permute(2, 0, 1).float() / 255.0

        return {
            "image": image,
            "target": torch.tensor(z_offset, dtype=torch.float32),
            "metadata": {
                "filename": vsi_name,
                "x": int(x),
                "y": int(y),
                "z_level": int(z_level),
                "optimal_z": int(row["optimal_z"]),
            },
        }


def get_vsi_dataset(
    index_df: pd.DataFrame,
    slide_dir: Path,
    downsample: int,
    patch_size: int = 224,
    transform: Any = None,
) -> VSIDataset:
    return VSIDataset(
        index_df=index_df,
        slide_dir=slide_dir,
        patch_size=patch_size,
        downsample=downsample,
        transform=transform,
    )
