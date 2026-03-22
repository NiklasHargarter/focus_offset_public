import os
from pathlib import Path
from typing import Any, TypedDict
import pandas as pd
import slideio
import torch
from torch.utils.data import Dataset, DataLoader

from focus_offset.utils.io_utils import suppress_stderr
from .config import SyntheticConfig


class SyntheticVSIMetadata(TypedDict):
    slide_name: str
    patch_x: int
    patch_y: int
    focal_plane_index: int
    relative_offset_steps: int
    z_step_size_microns: float


class SyntheticVSIDataset(Dataset):
    def __init__(
        self,
        index_df: pd.DataFrame,
        slide_dir: Path,
        config: SyntheticConfig,
        transform: Any | None = None,
    ):
        """
        Synthetic Dataset for Focus Offset parsing.
        """
        if config.top_percent_laplacian < 1.0:
            num_keep = int(len(index_df) * config.top_percent_laplacian)
            df = (
                index_df.sort_values("max_focus_score", ascending=False)
                .head(num_keep)
                .reset_index(drop=True)
            )
        else:
            df = index_df

        self.df = df
        self.slide_dir = Path(slide_dir)
        self.patch_size_input = config.patch_size_input
        self.patch_size_target = config.patch_size_target
        self.z_offset_steps = config.z_offset_steps
        self.downsample = config.downsample
        self.transform = transform
        self.read_target = not config.simulation_mode

        self._slides: dict[str, Any] | None = None
        self._owner_pid: int | None = None

    def __len__(self) -> int:
        return len(self.df)

    def _get_scene(self, vsi_filename: str) -> tuple[Any, float]:
        current_pid = os.getpid()
        if self._slides is None or self._owner_pid != current_pid:
            self._slides = {}
            self._owner_pid = current_pid

        if vsi_filename not in self._slides:
            actual_path = self.slide_dir / vsi_filename
            with suppress_stderr():
                slide = slideio.open_slide(str(actual_path), "VSI")
                scene = slide.get_scene(0)

            z_res_microns = scene.z_resolution * 1e6

            self._slides[vsi_filename] = (slide, scene, z_res_microns)

        return self._slides[vsi_filename][1], self._slides[vsi_filename][2]

    def _read_input_patch(
        self, scene, x: int, y: int, focal_plane_index: int
    ) -> torch.Tensor:
        block_input = scene.read_block(
            rect=(
                x,
                y,
                self.patch_size_input * self.downsample,
                self.patch_size_input * self.downsample,
            ),
            size=(self.patch_size_input, self.patch_size_input),
            slices=(focal_plane_index, focal_plane_index + 1),
        )
        return torch.from_numpy(block_input).permute(2, 0, 1).float() / 255.0

    def _read_target_patch(
        self, scene, x: int, y: int, focal_plane_index: int
    ) -> torch.Tensor | None:
        if not self.read_target:
            return None

        input_z_index = focal_plane_index + self.z_offset_steps
        center_x = x + (self.patch_size_input * self.downsample) // 2
        center_y = y + (self.patch_size_input * self.downsample) // 2
        target_x = center_x - (self.patch_size_target * self.downsample) // 2
        target_y = center_y - (self.patch_size_target * self.downsample) // 2

        block_target = scene.read_block(
            rect=(
                target_x,
                target_y,
                self.patch_size_target * self.downsample,
                self.patch_size_target * self.downsample,
            ),
            size=(self.patch_size_target, self.patch_size_target),
            slices=(input_z_index, input_z_index + 1),
        )
        return torch.from_numpy(block_target).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        vsi_name = row["slide_name"]

        x = int(row["x"])
        y = int(row["y"])
        focal_plane_index = int(row["optimal_z"])

        scene, z_step_size_microns = self._get_scene(vsi_name)

        tensor_input = self._read_input_patch(scene, x, y, focal_plane_index)

        tensor_target = self._read_target_patch(scene, x, y, focal_plane_index)

        if self.transform:
            tensor_input = self.transform(tensor_input)
            if tensor_target is not None:
                tensor_target = self.transform(tensor_target)

        metadata: SyntheticVSIMetadata = {
            "slide_name": vsi_name,
            "patch_x": x,
            "patch_y": y,
            "focal_plane_index": focal_plane_index,
            "relative_offset_steps": self.z_offset_steps,
            "z_step_size_microns": z_step_size_microns,
        }

        out = {"input": tensor_input, "metadata": metadata}
        if tensor_target is not None:
            out["target"] = tensor_target
        return out


def get_synthetic_dataloaders(config: SyntheticConfig, num_workers: int):
    """Create DataLoaders for the synthetic dataset defined by config."""
    slide_dir = Path(config.slide_dir)
    train_df = pd.read_parquet(config.index_dir / "train_synthetic.parquet")
    test_df = pd.read_parquet(config.index_dir / "test_synthetic.parquet")

    train_dataset = SyntheticVSIDataset(
        index_df=train_df, slide_dir=slide_dir, config=config
    )
    val_dataset = SyntheticVSIDataset(
        index_df=test_df, slide_dir=slide_dir, config=config
    )

    kwargs = {
        "batch_size": config.batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "prefetch_factor": 2 if num_workers > 0 else None,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **kwargs)
    return train_loader, val_loader
