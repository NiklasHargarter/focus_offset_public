import random
from pathlib import Path
from typing import Literal, Tuple

import albumentations as A
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from src.config import TrainConfig
from src.datasets.agnor_ome.dataset import get_ome_dataset


def get_ome_transforms(mode: Literal["train", "val"]):
    if mode == "train":
        return A.Compose(
            [
                A.D4(p=1.0),
                A.Compose(
                    [
                        A.ColorJitter(
                            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2, p=0.8
                        ),
                        A.ToGray(p=0.2),
                        A.ChannelShuffle(p=0.1),
                    ],
                    p=1.0,
                ),
                A.Normalize(mean=0.5, std=0.5),
                ToTensorV2(),
            ]
        )
    return A.Compose(
        [
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2(),
        ]
    )


def get_ome_dataloaders(
    train_parquet: Path,
    slide_dir: Path,
    downsample: int,
    train_cfg: TrainConfig,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    train_pool_df = pd.read_parquet(train_parquet)

    slides = sorted(list(train_pool_df["slide_name"].unique()))
    random.seed(seed)
    random.shuffle(slides)

    num_val = max(1, int(len(slides) * val_ratio)) if len(slides) > 1 else 0
    val_files, train_files = slides[:num_val], slides[num_val:]

    train_index = train_pool_df[
        train_pool_df["slide_name"].isin(train_files)
    ].reset_index(drop=True)
    val_index = train_pool_df[train_pool_df["slide_name"].isin(val_files)].reset_index(
        drop=True
    )

    train_ds = get_ome_dataset(
        index_df=train_index,
        slide_dir=slide_dir,
        downsample=downsample,
        transform=get_ome_transforms("train"),
    )
    val_ds = get_ome_dataset(
        index_df=val_index,
        slide_dir=slide_dir,
        downsample=downsample,
        transform=get_ome_transforms("val"),
    )

    kwargs = {
        "batch_size": train_cfg.batch_size,
        "num_workers": train_cfg.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": True,
        "persistent_workers": train_cfg.num_workers > 0,
        "prefetch_factor": 2 if train_cfg.num_workers > 0 else None,
    }
    return DataLoader(train_ds, shuffle=True, **kwargs), DataLoader(
        val_ds, shuffle=False, **kwargs
    )


def get_ome_test_loader(
    test_parquet: Path, slide_dir: Path, downsample: int, train_cfg: TrainConfig
) -> DataLoader:
    test_df = pd.read_parquet(test_parquet)
    test_ds = get_ome_dataset(
        index_df=test_df,
        slide_dir=slide_dir,
        downsample=downsample,
        transform=get_ome_transforms("val"),
    )
    return DataLoader(
        test_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=train_cfg.num_workers > 0,
        prefetch_factor=2 if train_cfg.num_workers > 0 else None,
    )
