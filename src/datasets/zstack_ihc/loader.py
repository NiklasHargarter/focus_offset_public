import random
from typing import Literal

import albumentations as A
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from src.config import TrainConfig
from .config import ZStackIHCConfig
from .dataset import VSIDataset


def get_transforms(mode: Literal["train", "val"]):
    """
    Get albumentations transforms for training or validation.
    """
    if mode == "train":
        return A.Compose(
            [
                A.D4(p=1.0),
                A.Compose(
                    [
                        A.ColorJitter(
                            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2, p=0.8
                        ),
                        A.ToGray(p=0.2),  # Force structure learning
                        A.ChannelShuffle(p=0.1),  # Break channel priors
                    ],
                    p=1.0,
                ),
                A.ToFloat(max_value=255.0),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.ToFloat(max_value=255.0),
                ToTensorV2(),
            ]
        )


def _load_train_df(
    dataset_cfg: ZStackIHCConfig,
) -> pd.DataFrame:
    train_path = dataset_cfg.get_train_index_path()

    if not train_path.exists():
        raise RuntimeError(
            f"Missing training data for {dataset_cfg.name}. Run prepare_data first."
        )

    df = pd.read_parquet(train_path)
    return df


def get_dataloaders(
    train_cfg: TrainConfig,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    dataset_cfg = ZStackIHCConfig()
    train_pool_df = _load_train_df(dataset_cfg)

    train_pool = sorted(list(train_pool_df["slide_name"].unique()))

    random.seed(seed)
    random.shuffle(train_pool)

    # Use at least 1 slide for validation if possible
    num_val = max(1, int(len(train_pool) * val_ratio)) if len(train_pool) > 1 else 0
    val_files = train_pool[:num_val]
    train_files = train_pool[num_val:]

    train_index = train_pool_df[
        train_pool_df["slide_name"].isin(train_files)
    ].reset_index(drop=True)
    val_index = train_pool_df[train_pool_df["slide_name"].isin(val_files)].reset_index(
        drop=True
    )

    train_dataset = VSIDataset(index_df=train_index, transform=get_transforms("train"))
    val_dataset = VSIDataset(index_df=val_index, transform=get_transforms("val"))

    print(
        f"Data Setup [ZStack IHC]: {len(train_dataset)} train, {len(val_dataset)} val samples."
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=train_cfg.num_workers,
        persistent_workers=train_cfg.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        persistent_workers=train_cfg.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def get_test_loader(
    train_cfg: TrainConfig,
):
    dataset_cfg = ZStackIHCConfig()
    test_path = dataset_cfg.get_test_index_path()

    if not test_path.exists():
        raise RuntimeError(
            f"Missing test data for {dataset_cfg.name}. Run prepare_data first."
        )

    test_df = pd.read_parquet(test_path)
    test_dataset = VSIDataset(index_df=test_df, transform=get_transforms("val"))

    print(f"Data Setup [{dataset_cfg.name} TEST]: {len(test_dataset)} samples.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        persistent_workers=train_cfg.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )

    return test_loader
