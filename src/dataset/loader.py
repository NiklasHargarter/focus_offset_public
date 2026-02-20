import json
import random
from typing import Literal

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from src.config import DatasetConfig, TrainConfig
from src.dataset.vsi_dataset import VSIDataset
from src.dataset.vsi_prep.preprocess import load_master_index
from src.dataset.vsi_types import MasterIndex, ProcessedIndex


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


def _load_data_indices(
    dataset_cfg: DatasetConfig,
) -> tuple[MasterIndex, dict]:
    master_index = load_master_index(
        dataset_cfg.name,
        dataset_cfg.prep.stride,
        dataset_cfg.prep.downsample_factor,
        dataset_cfg.prep.min_tissue_coverage,
    )
    split_path = dataset_cfg.split_path

    if master_index is None or not split_path.exists():
        raise RuntimeError(
            f"Missing data for {dataset_cfg.name}. Run prepare_data first."
        )

    with open(split_path, "r") as f:
        splits = json.load(f)

    return master_index, splits


def filter_index(
    master_index: MasterIndex, files: list[str], dataset_cfg: DatasetConfig
) -> ProcessedIndex:
    """Filter master index for a specific set of filenames and compute cumulative indices."""
    file_registry = master_index.file_registry
    name_to_entry = {entry.name: entry for entry in file_registry}

    filtered_registry = []
    counts = []

    for name in files:
        if name in name_to_entry:
            res = name_to_entry[name]
            filtered_registry.append(res)
            counts.append(res.total_samples)
        else:
            print(
                f"Warning: File {name} not found in master index for {dataset_cfg.name}"
            )

    cumulative_indices = np.cumsum(counts) if counts else np.array([], dtype=np.int64)

    return ProcessedIndex(
        file_registry=filtered_registry,
        cumulative_indices=cumulative_indices,
        patch_size=dataset_cfg.prep.patch_size,
        downsample_factor=master_index.config_state.downsample_factor,
        dataset_name=dataset_cfg.name,
    )


def get_holdout_dataloaders(
    dataset_cfg: DatasetConfig,
    train_cfg: TrainConfig,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    master_index, splits = _load_data_indices(dataset_cfg)
    train_pool = splits["train_pool"]

    random.seed(seed)
    random.shuffle(train_pool)

    num_val = int(len(train_pool) * val_ratio)
    val_files = train_pool[:num_val]
    train_files = train_pool[num_val:]

    train_index = filter_index(master_index, train_files, dataset_cfg)
    val_index = filter_index(master_index, val_files, dataset_cfg)

    train_dataset = VSIDataset(
        index_data=train_index, transform=get_transforms("train")
    )
    val_dataset = VSIDataset(index_data=val_index, transform=get_transforms("val"))

    print(
        f"Data Setup [HoldOut]: {len(train_dataset)} train, {len(val_dataset)} val samples."
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


def get_agnor_dataloader(
    dataset_cfg: DatasetConfig,
    train_cfg: TrainConfig,
):
    master_index, splits = _load_data_indices(dataset_cfg)
    test_files = splits.get("test", [])
    if not test_files:
        test_files = [s.name for s in master_index.file_registry]

    from src.dataset.ome_dataset import OMEDataset

    test_index = filter_index(master_index, test_files, dataset_cfg)
    test_dataset = OMEDataset(index_data=test_index, transform=get_transforms("val"))

    print(f"Data Setup [AgNor]: {len(test_dataset)} test samples.")

    return DataLoader(
        test_dataset,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        shuffle=False,
    )
