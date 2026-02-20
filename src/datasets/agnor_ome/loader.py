import json
from typing import Literal

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from src.config import TrainConfig
from .config import AgNorOMEConfig
from .dataset import OMEDataset
import pandas as pd


def get_transforms(mode: Literal["train", "val"]):
    return A.Compose(
        [
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ]
    )


def _load_ag_nor_test_df(dataset_cfg: AgNorOMEConfig) -> pd.DataFrame:
    # AgNor is primarily for evaluation, meaning we only care about the testing split.
    # If the user ran create_split, they have a test.parquet. Wait, preprocess wasn't updated to build test.parquet yet! 
    # Let's just load index.parquet and filter by JSON if needed, or if json isn't present, use the whole thing.
    index_path = dataset_cfg.get_index_path()
    split_path = dataset_cfg.split_path

    if not index_path.exists():
        raise RuntimeError(f"Missing data for {dataset_cfg.name}. Run prepare_data first.")

    df = pd.read_parquet(index_path)

    if split_path.exists():
        with open(split_path, "r") as f:
            splits = json.load(f)
        test_files = splits.get("test", [])
        if test_files:
            df = df[df['slide_name'].isin(test_files)].reset_index(drop=True)

    return df

def get_dataloader(
    train_cfg: TrainConfig,
):
    dataset_cfg = AgNorOMEConfig()
    test_index = _load_ag_nor_test_df(dataset_cfg)
    test_dataset = OMEDataset(index_df=test_index, transform=get_transforms("val"))

    print(f"Data Setup [AgNor]: {len(test_dataset)} test samples.")

    return DataLoader(
        test_dataset,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        shuffle=False,
    )
