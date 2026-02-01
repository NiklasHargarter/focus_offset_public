import lightning as L
from torch.utils.data import DataLoader
import json
from typing import Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

from src import config
from src.dataset.ome_dataset import OMEDataset
from src.dataset.vsi_types import MasterIndex, ProcessedIndex
from src.dataset.vsi_prep.preprocess import load_master_index


class OMEDataModule(L.LightningDataModule):
    """
    Lightning DataModule for OME-TIFF datasets.
    Handles AgNor and other multi-series OME-TIFF stacks.
    """

    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        num_workers: int,
        patch_size: int,
        stride: int,
        min_tissue_coverage: float,
        downsample_factor: int = 1,
        val_ratio: float = 0.1,
        seed: int = 42,
        prefetch_factor: int = 4,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.stride = stride
        self.downsample_factor = downsample_factor
        self.min_tissue_coverage = min_tissue_coverage
        self.val_ratio = val_ratio
        self.seed = seed
        self.prefetch_factor = prefetch_factor

        self.train_dataset: Optional[OMEDataset] = None
        self.val_dataset: Optional[OMEDataset] = None
        self.test_dataset: Optional[OMEDataset] = None

    @property
    def train_transform(self):
        return A.Compose(
            [
                A.D4(p=1.0),
                A.ColorJitter(
                    brightness=(0.9, 1.1),
                    contrast=(0.9, 1.1),
                    saturation=(0.9, 1.1),
                    hue=(-0.1, 0.1),
                    p=0.5,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    @property
    def val_transform(self):
        return A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def prepare_data(self):
        master_index_path = config.get_master_index_path(
            self.dataset_name, patch_size=self.patch_size
        )
        if not master_index_path.exists():
            print(
                f"\n[ERROR] Required master index for {self.dataset_name} is missing."
            )
            print(
                f"Please run: python src/dataset/ome_prep/preprocess.py --dataset {self.dataset_name}"
            )
            raise RuntimeError("Data preparation check failed.")

    def setup(self, stage: Optional[str] = None):
        master_index = load_master_index(self.dataset_name, self.patch_size)
        split_path = config.get_split_path(self.dataset_name)

        if not split_path.exists():
            # If no split exists, we can create a simple one or just handle it here
            # For AgNor, the user might want to run create_split.py first.
            print(
                f"Warning: Split file {split_path} missing. Using all data for train."
            )
            splits = {
                "train_pool": [s.name for s in master_index.file_registry],
                "test": [],
            }
        else:
            with open(split_path, "r") as f:
                splits = json.load(f)

        if stage == "fit":
            raise RuntimeError(
                f"{self.__class__.__name__} is for evaluation only. Training is disabled."
            )

        if stage == "test" or stage == "predict" or stage is None:
            test_files = splits.get("test", [])
            self.test_dataset = OMEDataset(
                index_data=self._filter_index(master_index, test_files),
                transform=self.val_transform,
            )

    def _filter_index(
        self, master_index: MasterIndex, files: list[str]
    ) -> ProcessedIndex:
        name_to_entry = {entry.name: entry for entry in master_index.file_registry}
        filtered = []
        counts = []
        for name in files:
            if name in name_to_entry:
                entry = name_to_entry[name]
                filtered.append(entry)
                counts.append(entry.total_samples)

        return ProcessedIndex(
            file_registry=filtered,
            cumulative_indices=np.cumsum(counts),
            patch_size=self.patch_size,
            downsample_factor=self.downsample_factor,
            dataset_name=self.dataset_name,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
