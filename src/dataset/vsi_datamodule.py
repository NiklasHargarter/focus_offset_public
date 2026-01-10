import lightning as L
from torch.utils.data import DataLoader
import torch
import pickle
import json
from typing import Optional
from pathlib import Path

import config
from src.dataset.vsi_dataset_lightning import VSIDatasetLightning
from src.dataset.vsi_prep.download import download_dataset
from src.dataset.vsi_prep.fix_zip import fix_zip_structure
from src.dataset.vsi_prep.create_split import create_split
from src.dataset.vsi_prep.preprocess import preprocess_dataset


from src.dataset.vsi_types import MasterIndex, ProcessedIndex


class BaseVSIDataModule(L.LightningDataModule):
    """
    Shared base class for VSI DataModules.
    Handles preparation, and master index filtering.
    """

    def __init__(
        self,
        dataset_name: str,
        batch_size: int = 128,
        num_workers: int = 16,
        patch_size: int = 224,
        stride: int = 224,
        downscale_factor: int = 8,
        min_tissue_coverage: float = 0.05,
        cache_dir: str = "cache",
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.stride = stride
        self.downscale_factor = downscale_factor
        self.min_tissue_coverage = min_tissue_coverage
        self.cache_dir = Path(cache_dir)

        self.train_dataset: Optional[torch.utils.data.Dataset] = None
        self.val_dataset: Optional[torch.utils.data.Dataset] = None
        self.test_dataset: Optional[torch.utils.data.Dataset] = None

    def prepare_data(self):
        """Preparation logic (download, unzip, split, preprocess) for the dataset."""
        print(f"Ensuring data environment for {self.dataset_name} is ready...")

        # 1. Download and fix zips if necessary
        download_dataset(dataset_name=self.dataset_name)
        fix_zip_structure(dataset_name=self.dataset_name)

        # 2. Preprocess entire dataset to get master index
        preprocess_dataset(
            dataset_name=self.dataset_name,
            patch_size=self.patch_size,
            stride=self.stride,
            downscale_factor=self.downscale_factor,
            min_tissue_coverage=self.min_tissue_coverage,
        )

        # 3. Create splits (test vs train_pool) based on master index
        create_split(
            dataset_name=self.dataset_name,
        )

        print(f"\nData preparation for {self.dataset_name} complete.")

    def _filter_index(
        self, master_index: MasterIndex, files: list[str]
    ) -> ProcessedIndex:
        """Filter master index for a specific set of filenames and compute cumulative indices."""
        file_registry = master_index.file_registry
        name_to_entry = {entry.name: entry for entry in file_registry}

        filtered_registry = []
        cumulative_indices = []
        cumulative_count = 0

        for name in files:
            if name in name_to_entry:
                res = name_to_entry[name]
                cumulative_count += res.total_samples
                cumulative_indices.append(cumulative_count)
                filtered_registry.append(res)
            else:
                print(
                    f"Warning: File {name} not found in master index for {self.dataset_name}"
                )

        return ProcessedIndex(
            file_registry=filtered_registry,
            cumulative_indices=cumulative_indices,
            patch_size=master_index.patch_size,
        )

    def _load_data_indices(self) -> tuple[MasterIndex, dict]:
        master_index_path = config.get_master_index_path(self.dataset_name)
        split_path = config.get_split_path(self.dataset_name)

        if not master_index_path.exists() or not split_path.exists():
            raise RuntimeError(
                f"Missing data for {self.dataset_name}. Run prepare_data first."
            )

        with open(master_index_path, "rb") as f:
            master_index: MasterIndex = pickle.load(f)
        with open(split_path, "r") as f:
            splits = json.load(f)

        return master_index, splits

    def _empty_dataloader(self):
        return DataLoader([], batch_size=self.batch_size)

    def train_dataloader(self):
        if self.train_dataset is None:
            return self._empty_dataloader()
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return self._empty_dataloader()
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return self._empty_dataloader()
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )

    def predict_dataloader(self):
        return self.test_dataloader()


class HEHoldOutDataModule(BaseVSIDataModule):
    """HE DataModule with a simple static train/val split from the train pool."""

    def __init__(self, val_ratio: float = 0.1, seed: int = 42, **kwargs):
        if "dataset_name" not in kwargs:
            kwargs["dataset_name"] = "ZStack_HE"
        super().__init__(**kwargs)
        self.val_ratio = val_ratio
        self.seed = seed

    def setup(self, stage: Optional[str] = None):
        master_index, splits = self._load_data_indices()

        if stage == "fit" or stage is None:
            train_pool = splits["train_pool"]

            import random

            random.seed(self.seed)
            random.shuffle(train_pool)

            num_val = int(len(train_pool) * self.val_ratio)
            val_files = train_pool[:num_val]
            train_files = train_pool[num_val:]

            train_index = self._filter_index(master_index, train_files)
            val_index = self._filter_index(master_index, val_files)

            self.train_dataset = VSIDatasetLightning(index_data=train_index)
            self.val_dataset = VSIDatasetLightning(index_data=val_index)

            print(
                f"DataModule Setup (fit) [HoldOut]: {len(self.train_dataset)} train, {len(self.val_dataset)} val samples."
            )

        if stage == "test" or stage == "predict" or stage is None:
            test_index = self._filter_index(master_index, splits["test"])
            self.test_dataset = VSIDatasetLightning(index_data=test_index)


class HEFoldDataModule(BaseVSIDataModule):
    """HE DataModule with K-fold cross-validation split from the train pool."""

    def __init__(self, fold_idx: int = 0, num_folds: int = 5, seed: int = 42, **kwargs):
        if "dataset_name" not in kwargs:
            kwargs["dataset_name"] = "ZStack_HE"
        super().__init__(**kwargs)
        self.fold_idx = fold_idx
        self.num_folds = num_folds
        self.seed = seed

    def setup(self, stage: Optional[str] = None):
        master_index, splits = self._load_data_indices()

        if stage == "fit" or stage is None:
            train_pool = splits["train_pool"]

            # Simple Round-Robin on a sorted list by size
            slides = []
            name_to_count = {
                entry.name: entry.total_samples for entry in master_index.file_registry
            }
            for name in train_pool:
                slides.append({"name": name, "count": name_to_count[name]})

            # Sort by size for packing (descending)
            slides = sorted(slides, key=lambda x: x["count"], reverse=True)

            folds = [[] for _ in range(self.num_folds)]
            for i, slide in enumerate(slides):
                folds[i % self.num_folds].append(slide["name"])

            val_files = folds[self.fold_idx]
            train_files = []
            for i, f in enumerate(folds):
                if i != self.fold_idx:
                    train_files.extend(f)

            train_index = self._filter_index(master_index, train_files)
            val_index = self._filter_index(master_index, val_files)

            self.train_dataset = VSIDatasetLightning(index_data=train_index)
            self.val_dataset = VSIDatasetLightning(index_data=val_index)

            print(
                f"DataModule Setup (fit) [Fold {self.fold_idx}]: {len(self.train_dataset)} train, {len(self.val_dataset)} val samples."
            )

        if stage == "test" or stage == "predict" or stage is None:
            test_index = self._filter_index(master_index, splits["test"])
            self.test_dataset = VSIDatasetLightning(index_data=test_index)


class IHCDataModule(BaseVSIDataModule):
    """IHC DataModule, typically used for tests/evaluation only."""

    def __init__(self, **kwargs):
        if "dataset_name" not in kwargs:
            kwargs["dataset_name"] = "ZStack_IHC"
        super().__init__(**kwargs)

    def setup(self, stage: Optional[str] = None):
        master_index, splits = self._load_data_indices()
        # For IHC, we often just use everything as test or have simplified splits
        # Here, we follow the same splits.json structure if it exists
        if stage == "fit" or stage is None:
            train_index = self._filter_index(master_index, splits.get("train_pool", []))
            self.train_dataset = VSIDatasetLightning(index_data=train_index)
            # Empty val
            self.val_dataset = None

        if stage == "test" or stage == "predict" or stage is None:
            test_files = splits.get("test", [])
            # If test is empty but train_pool is not, maybe it's an evaluation-only dataset?
            # Adjust as needed for IHC specific usage.
            test_index = self._filter_index(master_index, test_files)
            self.test_dataset = VSIDatasetLightning(index_data=test_index)
