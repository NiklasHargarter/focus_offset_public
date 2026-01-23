import lightning as L
from torch.utils.data import DataLoader
import torch
import pickle
import json
import random
from typing import Optional
from pathlib import Path

from src import config
from src.dataset.vsi_dataset_lightning import VSIDatasetLightning
from src.dataset.vsi_types import MasterIndex, ProcessedIndex, PreprocessConfig
from src.dataset.vsi_prep.preprocess import load_master_index


class BaseVSIDataModule(L.LightningDataModule):
    """
    Shared base class for VSI DataModules.
    Handles preparation, and master index filtering.
    """

    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        num_workers: int,
        patch_size: int,
        stride: int,
        min_tissue_coverage: float,
        split_ratio: float = 0.3,
        force_preprocess: bool = False,
        cache_dir: str = "cache",
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.stride = stride
        self.min_tissue_coverage = min_tissue_coverage
        self.split_ratio = split_ratio
        self.force_preprocess = force_preprocess
        self.cache_dir = Path(cache_dir)

        self.train_dataset: Optional[torch.utils.data.Dataset] = None
        self.val_dataset: Optional[torch.utils.data.Dataset] = None
        self.test_dataset: Optional[torch.utils.data.Dataset] = None

    def prepare_data(self):
        """
        Quick check for required precomputed data.
        For 800GB+ datasets, we decouple syncing from the training loop.
        """
        master_index_path = config.get_master_index_path(
            self.dataset_name, patch_size=self.patch_size
        )

        expected_config = PreprocessConfig(
            patch_size=self.patch_size,
            stride=self.stride,
            min_tissue_coverage=self.min_tissue_coverage,
            dataset_name=self.dataset_name,
        )

        missing = []
        if not master_index_path.exists():
            missing.append(f"Master Manifest: {master_index_path}")
        else:
            # Check manifest config
            with open(master_index_path, "rb") as f:
                manifest_data = pickle.load(f)

            if manifest_data["config_state"] != expected_config:
                missing.append(
                    f"Master Index Configuration Mismatch!\n"
                    f"  Expected: {expected_config}\n"
                    f"  Found:    {manifest_data['config_state']}"
                )

        # Also check if any slide indices exist
        indices_dir = config.get_slide_index_dir(self.dataset_name, self.patch_size)
        if not any(indices_dir.glob("*.pkl")):
            missing.append(f"No individual slide indices found in {indices_dir}")

        if missing:
            error_msg = "\n".join(missing)
            print(
                f"\n[ERROR] Required data for {self.dataset_name} is missing or outdated:\n{error_msg}"
            )
            print(
                "\nTo resolve this, please run the standalone synchronization script:"
            )
            print(
                f"  python src/dataset/vsi_prep/sync.py --dataset {self.dataset_name} "
                f"--patch_size {self.patch_size} --stride {self.stride}"
            )
            print(
                "\nNote: For 200GB+ datasets, this ensures a clean and traceable environment."
            )
            raise RuntimeError(
                "Data preparation check failed. See log above for sync instructions."
            )

        print(f"Data environment for {self.dataset_name} (p{self.patch_size}) is OK.")

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
        master_index = load_master_index(self.dataset_name, self.patch_size)
        split_path = config.get_split_path(self.dataset_name)

        if master_index is None or not split_path.exists():
            raise RuntimeError(
                f"Missing data for {self.dataset_name}. Run prepare_data first."
            )

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

            slides = []
            name_to_count = {
                entry.name: entry.total_samples for entry in master_index.file_registry
            }
            for name in train_pool:
                slides.append({"name": name, "count": name_to_count[name]})

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

        if stage == "fit" or stage is None:
            train_index = self._filter_index(master_index, splits.get("train_pool", []))
            self.train_dataset = VSIDatasetLightning(index_data=train_index)

            self.val_dataset = None

        if stage == "test" or stage == "predict" or stage is None:
            test_files = splits.get("test", [])

            test_index = self._filter_index(master_index, test_files)
            self.test_dataset = VSIDatasetLightning(index_data=test_index)


VSIDataModule = HEHoldOutDataModule
