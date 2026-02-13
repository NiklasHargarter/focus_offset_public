import json
import random

import albumentations as A
import lightning as L
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from src import config
from src.dataset.vsi_dataset_lightning import VSIDatasetLightning
from src.dataset.vsi_prep.preprocess import load_master_index
from src.dataset.vsi_types import MasterIndex, ProcessedIndex


class BaseVSIDataModule(L.LightningDataModule):
    """Shared base for all VSI DataModules.

    Preprocessing settings (stride, downsample, coverage) locate the correct
    cache directory.  Runtime settings (batch_size, workers) control the
    DataLoaders.
    """

    def __init__(
        self,
        dataset_name: str = config.DATASET_NAME,
        stride: int = config.STRIDE,
        downsample_factor: int = config.DOWNSAMPLE_FACTOR,
        min_tissue_coverage: float = config.MIN_TISSUE_COVERAGE,
        batch_size: int = config.BATCH_SIZE,
        num_workers: int = config.NUM_WORKERS,
        prefetch_factor: int = config.PREFETCH_FACTOR,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.stride = stride
        self.downsample_factor = downsample_factor
        self.min_tissue_coverage = min_tissue_coverage
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        self.train_dataset: torch.utils.data.Dataset | None = None
        self.val_dataset: torch.utils.data.Dataset | None = None
        self.test_dataset: torch.utils.data.Dataset | None = None

    @property
    def train_transform(self):
        return A.Compose(
            [
                # 1. Geometric (Scale/Rotation Invariance)
                A.D4(p=1.0),
                # 2. Aggressive Chroma/Stain Destruction
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
                A.ToFloat(max_value=255.0),  # uint8 → float32 [0, 1]
                ToTensorV2(),  # HWC → CHW (stain-agnostic)
            ]
        )

    @property
    def val_transform(self):
        return A.Compose(
            [
                A.ToFloat(max_value=255.0),  # uint8 → float32 [0, 1]
                ToTensorV2(),  # HWC → CHW (stain-agnostic)
            ]
        )

    def prepare_data(self):
        """
        Quick check that preprocessed data exists for this dataset.
        All preprocessing params are baked into the master index at preprocess time.
        """
        master_index = load_master_index(
            self.dataset_name, self.stride, self.downsample_factor, self.min_tissue_coverage
        )

        if master_index is None:
            print(
                f"\n[ERROR] Master index for {self.dataset_name} not found.\n"
                f"Run preprocessing first:\n"
                f"  python src/dataset/vsi_prep/sync.py --dataset {self.dataset_name}"
            )
            raise RuntimeError(
                "Data preparation check failed. See log above for instructions."
            )

        # Check slide indices exist
        indices_dir = config.get_slide_index_dir(
            self.dataset_name, self.stride, self.downsample_factor, self.min_tissue_coverage
        )
        if not any(indices_dir.glob("*.pkl")):
            raise RuntimeError(
                f"No individual slide indices found in {indices_dir}. "
                f"Run preprocessing for {self.dataset_name}."
            )

        cfg = master_index.config_state
        print(
            f"Data environment for {self.dataset_name} OK "
            f"(p{cfg.patch_size}, stride={cfg.stride}, ds={cfg.downsample_factor})."
        )

    def _filter_index(
        self, master_index: MasterIndex, files: list[str]
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
                    f"Warning: File {name} not found in master index for {self.dataset_name}"
                )

        cumulative_indices = (
            np.cumsum(counts) if counts else np.array([], dtype=np.int64)
        )

        return ProcessedIndex(
            file_registry=filtered_registry,
            cumulative_indices=cumulative_indices,
            patch_size=config.PATCH_SIZE,
            downsample_factor=master_index.config_state.downsample_factor,
            dataset_name=self.dataset_name,
        )

    def _load_data_indices(self) -> tuple[MasterIndex, dict]:
        master_index = load_master_index(
            self.dataset_name, self.stride, self.downsample_factor, self.min_tissue_coverage
        )
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
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
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
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
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
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def predict_dataloader(self):
        return self.test_dataloader()


class HEHoldOutDataModule(BaseVSIDataModule):
    """HE DataModule with a simple static train/val split from the train pool."""

    def __init__(self, val_ratio: float = 0.1, seed: int = 42, **kwargs):
        if "dataset_name" not in kwargs:
            kwargs["dataset_name"] = config.DATASET_NAME
        super().__init__(**kwargs)
        self.val_ratio = val_ratio
        self.seed = seed

    def setup(self, stage: str | None = None):
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

            self.train_dataset = VSIDatasetLightning(
                index_data=train_index, transform=self.train_transform
            )
            self.val_dataset = VSIDatasetLightning(
                index_data=val_index, transform=self.val_transform
            )

            print(
                f"DataModule Setup (fit) [HoldOut]: {len(self.train_dataset)} train, {len(self.val_dataset)} val samples."
            )

        if stage == "test" or stage == "predict" or stage is None:
            test_index = self._filter_index(master_index, splits["test"])
            self.test_dataset = VSIDatasetLightning(
                index_data=test_index, transform=self.val_transform
            )


class HEFoldDataModule(BaseVSIDataModule):
    """HE DataModule with K-fold cross-validation split from the train pool."""

    def __init__(self, fold_idx: int = 0, num_folds: int = 5, seed: int = 42, **kwargs):
        if "dataset_name" not in kwargs:
            kwargs["dataset_name"] = config.DATASET_NAME
        super().__init__(**kwargs)
        self.fold_idx = fold_idx
        self.num_folds = num_folds
        self.seed = seed

    def setup(self, stage: str | None = None):
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

            self.train_dataset = VSIDatasetLightning(
                index_data=train_index, transform=self.train_transform
            )
            self.val_dataset = VSIDatasetLightning(
                index_data=val_index, transform=self.val_transform
            )

            print(
                f"DataModule Setup (fit) [Fold {self.fold_idx}]: {len(self.train_dataset)} train, {len(self.val_dataset)} val samples."
            )

        if stage == "test" or stage == "predict" or stage is None:
            test_index = self._filter_index(master_index, splits["test"])
            self.test_dataset = VSIDatasetLightning(
                index_data=test_index, transform=self.val_transform
            )


class IHCDataModule(BaseVSIDataModule):
    """IHC DataModule, typically used for tests/evaluation only."""

    def __init__(self, **kwargs):
        if "dataset_name" not in kwargs:
            kwargs["dataset_name"] = "ZStack_IHC"
        super().__init__(**kwargs)

    def setup(self, stage: str | None = None):
        master_index, splits = self._load_data_indices()

        if stage == "fit":
            raise RuntimeError(
                f"{self.__class__.__name__} is for evaluation only. Training is disabled."
            )

        if stage == "test" or stage == "predict" or stage is None:
            test_files = splits.get("test", [])

            test_index = self._filter_index(master_index, test_files)
            self.test_dataset = VSIDatasetLightning(
                index_data=test_index, transform=self.val_transform
            )


VSIDataModule = HEHoldOutDataModule
