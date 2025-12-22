import lightning as L
from torch.utils.data import DataLoader
import torch
import pickle
from typing import Optional

import config
from src.dataset.vsi_dataset_lightning import VSIDatasetLightning
from src.processing.download import download_dataset
from src.processing.fix_zip import fix_zip_structure
from src.processing.create_split import create_split
from src.processing.preprocess import preprocess_dataset


class VSIDataModule(L.LightningDataModule):
    """
    Handles preparation, loading indices, and managing splits for a single dataset.
    """

    def __init__(
        self,
        dataset_name: str = config.DATASET_NAME,
        batch_size: int = config.BATCH_SIZE,
        num_workers: int = config.NUM_WORKERS,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset: Optional[torch.utils.data.Dataset] = None
        self.val_dataset: Optional[torch.utils.data.Dataset] = None
        self.test_dataset: Optional[torch.utils.data.Dataset] = None

    def prepare_data(self):
        """Preparation logic (download, unzip, split, preprocess) for the dataset."""
        print(f"Ensuring data environment for {self.dataset_name} is ready...")

        # Always prepare specified dataset
        create_split(dataset_name=self.dataset_name)
        download_dataset(dataset_name=self.dataset_name)
        fix_zip_structure(dataset_name=self.dataset_name)
        preprocess_dataset(dataset_name=self.dataset_name)

        print(f"\nData preparation for {self.dataset_name} complete.")

    def _load_index(self, mode: str) -> Optional[dict]:
        path = config.get_index_path(mode, dataset_name=self.dataset_name)
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_index = self._load_index("train")
            val_index = self._load_index("val")

            if train_index:
                self.train_dataset = VSIDatasetLightning(index_data=train_index)
            if val_index:
                self.val_dataset = VSIDatasetLightning(index_data=val_index)

            train_size = len(self.train_dataset) if self.train_dataset else 0
            val_size = len(self.val_dataset) if self.val_dataset else 0
            print(
                f"DataModule Setup (fit) for {self.dataset_name}: {train_size} train, {val_size} val samples."
            )

        if stage == "test" or stage == "predict" or stage is None:
            test_index = self._load_index("test")
            if test_index:
                self.test_dataset = VSIDatasetLightning(index_data=test_index)

            test_size = len(self.test_dataset) if self.test_dataset else 0
            print(
                f"DataModule Setup ({stage}) for {self.dataset_name}: {test_size} samples."
            )

    def _empty_dataloader(self):
        """Returns an empty DataLoader."""
        return DataLoader([], batch_size=self.batch_size)

    def train_dataloader(self):
        if self.train_dataset is None:
            print(
                f"Warning: No train dataset for {self.dataset_name}. Returning empty dataloader."
            )
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
            print(
                f"Warning: No validation dataset for {self.dataset_name}. Returning empty dataloader."
            )
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
            print(
                f"Warning: No test dataset for {self.dataset_name}. Returning empty dataloader."
            )
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
