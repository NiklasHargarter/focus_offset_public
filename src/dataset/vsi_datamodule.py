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
    LightningDataModule for the VSI dataset.
    Handles preparation (download/preprocess), loading indices from disk,
    and managing dataset splits.
    """

    def __init__(
        self,
        batch_size: int = config.BATCH_SIZE,
        num_workers: int = config.NUM_WORKERS,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset: Optional[torch.utils.data.Dataset] = None
        self.val_dataset: Optional[torch.utils.data.Dataset] = None
        self.test_dataset: Optional[torch.utils.data.Dataset] = None

    def prepare_data(self):
        """Preparation logic (download, unzip, split, preprocess)."""
        print("Ensuring data environment is ready...")

        create_split()
        download_dataset()
        fix_zip_structure()
        preprocess_dataset()

        print("\nData preparation check complete.")

    def _load_index(self, mode: str) -> dict:
        path = config.get_index_path(mode)
        with open(path, "rb") as f:
            return pickle.load(f)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_index = self._load_index("train")
            val_index = self._load_index("val")

            self.train_dataset = VSIDatasetLightning(index_data=train_index)
            self.val_dataset = VSIDatasetLightning(index_data=val_index)

            print(
                f"DataModule Setup (fit): {len(self.train_dataset)} train samples, "
                f"{len(self.val_dataset)} val samples."
            )

        if stage == "test" or stage == "predict" or stage is None:
            test_index = self._load_index("test")
            self.test_dataset = VSIDatasetLightning(index_data=test_index)
            print(f"DataModule Setup ({stage}): {len(self.test_dataset)} samples.")

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError(
                "Train dataset not initialized. Call setup('fit') first."
            )
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
            raise RuntimeError(
                "Validation dataset not initialized. Call setup('fit') first."
            )
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
            raise RuntimeError(
                "Test dataset not initialized. Call setup('test') first."
            )
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
