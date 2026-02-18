import re
import subprocess
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import lightning as L
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from src import config

from torch.utils.data import Dataset

LINKS = {
    "Data_channel.zip": "https://ndownloader.figshare.com/files/10616965",
}


class Jiang2018Dataset(Dataset):
    """Dataset for pre-tiled Jiang 2018 defocus segments.

    Labels are parsed directly from filenames (e.g. ``Seg01_defocus-1000.jpg``
    → offset = −1.0 µm).  No preprocessing or caching needed.
    """

    def __init__(self, root_dir: Path, transform=None):
        self.transform = transform
        self.samples: list[tuple[Path, float]] = []  # (path, offset_µm)

        for f in sorted(root_dir.rglob("Seg*.jpg")):
            if "incoherent_RGBchannels" not in f.as_posix():
                continue
            match = re.search(r"defocus(-?\d+)", f.name)
            if match:
                self.samples.append((f, float(match.group(1)) / 1000.0))

        print(f"Jiang2018Dataset: {len(self.samples)} samples found.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path, offset = self.samples[idx]
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=img)["image"]
        else:
            image = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return {
            "image": image,
            "target": torch.tensor(offset, dtype=torch.float32),
            "metadata": {"filename": path.name},
        }


class Jiang2018DataModule(L.LightningDataModule):
    """Self-contained test-only DataModule with auto-download and extraction."""

    def __init__(
        self,
        batch_size: int = 128,
        num_workers: int = 4,
    ):
        super().__init__()
        self.batch_size, self.num_workers = batch_size, num_workers
        self.dataset_name = "Jiang2018"
        self.zip_dir = config.get_vsi_zip_dir(self.dataset_name)
        self.raw_dir = config.get_vsi_raw_dir(self.dataset_name)
        self.test_ds = None

    @property
    def val_transform(self):
        return A.Compose(
            [
                A.ToFloat(max_value=255.0),
                ToTensorV2(),  # HWC → CHW
            ]
        )

    def prepare_data(self):
        self.zip_dir.mkdir(parents=True, exist_ok=True)
        for name, url in LINKS.items():
            if not (self.zip_dir / name).exists():
                print(f"Downloading {name}...")
                subprocess.run(
                    ["curl", "-L", "-o", str(self.zip_dir / name), url], check=True
                )

        expected_folders = ["incoherent_RGBchannels"]
        if all((self.raw_dir / d).exists() for d in expected_folders):
            print(
                f"Required data already exists in {self.raw_dir}. Skipping extraction."
            )
            return

        zips = sorted(list(self.zip_dir.glob("*.zip")))
        print(f"Extracting {len(zips)} zip files to {self.raw_dir}...")
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        for zip_path in zips:
            print(f"Unzipping {zip_path.name}...")
            try:
                subprocess.run(
                    ["unzip", "-q", "-o", str(zip_path), "-d", str(self.raw_dir)],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                print(f"Failed to unzip {zip_path.name}: {e}")

    def setup(self, stage: str | None = None):
        if self.test_ds is None:
            self.test_ds = Jiang2018Dataset(
                self.raw_dir,
                transform=self.val_transform,
            )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

    def predict_dataloader(self):
        return self.test_dataloader()
