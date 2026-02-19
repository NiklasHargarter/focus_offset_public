import pickle
import re
import subprocess
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import lightning as L
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src import config

# Folder names within the incoherent_RGBchannels root
_SPLIT_DIRS = {
    "train": "train_incoherent_RGBChannels",
    "test_same": "testRawData_incoherent_sameProtocol",
    "test_diff": "testRawData_incoherent_diffProtocol",
}

LINKS = {
    "Data_channel.zip": "https://ndownloader.figshare.com/files/10616965",
}

# Folder name pattern: s{segment}_l{location}
_FOLDER_RE = re.compile(r"^s(\d+)_l(\d+)$")

# Train filename: Seg{seg_id}_defocus{offset}.jpg
_TRAIN_FILE_RE = re.compile(r"^Seg(\d+)_defocus(-?\d+)\.jpg$")

# Test filename: defocus{offset}.jpg
_TEST_FILE_RE = re.compile(r"^defocus(-?\d+)\.jpg$")


class Jiang2018Dataset(Dataset):
    """Dataset for pre-tiled Jiang 2018 defocus segments.

    Supports the official folder-level train/test split.  Labels and metadata
    are parsed directly from filenames and folder names.

    Args:
        root_dir: Path to the ``incoherent_RGBchannels`` directory.
        split: One of ``"train"``, ``"test_same"``, ``"test_diff"``.
        transform: Optional albumentations transform.
    """

    def __init__(
        self,
        root_dir: Path,
        split: str,
        transform=None,
    ):
        if split not in _SPLIT_DIRS:
            raise ValueError(
                f"Invalid split '{split}'. Choose from {list(_SPLIT_DIRS.keys())}."
            )
        self.split = split
        self.transform = transform

        split_dir = root_dir / _SPLIT_DIRS[split]
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}\n"
                "Make sure the dataset has been downloaded and extracted."
            )

        self.samples: list[dict] = self._load_samples(split_dir, split)
        print(f"Jiang2018Dataset [{split}]: {len(self.samples)} samples found.")

    @staticmethod
    def _cache_path(split: str) -> Path:
        cache_dir = config.CACHE_DIR / "Jiang2018"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"samples_{split}.pkl"

    @staticmethod
    def _load_samples(split_dir: Path, split: str) -> list[dict]:
        cache_path = Jiang2018Dataset._cache_path(split)
        if cache_path.exists():
            print(f"Jiang2018Dataset [{split}]: loading from cache {cache_path}")
            with cache_path.open("rb") as fh:
                return pickle.load(fh)

        print(
            f"Jiang2018Dataset [{split}]: scanning {split_dir} (first run, will cache)..."
        )
        samples: list[dict] = []

        # Fixed-depth scan: split_dir/s{n}_l{m}/*.jpg (always exactly 2 levels)
        all_files: list[Path] = []
        for subdir in split_dir.iterdir():
            if subdir.is_dir():
                all_files.extend(subdir.glob("*.jpg"))
        all_files.sort()

        for f in all_files:
            folder_match = _FOLDER_RE.match(f.parent.name)
            if not folder_match:
                continue

            segment = int(folder_match.group(1))
            location = int(folder_match.group(2))

            if split == "train":
                file_match = _TRAIN_FILE_RE.match(f.name)
                if not file_match:
                    continue
                seg_id = int(file_match.group(1))
                defocus_nm = int(file_match.group(2))
                samples.append(
                    {
                        "path": f,
                        "offset_um": defocus_nm / 1000.0,
                        "defocus_nm": defocus_nm,
                        "segment": segment,
                        "location": location,
                        "seg_id": seg_id,
                        "split": split,
                    }
                )
            else:
                file_match = _TEST_FILE_RE.match(f.name)
                if not file_match:
                    continue
                seg_id = -1
                defocus_nm = int(file_match.group(1))
                # Protocol: Test images (2048x2448) are binned 4:1 (resize to 1024x1224)
                # and then tiled into 224x224 segments. (4x5 = 20 tiles)
                for r in range(4):
                    for c in range(5):
                        samples.append(
                            {
                                "path": f,
                                "offset_um": defocus_nm / 1000.0,
                                "defocus_nm": defocus_nm,
                                "segment": segment,
                                "location": location,
                                "seg_id": seg_id,
                                "split": split,
                                "tile_coords": (r, c),
                            }
                        )

        with cache_path.open("wb") as fh:
            pickle.dump(samples, fh)
        print(
            f"Jiang2018Dataset [{split}]: cached {len(samples)} samples to {cache_path}"
        )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        img = cv2.imread(str(sample["path"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if sample["split"] != "train":
            # Protocol: 4:1 binning (pixel size doubled -> image dimension halved)
            img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            # Tiling: Crop 224x224 segment
            r, c = sample["tile_coords"]
            y, x = r * 224, c * 224
            img = img[y : y + 224, x : x + 224]

        if self.transform:
            image = self.transform(image=img)["image"]
        else:
            image = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        res = {
            "image": image,
            "target": torch.tensor(sample["offset_um"], dtype=torch.float32),
            "metadata": {
                "filename": sample["path"].name,
                "split": sample["split"],
                "segment": sample["segment"],
                "location": sample["location"],
                "seg_id": sample["seg_id"] if sample["seg_id"] is not None else -1,
                "defocus_nm": sample["defocus_nm"],
            },
        }
        if "tile_coords" in sample:
            res["metadata"]["tile_coords"] = torch.tensor(sample["tile_coords"])
        return res


class Jiang2018DataModule(L.LightningDataModule):
    """Self-contained DataModule with auto-download and extraction.

    Provides the official train/test splits for the incoherent illumination
    condition.  ``test_dataloader`` / ``predict_dataloader`` combine both test
    splits (same-protocol and diff-protocol) into a single loader; the
    ``split`` field in each sample's metadata identifies which sub-split it
    came from.
    """

    def __init__(
        self,
        batch_size: int = 128,
        num_workers: int = 4,
    ):
        super().__init__()
        self.batch_size, self.num_workers = batch_size, num_workers
        self.dataset_name = "Jiang2018"
        self.zip_dir = config.get_vsi_zip_dir(self.dataset_name)
        self.raw_dir = (
            config.get_vsi_raw_dir(self.dataset_name) / "incoherent_RGBchannels"
        )
        self.train_ds = None
        self.test_same_ds = None
        self.test_diff_ds = None

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

        expected_folders = [
            "incoherent_RGBchannels/train_incoherent_RGBChannels",
            "incoherent_RGBchannels/testRawData_incoherent_sameProtocol",
            "incoherent_RGBchannels/testRawData_incoherent_diffProtocol",
        ]
        raw_base = config.get_vsi_raw_dir(self.dataset_name)
        if all((raw_base / d).exists() for d in expected_folders):
            print(f"Required data already exists in {raw_base}. Skipping extraction.")
            return

        zips = sorted(list(self.zip_dir.glob("*.zip")))
        print(f"Extracting {len(zips)} zip files to {raw_base}...")
        raw_base.mkdir(parents=True, exist_ok=True)

        for zip_path in zips:
            print(f"Unzipping {zip_path.name}...")
            try:
                subprocess.run(
                    ["unzip", "-q", "-o", str(zip_path), "-d", str(raw_base)],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                print(f"Failed to unzip {zip_path.name}: {e}")

    def setup(self, stage: str | None = None):
        if stage in ("fit", "train", None) and self.train_ds is None:
            self.train_ds = Jiang2018Dataset(
                self.raw_dir,
                split="train",
                transform=self.val_transform,
            )
        if stage in ("test", "predict", None):
            if self.test_same_ds is None:
                self.test_same_ds = Jiang2018Dataset(
                    self.raw_dir,
                    split="test_same",
                    transform=self.val_transform,
                )
            if self.test_diff_ds is None:
                self.test_diff_ds = Jiang2018Dataset(
                    self.raw_dir,
                    split="test_diff",
                    transform=self.val_transform,
                )

    def _make_loader(self, dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=torch.cuda.is_available(),
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_ds, shuffle=True)

    def test_dataloader(self) -> DataLoader:
        combined = ConcatDataset([self.test_same_ds, self.test_diff_ds])
        return self._make_loader(combined)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
