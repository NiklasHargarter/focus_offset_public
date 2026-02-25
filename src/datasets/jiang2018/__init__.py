import pickle
import re
import subprocess
from pathlib import Path
from typing import Any

import albumentations as A
import cv2

import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .config import CACHE_DIR, Jiang2018Config

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
    """Dataset for pre-tiled Jiang 2018 defocus segments."""

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

    @staticmethod
    def _cache_path(split: str) -> Path:
        cache_dir = CACHE_DIR / "Jiang2018"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"samples_{split}.pkl"

    @staticmethod
    def _load_samples(split_dir: Path, split: str) -> list[dict]:
        cache_path = Jiang2018Dataset._cache_path(split)
        if cache_path.exists():
            with cache_path.open("rb") as fh:
                return pickle.load(fh)

        samples: list[dict] = []

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
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        img = cv2.imread(str(sample["path"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if sample["split"] != "train":
            img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
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
                "x": 0,
                "y": 0,
            },
        }
        if "tile_coords" in sample:
            r, c = sample["tile_coords"]
            res["metadata"]["x"] = c * 224
            res["metadata"]["y"] = r * 224
        return res


def download_and_extract_jiang2018():
    """Download and extract the Jiang2018 dataset."""
    dataset_cfg = Jiang2018Config()
    zip_dir = dataset_cfg.zip_dir

    zip_dir.mkdir(parents=True, exist_ok=True)
    for name, url in LINKS.items():
        if not (zip_dir / name).exists():
            print(f"Downloading {name}...")
            subprocess.run(["curl", "-L", "-o", str(zip_dir / name), url], check=True)

    expected_folders = [
        "incoherent_RGBchannels/train_incoherent_RGBChannels",
        "incoherent_RGBchannels/testRawData_incoherent_sameProtocol",
        "incoherent_RGBchannels/testRawData_incoherent_diffProtocol",
    ]
    raw_base = dataset_cfg.raw_dir
    if all((raw_base / d).exists() for d in expected_folders):
        print(f"Required data already exists in {raw_base}. Skipping extraction.")
        return

    zips = sorted(list(zip_dir.glob("*.zip")))
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


def get_jiang2018_dataloaders(
    batch_size: int = 128,
    num_workers: int = 4,
):
    """Get train and test dataloaders for Jiang2018."""
    # Ensure data is ready
    download_and_extract_jiang2018()
    raw_dir = Jiang2018Config().raw_dir / "incoherent_RGBchannels"

    val_transform = A.Compose(
        [
            A.ToFloat(max_value=255.0),
            ToTensorV2(),  # HWC → CHW
        ]
    )

    train_ds = Jiang2018Dataset(
        raw_dir,
        split="train",
        transform=val_transform,
    )
    test_same_ds = Jiang2018Dataset(
        raw_dir,
        split="test_same",
        transform=val_transform,
    )
    test_diff_ds = Jiang2018Dataset(
        raw_dir,
        split="test_diff",
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )

    combined_test = ConcatDataset([test_same_ds, test_diff_ds])
    test_loader = DataLoader(
        combined_test,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader
