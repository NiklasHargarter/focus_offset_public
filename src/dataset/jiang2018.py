import re
import cv2
import numpy as np
import torch
import subprocess
import pickle
import lightning as L
from pathlib import Path
from typing import Optional, Tuple
from torch.utils.data import Dataset, DataLoader

import config

# Figshare download links for Jiang 2018
LINKS = {
    # "Data_domain_part1.zip": "https://ndownloader.figshare.com/files/10616956",
    # "Data_domain_part2.zip": "https://ndownloader.figshare.com/files/10616962",
    "Data_channel.zip": "https://ndownloader.figshare.com/files/10616965",
}


def _compute_brenner(image: np.ndarray) -> int:
    """Focus score for ground truth detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.int32)
    return int(np.sum((gray - np.roll(gray, -2, axis=1)) ** 2))


class Jiang2018Dataset(Dataset):
    """Minimal dataset for pre-tiled Jiang 2018 segments with caching."""

    def __init__(self, root_dir: Path, force_recompute: bool = False):
        self.samples = []
        cache_path = config.CACHE_DIR / "jiang2018_index_v5.pkl"

        if cache_path.exists() and not force_recompute:
            print(f"Loading Jiang 2018 dataset index from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                self.samples = pickle.load(f)
            print(f"Jiang2018Dataset: Loaded {len(self.samples)} samples from cache.")
            return

        print(
            f"Scanning {root_dir} and computing focus scores (this may take a few minutes)..."
        )
        all_leaf_dirs = {p.parent for p in root_dir.rglob("Seg*.jpg")}
        # Only include the requested domains/channels
        leaf_dirs = [
            p for p in all_leaf_dirs if "incoherent_RGBchannels" in p.as_posix()
        ]

        for folder in sorted(leaf_dirs):
            jpgs = list(folder.glob("Seg*.jpg"))
            if not jpgs:
                continue

            # Group JPGs by Segment (e.g., "Seg1", "Seg2", ...)
            seg_groups = {}
            for f in jpgs:
                seg_match = re.match(r"(Seg\d+)_", f.name)
                if seg_match:
                    seg_id = seg_match.group(1)
                    if seg_id not in seg_groups:
                        seg_groups[seg_id] = []
                    seg_groups[seg_id].append(f)

            for seg_id, group_files in seg_groups.items():
                best_score, best_abs_nm, stack = -1, 0, []
                for f in group_files:
                    match = re.search(r"defocus(-?\d+)", f.name)
                    if match:
                        abs_nm = float(match.group(1))
                        # Read and compute score
                        img = cv2.imread(str(f))
                        if img is None:
                            continue
                        score = _compute_brenner(img)
                        if score > best_score:
                            best_score, best_abs_nm = score, abs_nm
                        stack.append((f, abs_nm, score))

                for f, abs_nm, score in stack:
                    self.samples.append(
                        {
                            "path": f,
                            "offset": (abs_nm - best_abs_nm) / 1000.0,
                            "abs_nm": abs_nm,
                            "score": score,
                        }
                    )

        # Save to cache
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(self.samples, f)

        print(
            f"Jiang2018Dataset: Discovery complete. Saved {len(self.samples)} samples across {len(leaf_dirs)} folders to cache."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        img = cv2.imread(str(sample["path"]))

        # Convert BGR to RGB for the model.
        # For grayscale images (where B=G=R), this remains identical grayscale.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return tensor, torch.tensor(sample["offset"], dtype=torch.float32)


class Jiang2018DataModule(L.LightningDataModule):
    """Self-contained DataModule with auto-download and extraction."""

    def __init__(
        self,
        batch_size: int = config.BATCH_SIZE,
        num_workers: int = config.NUM_WORKERS,
        force_recompute: bool = False,
    ):
        super().__init__()
        self.batch_size, self.num_workers = batch_size, num_workers
        self.force_recompute = force_recompute
        self.dataset_name = "Jiang2018"
        self.zip_dir = config.get_vsi_zip_dir(self.dataset_name)
        self.raw_dir = config.get_vsi_raw_dir(self.dataset_name)
        self.test_ds = None

    def prepare_data(self):
        self.zip_dir.mkdir(parents=True, exist_ok=True)
        for name, url in LINKS.items():
            if not (self.zip_dir / name).exists():
                print(f"Downloading {name}...")
                subprocess.run(
                    ["curl", "-L", "-o", str(self.zip_dir / name), url], check=True
                )

        # Check if we need to extract (check for specific folders)
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

    def setup(self, stage: Optional[str] = None):
        self.test_ds = Jiang2018Dataset(
            self.raw_dir, force_recompute=self.force_recompute
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
