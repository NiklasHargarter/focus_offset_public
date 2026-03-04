import matplotlib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from shared_datasets.jiang2018 import Jiang2018Dataset


def denormalize(tensor):
    # Images are loaded as float [0, 1] in the dataloader (ToFloat(255))
    return tensor.permute(1, 2, 0).numpy()


def get_unbinned_patch(dataset, idx):
    """Load the raw image without binning and extract a matching crop."""
    print(f"Loading unbinned patch for index {idx}...")
    sample = dataset.samples[idx]
    path = sample["path"]

    # Load raw
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get coordinates of the tile directly from x and y metadata
    metadata = dataset[idx]["metadata"]
    x_binned = metadata["x"]
    y_binned = metadata["y"]

    # In unbinned space (2048x2448), this corresponds to:
    y_orig = y_binned * 2
    x_orig = x_binned * 2

    # Crop 224x224 from the unbinned image at this location
    # This effectively shows a "zoomed in" view compared to the binned tile
    patch = img[y_orig : y_orig + 224, x_orig : x_orig + 224]

    return patch


def main():
    print("Script started.")
    sns.set_theme(style="white")

    print("Loading datasets...")
    from shared_datasets.jiang2018.config import Jiang2018Config
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    raw_dir = Jiang2018Config().raw_dir / "incoherent_RGBchannels"
    val_transform = A.Compose([A.ToFloat(max_value=255.0), ToTensorV2()])

    train_ds = Jiang2018Dataset(raw_dir, split="train", transform=val_transform)
    test_ds = Jiang2018Dataset(raw_dir, split="test_same", transform=val_transform)
    print(f"Test Same dataset loaded. {len(test_ds)} samples.")

    # Randomly select a few samples from Test Same
    n_samples = 12
    print(f"Selecting {n_samples} random samples...")
    test_indices = np.random.choice(len(test_ds), n_samples, replace=False)
    train_indices = np.random.choice(len(train_ds), n_samples, replace=False)

    # Sort indices by defocus distance for easier comparison
    print("Sorting samples by defocus distance...")

    def get_defocus(dataset, idx):
        return dataset[idx]["metadata"]["defocus_nm"]

    test_indices = sorted(test_indices, key=lambda idx: get_defocus(test_ds, idx))
    train_indices = sorted(train_indices, key=lambda idx: get_defocus(train_ds, idx))

    print(f"Selected indices (sorted): Test={test_indices}, Train={train_indices}")

    rows = [
        "Train (Reference)\n(Pre-processed by Authors)",
        "Test Same (With Binning)\n(Our Pipeline: 0.5x resize)",
        "Test Same (No Binning)\n(Raw 224x224 crop)",
    ]

    print("Creating figure...")
    fig, axes = plt.subplots(len(rows), n_samples, figsize=(24, 3 * len(rows)))

    print("Plotting Train samples...")
    # Row 1: Train
    for i, idx in enumerate(train_indices):
        print(f"  Plotting Train sample {i + 1}/{n_samples} (idx={idx})...")
        sample = train_ds[idx]
        img = denormalize(sample["image"])
        ax = axes[0, i]
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{sample['metadata']['defocus_nm']}nm")

    print("Plotting Test samples...")
    # Row 2 & 3: Test Same (Binned vs Unbinned)
    for i, idx in enumerate(test_indices):
        print(f"  Plotting Test sample {i + 1}/{n_samples} (idx={idx})...")
        # Binned (Standard Pipeline)
        sample = test_ds[idx]
        img_binned = denormalize(sample["image"])
        defocus = sample["metadata"]["defocus_nm"]

        ax1 = axes[1, i]
        ax1.imshow(img_binned)
        ax1.axis("off")
        ax1.set_title(f"{defocus}nm")

        # Unbinned (Raw Crop)
        img_unbinned = get_unbinned_patch(test_ds, idx)

        ax2 = axes[2, i]
        ax2.imshow(img_unbinned)
        ax2.axis("off")
        ax2.set_title(f"{defocus}nm")

    # Add row labels
    for row_idx, label in enumerate(rows):
        axes[row_idx, 0].text(
            -0.2,
            0.5,
            label,
            transform=axes[row_idx, 0].transAxes,
            va="center",
            ha="right",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    output_path = "jiang_binning_check.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
