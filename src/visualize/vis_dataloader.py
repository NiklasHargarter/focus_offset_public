import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from src import config
from src.dataset.vsi_datamodule import VSIDataModule
from src.dataset.vsi_prep.preprocess import load_master_index
from src.dataset.vsi_dataset_lightning import VSIDatasetLightning
import matplotlib.pyplot as plt


def visualize_dataloader(
    dataset_name="ZStack_HE",
    num_samples=10,
    output_dir="visualizations/debug_dataloader",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize DataModule to get transforms
    dm = VSIDataModule(
        dataset_name=dataset_name,
        batch_size=1,
        num_workers=0,
        patch_size=224,
        stride=448,
        min_tissue_coverage=0.05,
    )

    # We need to manually setup to get the dataset
    master_index = load_master_index(dataset_name, 224)
    if master_index is None:
        print(f"Master index for {dataset_name} not found.")
        return

    # Just use the first slide for debugging
    slide_meta = master_index.file_registry[0]
    processed_index = dm._filter_index(master_index, [slide_meta.name])

    # Create two datasets: one with transforms, one without
    ds_raw = VSIDatasetLightning(index_data=processed_index, transform=None)
    ds_aug = VSIDatasetLightning(
        index_data=processed_index, transform=dm.train_transform
    )

    print(f"Total samples in debugging dataset: {len(ds_raw)}")

    # Randomly sample some indices
    indices = np.random.choice(
        len(ds_raw), min(num_samples, len(ds_raw)), replace=False
    )

    for i, idx in enumerate(indices):
        img_raw, offset, meta = ds_raw[idx]
        img_aug, _, _ = ds_aug[idx]

        # Convert tensors to numpy for plotting
        # Raw image (normalized to 0-1)
        raw_np = img_raw.permute(1, 2, 0).numpy()

        # Augmented image (needs de-normalization if it was normalized)
        # Note: dm.train_transform includes A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        aug_np = img_aug.permute(1, 2, 0).numpy()

        # De-normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        aug_np = (aug_np * std) + mean
        aug_np = np.clip(aug_np, 0, 1)

        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(raw_np)
        axes[0].set_title(f"Raw (Offset: {offset:.2f}um)")
        axes[0].axis("off")

        axes[1].imshow(aug_np)
        axes[1].set_title("Augmented + Normalized")
        axes[1].axis("off")

        plt.tight_layout()
        save_name = output_path / f"sample_{i}_idx_{idx}.png"
        plt.savefig(save_name)
        plt.close()
        print(f"Saved visualization to {save_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ZStack_HE")
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()
    visualize_dataloader(args.dataset, args.num_samples)
