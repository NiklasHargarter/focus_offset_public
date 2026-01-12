import time

from src.dataset.vsi_datamodule import HEHoldOutDataModule


def main():
    print("Initializing DataModule...")
    # Use default params from VSIDataModule or specify them here
    datamodule = HEHoldOutDataModule(
        dataset_name="ZStack_HE",
        batch_size=8,
        num_workers=4,
        patch_size=224,
        downscale_factor=16,
        min_tissue_coverage=0.05
    )

    print("Setting up dataset...")
    datamodule.setup(stage="fit")

    loader = datamodule.train_dataloader()

    if datamodule.train_dataset is None:
        print("Train dataset not found. Ensure preprocessing is complete.")
        return

    print(f"Dataset size: {len(datamodule.train_dataset)} samples (flattened Z)")
    print(
        f"DataLoader config: workers={datamodule.num_workers}, batch_size={datamodule.batch_size}"
    )

    start = time.time()
    max_steps = 100

    print(f"Starting integrity test for {max_steps} batches...")

    for i, batch in enumerate(loader):
        if i >= max_steps:
            break
        if i % 20 == 0:
            imgs, targets = batch
            print(
                f"Batch {i}/{max_steps} - Img: {imgs.shape}, Targets: {targets.shape}"
            )

    end = time.time()
    duration = end - start
    total_items = (i + 1) * datamodule.batch_size
    rate = total_items / duration

    print(f"\nProcessed {total_items} items in {duration:.2f}s")
    print(f"Throughput: {rate:.2f} items/sec")


if __name__ == "__main__":
    main()
