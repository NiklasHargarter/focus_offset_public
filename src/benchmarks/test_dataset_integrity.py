import time
import os

from torch.utils.data import DataLoader
from src.dataset.vsi_dataset import VSIDataset
from src import config


def main():
    workers = os.cpu_count()
    batch_size = config.BATCH_SIZE
    persistent = True
    pin_memory = True
    prefetch = 4

    # Check for index existence (optional visual check, dataset handles it too)
    if not config.get_index_path("train").exists():
        print("Index not found. Run preprocess.py first.")
        return

    print("Loading dataset...")
    dataset = VSIDataset(mode="train")
    print(f"Dataset size: {len(dataset)} samples (flattened Z)")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=True,
        persistent_workers=persistent,
        pin_memory=pin_memory,
        prefetch_factor=prefetch if workers > 0 else None,
    )

    print(
        f"Starting DataLoader test: Workers={workers}, Persistent={persistent}, Pin={pin_memory}, Prefetch={prefetch}"
    )
    start = time.time()

    max_steps = 200  # Increased steps for stable measurement
    for i, batch in enumerate(loader):
        if i >= max_steps:
            break
        if i % 20 == 0:
            if isinstance(batch, (list, tuple)):
                imgs, targets = batch
                print(f"Batch {i}/{max_steps} - Img: {imgs.shape}")
            else:
                pass

    end = time.time()
    duration = end - start
    total_items = max_steps * batch_size
    rate = total_items / duration

    print(f"\nProcessed {total_items} items in {duration:.2f}s")
    print(f"Throughput: {rate:.2f} items/sec")


if __name__ == "__main__":
    main()
