import time
import sys
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.dataset.vsi_datamodule import VSIDataModule
"""Benchmark The VSI Datamodule"""

def measure_throughput(loader, batch_size, steps=50, warmup=10):
    iter_loader = iter(loader)

    for _ in range(warmup):
        next(iter_loader)

    start_time = time.time()
    for _ in range(steps):
        next(iter_loader)

    end_time = time.time()
    duration = end_time - start_time

    items_per_sec = (steps * batch_size) / duration
    return items_per_sec

def main():
    print("--- VSI Loader Scaling Benchmark (DataModule Version) ---")

    cpu_count = os.cpu_count() or 8

    worker_options = sorted(list(set([8, 16, cpu_count])))
    worker_options = [w for w in worker_options if w <= cpu_count + 4]

    batch_size_options = [64, 128, 256]

    print(f"Testing Workers: {worker_options}")
    print(f"Testing Batch Sizes: {batch_size_options}")
    print("-" * 60)

    print("Initial integrity check...")
    check_dm = VSIDataModule(dataset_name="ZStack_HE", batch_size=batch_size_options[0], num_workers=worker_options[0])
    check_dm.setup(stage="fit")
    first_batch = next(iter(check_dm.train_dataloader()))
    print(f"Sample Batch Shape - Img: {first_batch[0].shape}, Target: {first_batch[1].shape}")
    print("-" * 60)

    results = []
    best_throughput = 0.0
    best_config = (0, 0)

    print(
        f"{'Workers':<10} {'Batch Size':<12} {'Avg Throughput (img/s)':<25} {'Runs':<20}"
    )

    for num_workers in worker_options:
        for batch_size in batch_size_options:
            try:

                trial_speeds = []
                for i in range(3):

                    datamodule = VSIDataModule(
                        dataset_name="ZStack_HE",
                        batch_size=batch_size,
                        num_workers=num_workers,
                    )
                    datamodule.setup(stage="fit")

                    loader = datamodule.train_dataloader()

                    speed = measure_throughput(
                        loader, batch_size=batch_size, steps=40, warmup=10
                    )
                    trial_speeds.append(speed)

                avg_speed = sum(trial_speeds) / len(trial_speeds)
                runs_str = str([round(s, 1) for s in trial_speeds])

                print(
                    f"{num_workers:<10} {batch_size:<12} {avg_speed:<25.1f} {runs_str:<20}"
                )

                results.append((num_workers, batch_size, avg_speed))

                if avg_speed > best_throughput:
                    best_throughput = avg_speed
                    best_config = (num_workers, batch_size)

            except Exception as e:
                print(f"{num_workers:<10} {batch_size:<12} {'Failed':<25} ({e})")

    print("-" * 60)
    print("\n--- Optimization Result ---")
    print(f"Highest Avg Throughput: {best_throughput:.1f} items/sec")
    print(
        f"Optimal Configuration: Workers={best_config[0]}, Batch Size={best_config[1]}"
    )

if __name__ == "__main__":
    main()
