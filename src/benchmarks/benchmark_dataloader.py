import time
import sys
from pathlib import Path
import os

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from torch.utils.data import DataLoader
import config
from src.dataset.vsi_dataset import VSIDataset


def measure_throughput(loader, steps=50, warmup=10):
    iter_loader = iter(loader)

    # Warmup
    for _ in range(warmup):
        next(iter_loader)

    start_time = time.time()
    for _ in range(steps):
        next(iter_loader)

    end_time = time.time()
    duration = end_time - start_time

    items_per_sec = (steps * loader.batch_size) / duration
    return items_per_sec


def main():
    print("--- VSI Loader Scaling Benchmark ---")

    if not config.get_index_path("train").exists():
        print("Index not found. Run preprocess first.")
        return

    cpu_count = os.cpu_count() or 8
    # Define search grid
    worker_options = sorted(list(set([8, 16, cpu_count])))
    # Limit workers to reasonable max (e.g. cpu_count * 1.5 rounded or just static list)
    worker_options = [w for w in worker_options if w <= cpu_count + 4]

    batch_size_options = [16, 32, 64, 128, 256]

    print(f"Testing Workers: {worker_options}")
    print(f"Testing Batch Sizes: {batch_size_options}")
    print("-" * 60)

    results = []
    best_throughput = 0.0
    best_config = (0, 0)

    # Header
    print(
        f"{'Workers':<10} {'Batch Size':<12} {'Avg Throughput (img/s)':<25} {'Runs':<20}"
    )

    for num_workers in worker_options:
        for batch_size in batch_size_options:
            try:
                # Run 3 trials
                trial_speeds = []
                for i in range(3):
                    # Re-initialize dataset for each run to avoid side-effects
                    ds = VSIDataset(mode="train")

                    # persistent_workers must be False if num_workers is 0
                    use_persistent = num_workers > 0

                    loader = DataLoader(
                        ds,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=True,
                        persistent_workers=use_persistent,
                        pin_memory=True,
                    )

                    # Using fewer steps for larger batches to save time, or keep constant?
                    # throughput measurement should be robust. Let's keep 50 steps.
                    speed = measure_throughput(loader, steps=50, warmup=20)
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
