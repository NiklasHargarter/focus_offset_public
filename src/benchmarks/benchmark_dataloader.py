import time


from torch.utils.data import DataLoader
from src import config
from src.dataset.vsi_dataset import VSIDataset


def measure_throughput(loader, steps=50, warmup=100):
    start_time = None
    count = 0

    iter_loader = iter(loader)

    # Warmup
    for _ in range(warmup):
        try:
            next(iter_loader)
        except StopIteration:
            break

    start_time = time.time()
    try:
        for _ in range(steps):
            next(iter_loader)
            count += 1
    except StopIteration:
        pass

    end_time = time.time()
    duration = end_time - start_time

    if count == 0:
        return 0.0

    items_per_sec = (count * loader.batch_size) / duration
    return items_per_sec


def main():
    print("--- VSI Loader Scaling Benchmark ---")

    # Check checks strictly for visualization purposes here, though Dataset handles it too.
    if not config.get_index_path("train").exists():
        print("Index not found. Run preprocess first.")
        return

    ds = VSIDataset(mode="train")

    # Scenario 1: Single Process (num_workers=0)
    print("\nScenario 1: Single Process (Workers=0)")
    loader1 = DataLoader(ds, batch_size=16, num_workers=0, shuffle=True)
    speed1 = measure_throughput(loader1)
    print(f"-> Speed: {speed1:.1f} items/sec")

    ds = VSIDataset(mode="train")
    # Scenario 2: Multi-Process (Workers=8)
    print("\nScenario 2: Multi-Process (Workers=8)")
    # Note: persistent_workers=True is crucial for efficiency
    loader2 = DataLoader(
        ds, batch_size=16, num_workers=8, shuffle=True, persistent_workers=True
    )
    speed2 = measure_throughput(loader2)
    print(f"-> Speed: {speed2:.1f} items/sec")

    print("\n--- Summary ---")
    print(f"Workers=0: {speed1:.1f} items/sec")
    print(f"Workers=2: {speed2:.1f} items/sec")

    if speed1 > 0:
        print(f"Speedup: {speed2 / speed1:.1f}x")


if __name__ == "__main__":
    main()
