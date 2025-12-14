import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Enable Tensor Cores
torch.set_float32_matmul_precision("medium")

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import config  # noqa: E402
from src.dataset.vsi_dataset import VSIDataset  # noqa: E402
from src.models.factory import get_model, ModelArch  # noqa: E402


def benchmark_architecture(
    arch, dataset, device, batch_size=config.BATCH_SIZE, num_steps=500
):
    print(f"\nBenchmarking architecture: {arch.value}")

    # Load Model
    try:
        model = get_model(arch).to(device)
    except Exception as e:
        print(f"Failed to load model {arch}: {e}")
        return None

    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.MSELoss()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=True,
        persistent_workers=True,
    )

    # Warmup
    print("  Warming up...")
    iter_loader = iter(loader)
    for _ in range(100):
        images, targets = next(iter_loader)

        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = model(images)
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Benchmark
    print(f"  Running {num_steps} steps...")
    torch.cuda.synchronize() if device.type == "cuda" else None
    start_time = time.time()

    for _ in range(num_steps):
        images, targets = next(iter_loader)

        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = model(images)
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize() if device.type == "cuda" else None
    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_step = total_time / num_steps
    throughput = (batch_size * num_steps) / total_time

    # Estimate full epoch time
    steps_per_epoch = len(dataset) / batch_size
    estimated_epoch_time = avg_time_per_step * steps_per_epoch

    print(f"  Average batch time: {avg_time_per_step * 1000:.2f} ms")
    print(f"  Throughput: {throughput:.2f} samples/sec")
    print(f"  Estimated Epoch Time: {estimated_epoch_time / 60:.2f} minutes")

    return {
        "arch": arch.value,
        "batch_time_ms": avg_time_per_step * 1000,
        "throughput": throughput,
        "epoch_time_min": estimated_epoch_time / 60,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset once
    index_path = config.get_index_path("train")
    if not index_path.exists():
        print("Train index not found. Run preprocess first.")
        return

    print("Loading dataset...")
    dataset = VSIDataset(mode="train")
    print(f"Dataset Size: {len(dataset)}")

    results = []

    for arch in ModelArch:
        res = benchmark_architecture(arch, dataset, device)
        if res:
            results.append(res)

    print("\n\n=== BENCHMARK SUMMARY ===")
    print(
        f"{'Architecture':<25} | {'Throughput (img/s)':<20} | {'Est. Epoch Time (min)':<25}"
    )
    print("-" * 75)
    for r in results:
        print(
            f"{r['arch']:<25} | {r['throughput']:<20.2f} | {r['epoch_time_min']:<25.2f}"
        )


if __name__ == "__main__":
    main()
