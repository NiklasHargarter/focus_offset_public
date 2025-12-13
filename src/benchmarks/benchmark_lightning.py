import time
import torch
from torch.utils.data import DataLoader
import lightning as L
import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import config
from src.dataset.vsi_dataset import VSIDataset
from src.models.lightning_module import FocusOffsetRegressor
from src.models.factory import ModelArch

# Enable Tensor Cores globally if not handled by Trainer
torch.set_float32_matmul_precision("medium")


def benchmark_architecture(arch: ModelArch, train_loader, val_loader):
    print(f"\nBenchmarking architecture: {arch.value}")

    model = FocusOffsetRegressor(arch_name=arch, learning_rate=config.LEARNING_RATE)

    # 1. Warmup Run (small number of batches to compile graph/allocate memory)
    print("  Warming up...")
    warmup_trainer = L.Trainer(
        max_epochs=1,
        limit_train_batches=10,
        limit_val_batches=0,  # No validation during benchmark
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    warmup_trainer.fit(model, train_loader)

    # 2. Benchmark Run
    # Run for 100 batches to get a stable throughput estimate
    NUM_BATCHES = 100

    benchmark_trainer = L.Trainer(
        max_epochs=1,
        limit_train_batches=NUM_BATCHES,
        limit_val_batches=0,
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        log_every_n_steps=1000,  # Suppress logs
    )

    print(f"  Running benchmark ({NUM_BATCHES} batches)...")
    torch.cuda.synchronize()
    start_time = time.time()

    benchmark_trainer.fit(model, train_loader)

    torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    total_samples = NUM_BATCHES * config.BATCH_SIZE
    throughput = total_samples / total_time

    # Estimate full epoch time
    dataset_size = len(train_loader.dataset)
    steps_per_epoch = dataset_size / config.BATCH_SIZE
    epoch_time_seconds = steps_per_epoch * (total_time / NUM_BATCHES)
    epoch_time_min = epoch_time_seconds / 60

    print(f"  Time: {total_time:.2f}s")
    print(f"  Throughput: {throughput:.2f} samples/sec")
    print(f"  Est. Epoch Time: {epoch_time_min:.2f} min")

    return {
        "arch": arch.value,
        "throughput": throughput,
        "epoch_time_min": epoch_time_min,
    }


def main():
    L.seed_everything(42)

    # Load Dataset Once
    index_path = config.get_index_path("train")
    if not index_path.exists():
        print("Train index not found. Run preprocess first.")
        return

    print("Loading dataset...")
    full_dataset = VSIDataset(mode="train")

    train_loader = DataLoader(
        full_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count(),
        persistent_workers=True,
        pin_memory=True,
    )

    val_loader = DataLoader(full_dataset, batch_size=config.BATCH_SIZE)

    results = []

    for arch in ModelArch:
        try:
            res = benchmark_architecture(arch, train_loader, val_loader)
            results.append(res)
        except Exception as e:
            print(f"Skipping {arch.value} due to error: {e}")

    print("\n\n=== LIGHTNING BENCHMARK SUMMARY (BF16) ===")
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
