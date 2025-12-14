import time
import torch
from torch.utils.data import DataLoader
import lightning as L
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


class BenchmarkCallback(L.Callback):
    def __init__(self, warmup_steps=100, total_steps=600):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.start_time = 0.0
        self.end_time = 0.0
        self.processed_batches = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.processed_batches == self.warmup_steps:
            print("  [Benchmark] Warmup complete. Starting measurement...")
            self.start_time = time.time()

        self.processed_batches += 1

        if self.processed_batches % 100 == 0:
            print(
                f"  [Benchmark] Progress: {self.processed_batches}/{self.total_steps} batches"
            )

    def on_train_end(self, trainer, pl_module):
        self.end_time = time.time()

    def get_throughput(self, batch_size):
        measured_batches = self.processed_batches - self.warmup_steps
        if measured_batches <= 0:
            return 0.0
        duration = self.end_time - self.start_time
        return (measured_batches * batch_size) / duration


def benchmark_architecture(arch: ModelArch, train_loader, val_loader):
    print(f"\nBenchmarking architecture: {arch.value}")

    model = FocusOffsetRegressor(arch_name=arch, learning_rate=config.LEARNING_RATE)

    WARMUP_STEPS = 100
    MEASURE_STEPS = 500
    TOTAL_STEPS = WARMUP_STEPS + MEASURE_STEPS

    benchmark_callback = BenchmarkCallback(
        warmup_steps=WARMUP_STEPS, total_steps=TOTAL_STEPS
    )

    benchmark_trainer = L.Trainer(
        max_epochs=1,
        limit_train_batches=TOTAL_STEPS,
        limit_val_batches=0,
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[benchmark_callback],
    )

    print(f"  Running benchmark ({WARMUP_STEPS} warmup + {MEASURE_STEPS} measure)...")

    benchmark_trainer.fit(model, train_loader)

    throughput = benchmark_callback.get_throughput(config.BATCH_SIZE)
    total_time = benchmark_callback.end_time - benchmark_callback.start_time

    # Estimate full epoch time
    dataset_size = len(train_loader.dataset)
    steps_per_epoch = dataset_size / config.BATCH_SIZE
    # Time per batch
    time_per_batch = total_time / MEASURE_STEPS
    epoch_time_seconds = steps_per_epoch * time_per_batch
    epoch_time_min = epoch_time_seconds / 60

    print(f"  Measurement Time: {total_time:.2f}s")
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
        num_workers=config.NUM_WORKERS,
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
