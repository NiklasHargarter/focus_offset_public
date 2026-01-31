import json
import traceback
import time
import torch
import lightning as L

from src.dataset.vsi_datamodule import VSIDataModule
from src.models.lightning_module import FocusOffsetRegressor
from src.models.architectures import ConvNeXtV2FocusRegressor

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


def benchmark_architecture(
    arch_name: str, train_loader, val_loader, dataset_size, batch_size
):
    print(f"\nBenchmarking architecture: {arch_name}")

    if arch_name == "convnext_v2_tiny_multimodal":
        backbone = ConvNeXtV2FocusRegressor(pretrained=False, use_transforms=True)
    elif arch_name == "convnext_v2_tiny_baseline":
        backbone = ConvNeXtV2FocusRegressor(pretrained=False, use_transforms=False)
    else:
        print(f"Unknown architecture {arch_name}, skipping.")
        return None

    model = FocusOffsetRegressor(backbone=backbone)

    WARMUP_STEPS = 50
    MEASURE_STEPS = 100
    TOTAL_STEPS = WARMUP_STEPS + MEASURE_STEPS

    benchmark_callback = BenchmarkCallback(
        warmup_steps=WARMUP_STEPS, total_steps=TOTAL_STEPS
    )

    benchmark_trainer = L.Trainer(
        max_epochs=1,
        limit_train_batches=TOTAL_STEPS,
        limit_val_batches=1,
        accelerator="auto",
        devices=1,
        precision="bf16-mixed",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[benchmark_callback],
    )

    print(f"  Running benchmark ({WARMUP_STEPS} warmup + {MEASURE_STEPS} measure)...")

    benchmark_trainer.fit(model, train_loader, val_loader)

    throughput = benchmark_callback.get_throughput(batch_size)
    total_time = benchmark_callback.end_time - benchmark_callback.start_time

    steps_per_epoch = dataset_size / batch_size

    time_per_batch = total_time / MEASURE_STEPS
    epoch_time_seconds = steps_per_epoch * time_per_batch
    epoch_time_min = epoch_time_seconds / 60

    print(f"  Measurement Time: {total_time:.2f}s")
    print(f"  Throughput: {throughput:.2f} samples/sec")
    print(f"  Est. Epoch Time: {epoch_time_min:.2f} min")

    return {
        "arch": arch_name,
        "throughput": throughput,
        "epoch_time_min": epoch_time_min,
    }


def main():
    L.seed_everything(42)

    print("Initializing DataModule...")
    # Updated with optimized parameters from dataloader benchmark
    datamodule = VSIDataModule(
        dataset_name="ZStack_HE",
        batch_size=128,
        num_workers=4, # Updated to 4 based on autotune
        patch_size=224,
        stride=448,
        min_tissue_coverage=0.05,
        downsample_factor=2,
    )
    datamodule.setup(stage="fit")

    if datamodule.train_dataset is None:
        print("Train dataset not found. Ensure preprocessing is complete.")
        return

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    dataset_size = len(datamodule.train_dataset)
    batch_size = datamodule.batch_size

    results = []

    arch_list = [
        "convnext_v2_tiny_multimodal",
        "convnext_v2_tiny_baseline",
    ]

    for arch_name in arch_list:
        try:
            res = benchmark_architecture(
                arch_name, train_loader, val_loader, dataset_size, batch_size
            )
            if res:
                results.append(res)
        except Exception as e:
            print(f"Skipping {arch_name} due to error: {e}")
            traceback.print_exc()

    print("\n\n=== LIGHTNING BENCHMARK SUMMARY (AMP 16) ===")
    print(
        f"{'Architecture':<30} | {'Throughput (img/s)':<20} | {'Est. Epoch Time (min)':<25}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['arch']:<30} | {r['throughput']:<20.2f} | {r['epoch_time_min']:<25.2f}"
        )

    # Save results to JSON
    output_path = "benchmark_models.json"
    with open(output_path, "w") as f:
        json.dump(
            {"results": results, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
            f,
            indent=4,
        )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
