import os
import time
import torch
import lightning as L
from typing import List, Tuple
from src.dataset.vsi_datamodule import VSIDataModule
from src.models.lightning_module import FocusOffsetRegressor
import warnings

warnings.filterwarnings("ignore")

# Force matmul precision for RTX 40 series
torch.set_float32_matmul_precision("medium")


class ThroughputCallback(L.Callback):
    def __init__(self, warmup_steps: int = 20, measure_steps: int = 50):
        self.warmup_steps = warmup_steps
        self.measure_steps = measure_steps
        self.total_steps = warmup_steps + measure_steps
        self.start_time = 0.0
        self.end_time = 0.0
        self.step_count = 0
        self.throughput = 0.0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.step_count == self.warmup_steps:
            self.start_time = time.time()
        self.step_count += 1

    def on_train_end(self, trainer, pl_module):
        self.end_time = time.time()
        measured_steps = self.step_count - self.warmup_steps
        if measured_steps > 0 and self.start_time > 0:
            duration = self.end_time - self.start_time
            self.throughput = (
                measured_steps * trainer.train_dataloader.batch_size
            ) / duration


def run_trial(batch_size: int, num_workers: int, use_transforms: bool) -> float:
    """Runs a short training trial and returns throughput (samples/sec)."""
    try:
        datamodule = VSIDataModule(
            dataset_name="ZStack_HE",
            batch_size=batch_size,
            num_workers=num_workers,
            patch_size=224,
            stride=448,
            min_tissue_coverage=0.05,
            downsample_factor=2,
            prefetch_factor=2,
        )
        datamodule.setup(stage="fit")

        # Use simple model for tuning
        model = FocusOffsetRegressor(
            pretrained=False, use_transforms=use_transforms
        )

        callback = ThroughputCallback(warmup_steps=10, measure_steps=40)

        trainer = L.Trainer(
            max_steps=50,
            limit_val_batches=0,  # Skip validation for speed
            accelerator="gpu",
            devices=1,
            precision="bf16-mixed",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[callback],
        )

        print(f"      - Starting trainer.fit...", flush=True)
        trainer.fit(model, datamodule=datamodule)
        return callback.throughput

    except torch.cuda.OutOfMemoryError:
        print(f"  [OOM] Batch size {batch_size} is too large.")
        return -1.0
    except Exception as e:
        print(f"  [Error] {e}")
        return 0.0



def find_max_batch_size(use_transforms: bool) -> int:
    """Finds the maximum batch size that fits in memory using a fixed worker count."""
    print(f"\n   --- STAGE 1: Find Max Batch Size ---")
    
    # Powers of 2
    candidate_sizes = [16, 32, 64, 128, 256, 512]
    # Use a low, safe worker count for this test to minimize overhead
    test_workers = 4 
    
    max_safe_bs = 0
    
    for bs in candidate_sizes:
        print(f"  > Testing Batch Size: {bs}...", flush=True)
        throughput = run_trial(bs, num_workers=test_workers, use_transforms=use_transforms)
        
        if throughput < 0: # OOM
            print(f"    [OOM] Batch Size {bs} is too large.")
            break
        elif throughput == 0: # Error
            print(f"    [Error] Failed at Batch Size {bs}.")
            break
        else:
            print(f"    [Success] {throughput:.2f} img/s")
            max_safe_bs = bs
            
    print(f"   >>> Max Safe Batch Size: {max_safe_bs}")
    return max_safe_bs

def find_ideal_workers(batch_size: int, use_transforms: bool) -> Tuple[int, float]:
    """Finds the optimal worker count for a given batch size."""
    print(f"\n   --- STAGE 2: Find Ideal Worker Count (BS={batch_size}) ---")
    
    candidate_workers = [4, 8, 12, 16]
    
    best_workers = 0
    best_throughput = 0.0
    
    print(f"{'Workers':<10} | {'Throughput (img/s)':<20}")
    print("-" * 40)
    
    for workers in candidate_workers:
        print(f"  > Testing Workers: {workers}...", flush=True)
        throughput = run_trial(batch_size, num_workers=workers, use_transforms=use_transforms)
        
        if throughput <= 0:
            print(f"    [Failed] Workers={workers}")
            continue
            
        print(f"    Result: {throughput:.2f} img/s")
        
        if throughput > best_throughput:
            best_throughput = throughput
            best_workers = workers
            
    print("-" * 40)
    return best_workers, best_throughput

def main():
    print("=" * 60)
    print("      AUTOMATIC TRAINING HYPERPARAMETER TUNER (2-STAGE V2)")
    print("=" * 60)

    cpu_count = os.cpu_count() or 8
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"

    print(f"System: {cpu_count} CPUs | GPU: {gpu_name}")
    
    model_configs = [("Multimodal", True), ("RGB Only", False)]

    for mode_name, use_transforms in model_configs:
        print("\n" + "=" * 60)
        print(f"   BENCHMARKING: {mode_name}")
        print("=" * 60)
        
        # 1. Find Max Batch Size
        max_bs = find_max_batch_size(use_transforms)
        
        if max_bs == 0:
            print("Could not find any working batch size configuration.")
            continue
            
        # 2. Find Ideal Workers
        best_workers, max_throughput = find_ideal_workers(max_bs, use_transforms)
        
        print("\n" + "-" * 60)
        print(f"--- OPTIMAL SETTINGS FOR {mode_name.upper()} ---")
        print(f"Optimal Batch Size:   {max_bs}")
        print(f"Optimal Worker Count: {best_workers}")
        print(f"Max Throughput:       {max_throughput:.2f} img/s")
        print("-" * 60)


if __name__ == "__main__":
    main()
