
import os
import time
import torch
import lightning as L
from typing import List, Tuple
from src.dataset.vsi_datamodule import VSIDataModule
from src.models.lightning_module import FocusOffsetRegressor
from src.models.architectures import EfficientNetFocusRegressor
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
            self.throughput = (measured_steps * trainer.train_dataloader.batch_size) / duration

def run_trial(batch_size: int, num_workers: int) -> float:
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
            prefetch_factor=2 # Conservative default
        )
        datamodule.setup(stage="fit")
        
        # Use EfficientNet B0 as a standard baseline for tuning
        backbone = EfficientNetFocusRegressor(version="b0", pretrained=False)
        model = FocusOffsetRegressor(backbone=backbone)
        
        callback = ThroughputCallback(warmup_steps=10, measure_steps=40)
        
        trainer = L.Trainer(
            max_steps=50,
            limit_val_batches=0, # Skip validation for speed
            accelerator="gpu",
            devices=1,
            precision="bf16-mixed",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[callback]
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

def main():
    print("="*60)
    print("      AUTOMATIC TRAINING HYPERPARAMETER TUNER")
    print("="*60)
    
    cpu_count = os.cpu_count() or 8
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    
    print(f"System: {cpu_count} CPUs | GPU: {gpu_name}")
    
    # Candidate values
    batch_sizes = [128, 256, 512]
    worker_counts = [8, 12, 16]

    print(f"Batch sizes to test: {batch_sizes}")
    print(f"Worker counts to test: {worker_counts}")
    print("-" * 60)
    
    results = []
    best_throughput = 0.0
    best_config = (0, 0)
    
    print(f"{'Workers':<10} | {'Batch Size':<12} | {'Throughput (img/s)':<20}")
    print("-" * 60)
    
    for workers in worker_counts:
        oom_hit = False
        for bs in batch_sizes:
            if oom_hit:
                break
            
            print(f"  > Testing Config: Workers={workers}, Batch Size={bs}...", flush=True)    
            throughput = run_trial(bs, workers)
            
            if throughput < 0:
                print(f"    [OOM] Batch size {bs} failed.")
                oom_hit = True
                continue
            
            if throughput == 0:
                print(f"    [Error] Configuration failed.")
                continue
                
            print(f"    Result: {throughput:.2f} img/s")
            results.append((workers, bs, throughput))
            
            if throughput > best_throughput:
                best_throughput = throughput
                best_config = (workers, bs)

    print("-" * 60)
    print("\n--- OPTIMAL SETTINGS FOR THIS MACHINE ---")
    if best_throughput > 0:
        print(f"Optimal Batch Size:  {best_config[1]}")
        print(f"Optimal Worker Count: {best_config[0]}")
        print(f"Max Throughput:       {best_throughput:.2f} img/s")
    else:
        print("No successful trials completed.")
    print("-" * 60)

if __name__ == "__main__":
    main()
