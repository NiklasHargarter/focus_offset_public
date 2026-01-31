import time
import torch
import os
from src.dataset.vsi_datamodule import VSIDataModule
from src.models.lightning_module import FocusOffsetRegressor
from src.models.architectures import ConvNeXtV2FocusRegressor

"""
🔬 Hyper-Scientific Data Bottleneck Diagnostic (Stable Version)
Compares:
1. CPU/Disk Loading (Raw I/O)
2. Synthetic GPU Processing (Model Peak Speed)
3. Full Combined Pipeline (Real Training)
Uses BF16 and no_grad for throughput measurements.
"""


def main():
    BS = 256
    WORKERS = os.cpu_count()
    STEPS = 50
    torch.set_float32_matmul_precision("medium")
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    print(f"--- Hyper-Scientific Diagnostic (BS={BS}, Workers={WORKERS}) ---")

    # Initialize Data
    dm = VSIDataModule(
        dataset_name="ZStack_HE",
        batch_size=BS,
        num_workers=WORKERS,
        patch_size=224,
        stride=448,
        min_tissue_coverage=0.05,
        downsample_factor=2,
        prefetch_factor=4,
    )
    dm.setup(stage="fit")
    loader = dm.train_dataloader()

    # 1. PURE DATA SPEED (CPU/Disk)
    print("\nPhase 1: Pure Data Loading (CPU Only)")
    it = iter(loader)
    for _ in range(10):
        next(it)

    start = time.time()
    for _ in range(STEPS):
        batch = next(it)
        _ = batch[0].shape
    end = time.time()
    data_throughput = (STEPS * BS) / (end - start)
    print(f">> Raw I/O Throughput: {data_throughput:.2f} img/s")

    # 2. PURE GPU SPEED (Model Only)
    print("\nPhase 2: Synthetic GPU Speed (Model Peak)")
    backbone = ConvNeXtV2FocusRegressor().cuda()
    model = FocusOffsetRegressor(backbone=backbone).cuda()
    model.backbone.to(memory_format=torch.channels_last)

    # Use BF16 to match actual run
    dummy_imgs = (
        torch.randn(BS, 3, 224, 224)
        .cuda()
        .to(memory_format=torch.channels_last)
        .bfloat16()
    )

    print("  Measuring peak model throughput (BF16, no_grad)...")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            for _ in range(20):
                _ = model(dummy_imgs)

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(STEPS):
                _ = model(dummy_imgs)
            torch.cuda.synchronize()
            end = time.time()

    gpu_throughput = (STEPS * BS) / (end - start)
    print(f">> Peak GPU Throughput: {gpu_throughput:.2f} img/s")

    # 3. REAL SPEED (Combined)
    print("\nPhase 3: Real Combined Pipeline (BF16)")
    it = iter(loader)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                batch = next(it)
                gpu_img = (
                    batch[0].cuda().to(memory_format=torch.channels_last).bfloat16()
                )
                _ = model(gpu_img)

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(STEPS):
                batch = next(it)
                gpu_img = (
                    batch[0].cuda().to(memory_format=torch.channels_last).bfloat16()
                )
                _ = model(gpu_img)
            torch.cuda.synchronize()
            end = time.time()

    real_throughput = (STEPS * BS) / (end - start)
    print(f">> Real-World Throughput: {real_throughput:.2f} img/s")

    print("\n--- Final Diagnostic Result ---")
    print(f"Max I/O Speed:  {data_throughput:>10.2f} img/s")
    print(f"Max GPU Speed:  {gpu_throughput:>10.2f} img/s")
    print(f"Realistic Speed:{real_throughput:>10.2f} img/s")

    utilization = (real_throughput / data_throughput) * 100
    print(f"Data Saturation: {utilization:.1f}%")


if __name__ == "__main__":
    main()
