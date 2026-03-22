"""
Specialized training script for supervised training on synthetic data.
This script trains a single Conv2d layer model to map an offset input patch
to its corresponding ground-truth focal plane patch.
"""

import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from skimage.morphology import disk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from synthetic_data.config import SyntheticConfig
from focus_offset.utils.env import setup_environment
from synthetic_data.dataset import get_synthetic_dataloaders
from synthetic_data.model import SyntheticConvModel
from synthetic_data.plot.plot_kernel_weights import plot_kernel_weights


def get_teacher_kernel(k_size=31, radius=5.0, device="cpu"):
    # disk(radius) creates a morphological pillbox with diameter 2*radius+1
    d_base = disk(int(radius)).astype(float)

    pad_size = (k_size - d_base.shape[0]) // 2
    d_k = np.pad(d_base, ((pad_size, pad_size), (pad_size, pad_size)), mode="constant")

    # Normalize so energy is conserved (Sum = 1.0)
    d_k /= d_k.sum()

    # Convert to Teacher Weights [3 channels, 1, k_size, k_size]
    teacher_kernel = torch.from_numpy(d_k).float().expand(3, 1, k_size, k_size).clone()
    return teacher_kernel.to(device)


def _autocast_context(device: str, enabled: bool):
    """Use mixed precision on CUDA only when explicitly enabled."""
    if enabled and "cuda" in device:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def train_synthetic(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    log_name: str,
    *,
    max_epochs: int = 50,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    dry_run: bool = False,
    config: SyntheticConfig = None,
):
    log_dir = config.log_dir if config else Path("logs") / log_name
    log_dir.mkdir(parents=True, exist_ok=True)
    use_amp = bool(getattr(config, "use_amp", False))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    model.to(device)

    best_val_metric = None

    csv_path = log_dir / "metrics_synthetic.csv"
    with open(csv_path, "w") as f:
        f.write("epoch,train_loss,val_loss,lr,epoch_duration\n")

    print(f"Starting Synthetic Training: {log_name}")

    # Pre-build simulation kernel if in simulation mode
    teacher_kernel = None
    if config.simulation_mode:
        teacher_kernel = get_teacher_kernel(
            k_size=config.kernel_size,
            radius=config.simulation_radius,
            device=device,
        )
        print("Using GPU Simulation Mode.")

    # Save initial state before any training
    torch.save(model.state_dict(), log_dir / "initial_model.pt")
    try:
        plot_kernel_weights(weights=model.conv.weight, epoch=0, log_dir=log_dir)
    except Exception as exc:
        print(f"Warning: Failed to plot initial kernel heatmap: {exc}")

    for epoch in range(max_epochs):
        epoch_start = time.time()
        kernel_mae = None
        kernel_max_abs = None
        kernel_l1_sum = None
        kernel_l2_sum = None

        # Train
        model.train()
        train_loss_sum = 0.0
        train_steps = 0
        for batch_idx, batch in enumerate(train_loader):
            if dry_run and batch_idx >= 5:
                break

            inputs = batch["input"].to(device)

            if config.simulation_mode:
                # Dynamically generate targets on GPU using Teacher Gaussian Kernel
                with torch.no_grad():
                    targets = F.conv2d(inputs, teacher_kernel, groups=3, padding=0)
            else:
                targets = batch["target"].to(device)

            optimizer.zero_grad()
            with _autocast_context(device, enabled=use_amp):
                preds = model(inputs)
                loss = criterion(preds, targets)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_steps += 1

        avg_train_loss = train_loss_sum / train_steps if train_steps > 0 else 0.0

        # Validation
        model.eval()
        with torch.no_grad():
            if config.simulation_mode:
                # Directly compare learned kernel to teacher kernel — no data needed.
                # Validation value is weight-wise absolute difference summed over all weights.
                kernel_diff = model.conv.weight - teacher_kernel
                kernel_l1_sum = kernel_diff.abs().sum().item()
                kernel_l2_sum = kernel_diff.pow(2).sum().item()
                kernel_mae = F.l1_loss(model.conv.weight, teacher_kernel).item()
                kernel_max_abs = kernel_diff.abs().max().item()
                avg_val_loss = kernel_mae
            else:
                val_loss_sum = 0.0
                val_steps = 0
                for batch_idx, batch in enumerate(val_loader):
                    if dry_run and batch_idx >= 5:
                        break
                    inputs = batch["input"].to(device)
                    targets = batch["target"].to(device)
                    with _autocast_context(device, enabled=use_amp):
                        preds = model(inputs)
                        loss = criterion(preds, targets)
                    val_loss_sum += loss.item()
                    val_steps += 1
                avg_val_loss = val_loss_sum / val_steps if val_steps > 0 else 0.0
        duration = time.time() - epoch_start

        if config.simulation_mode:
            print(
                "Epoch "
                f"{epoch}: Train LOSS={avg_train_loss:.6f}, "
                f"Val K-MAE={avg_val_loss:.6f}, "
                f"K-L1-SUM={kernel_l1_sum:.6f}, "
                f"K-L2-SUM={kernel_l2_sum:.3e}, "
                f"K-MAX={kernel_max_abs:.3e}, "
                f"Time={duration:.1f}s",
                flush=True,
            )
        else:
            print(
                f"Epoch {epoch}: Train LOSS={avg_train_loss:.6f}, Val LOSS={avg_val_loss:.6f}, Time={duration:.1f}s",
                flush=True,
            )

        with open(csv_path, "a") as f:
            f.write(
                f"{epoch},{avg_train_loss},{avg_val_loss},{learning_rate},{duration:.2f}\n"
            )

        # Checkpointing
        if config.simulation_mode:
            # Use raw weight-wise absolute error sum for checkpoint selection.
            is_better = best_val_metric is None or avg_val_loss < best_val_metric
            if is_better:
                best_val_metric = avg_val_loss
                torch.save(model.state_dict(), log_dir / "best_model.pt")
                print(
                    f"  --> Saved new best model (K-MAE: {best_val_metric:.6f})",
                    flush=True,
                )
        else:
            is_better = best_val_metric is None or avg_val_loss < best_val_metric
            if is_better:
                best_val_metric = avg_val_loss
                torch.save(model.state_dict(), log_dir / "best_model.pt")
                print(
                    f"  --> Saved new best model (Val Loss: {best_val_metric:.3e})",
                    flush=True,
                )

        # Periodic waypoint checkpointing (5, 10, 15, ...)
        if (epoch + 1) % config.vis_interval == 0:
            print(f"Saving checkpoint for epoch {epoch + 1}...")
            torch.save(model.state_dict(), log_dir / f"model_epoch_{epoch + 1}.pt")
            try:
                plot_kernel_weights(
                    weights=model.conv.weight,
                    epoch=epoch + 1,
                    log_dir=log_dir,
                )
            except Exception as exc:
                print(
                    f"Warning: Failed to plot kernel heatmap for epoch {epoch + 1}: {exc}"
                )

    if config.simulation_mode:
        print(f"Training finished. Best K-MAE: {best_val_metric:.6f}")
    else:
        print(f"Training finished. Best Val Loss: {best_val_metric:.6f}")
    return model


def main():
    setup_environment()

    synth_config = SyntheticConfig()

    num_workers = synth_config.workers

    if synth_config.simulation_mode:
        train_loader, _ = get_synthetic_dataloaders(
            config=synth_config, num_workers=num_workers
        )
        val_loader = None
    else:
        train_loader, val_loader = get_synthetic_dataloaders(
            config=synth_config, num_workers=num_workers
        )

    model = SyntheticConvModel(
        kernel_size=synth_config.kernel_size,
        groups=synth_config.groups,
        weight_init=synth_config.weight_init,
    )

    log_name = synth_config.experiment_name

    train_synthetic(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        log_name=log_name,
        max_epochs=synth_config.max_epochs,
        learning_rate=synth_config.lr,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dry_run=synth_config.dry_run,
        config=synth_config,
    )


if __name__ == "__main__":
    main()
