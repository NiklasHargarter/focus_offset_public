"""
Provides ``train_one()`` — the single entry-point for fitting a model.
"""

import warnings
from pathlib import Path
import time

import torch
import torch.nn as nn


# Filter specific warnings if needed
warnings.filterwarnings(
    "ignore",
    message=".*Torchinductor does not support code generation for complex operators.*",
)


def train_one(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    log_name: str,
    *,
    max_epochs: int = 50,
    patience: int = 5,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.05,
    scheduler_patience: int = 5,
    seed: int = 42,
    dry_run: bool = False,
):
    """
    Train a single model using Accelerate.

    Args:
        model: nn.Module to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        log_name: Name for the logger directory (under logs/).
        max_epochs: Maximum training epochs.
        patience: Early stopping patience.
        learning_rate: Initial learning rate.
        weight_decay: Weight decay for AdamW.
        scheduler_patience: Patience for ReduceLROnPlateau.
        seed: Random seed.
        dry_run: If True, limit to 2 epochs and 20 batches.
    """
    # 0. Setup Environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = Path("logs_smoke_test" if dry_run else "logs") / log_name
    log_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)

    # 2. Loss & Optimizer & Scheduler
    # Use HuberLoss by default if model doesn't have one
    criterion = getattr(model, "criterion", nn.HuberLoss(delta=1.0))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=scheduler_patience
    )

    # 3. Prepare Model
    model = model.to(device)

    best_val_loss = float("inf")
    patience_counter = 0
    start_epoch = 0

    csv_path = log_dir / "metrics.csv"
    with open(csv_path, "w") as f:
        f.write("epoch,train_loss,val_loss,val_mae,lr,epoch_duration,timestamp\n")

    if dry_run:
        max_epochs = 2

    print(f"Training {log_name} for max {max_epochs} epochs on {device}...")

    for epoch in range(start_epoch, max_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        # Train Loop
        for batch_idx, batch in enumerate(train_loader):
            if dry_run and batch_idx >= 20:
                break

            dataset_images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            # Forward
            optimizer.zero_grad()
            with torch.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                dtype=torch.bfloat16,
            ):
                preds = model(dataset_images)
                loss = criterion(preds, targets.unsqueeze(1))

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Metrics
            loss_val = loss.item()
            train_loss_sum += loss_val
            train_steps += 1

        avg_train_loss = train_loss_sum / train_steps if train_steps > 0 else 0.0

        # Validation Loop
        model.eval()
        val_loss_sum = 0.0
        val_mae_sum = 0.0
        val_steps = 0

        for batch_idx, batch in enumerate(val_loader):
            if dry_run and batch_idx >= 10:
                break

            with torch.no_grad():
                dataset_images = batch["image"].to(device, non_blocking=True)
                targets = batch["target"].to(device, non_blocking=True)

                with torch.autocast(
                    device_type="cuda" if device.type == "cuda" else "cpu",
                    dtype=torch.bfloat16,
                ):
                    preds = model(dataset_images)
                    loss = criterion(preds, targets.unsqueeze(1))

                loss_val = loss.item()
                val_loss_sum += loss_val

                # MAE
                mae = torch.abs(preds - targets.unsqueeze(1)).mean().item()
                val_mae_sum += mae

                val_steps += 1

        avg_val_loss = val_loss_sum / val_steps if val_steps > 0 else 0.0
        avg_val_mae = val_mae_sum / val_steps if val_steps > 0 else 0.0

        # Update Scheduler
        scheduler.step(avg_val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_duration = time.time() - epoch_start_time
        current_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Logging
        print(
            f"Epoch {epoch}: "
            f"Train Loss={avg_train_loss:.4f}, "
            f"Val Loss={avg_val_loss:.4f}, "
            f"Val MAE={avg_val_mae:.4f}, "
            f"LR={current_lr:.2e}, "
            f"Time={epoch_duration:.1f}s"
        )

        with open(csv_path, "a") as f:
            f.write(
                f"{epoch},{avg_train_loss},{avg_val_loss},{avg_val_mae},{current_lr},{epoch_duration:.2f},{current_timestamp}\n"
            )

        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            checkpoint_dir = log_dir / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)

            # Save model
            save_path = checkpoint_dir / f"best-e{epoch:03d}-vl{avg_val_loss:.6f}.pt"

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "loss": avg_val_loss,
                },
                save_path,
            )

            # Cleanup old checkpoints: keep only top 3 best
            ckpt_files = sorted(
                checkpoint_dir.glob("best-*.pt"),
                key=lambda p: float(p.name.split("vl")[1].replace(".pt", "")),
            )
            for ckpt_to_remove in ckpt_files[3:]:
                ckpt_to_remove.unlink()

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Post-training cleanup and final model saving
    checkpoint_dir = log_dir / "checkpoints"
    if checkpoint_dir.exists():
        # Find the absolute best checkpoint
        ckpt_files = sorted(
            checkpoint_dir.glob("best-*.pt"),
            key=lambda p: float(p.name.split("vl")[1].replace(".pt", "")),
        )
        if ckpt_files:
            best_ckpt_path = ckpt_files[0]  # The lowest validation loss

            # Load the best state dictated by the checkpoint
            checkpoint_data = torch.load(best_ckpt_path, map_location="cpu")
            best_epoch = checkpoint_data["epoch"]
            best_loss = checkpoint_data["loss"]

            # Name it descriptively without redundantly including the log_name
            final_name = f"best_e{best_epoch:03d}_vloss{best_loss:.4f}.pt"
            final_path = log_dir / final_name

            # Save purely the model weights (stripping optimizer/epoch metadata for clean inference)
            torch.save(checkpoint_data["model_state_dict"], final_path)
            print(f"Saved final pristine model to {final_path}")

        # Clean up the intermediate checkpointing directory to save disk space
        import shutil

        shutil.rmtree(checkpoint_dir)

    return model
