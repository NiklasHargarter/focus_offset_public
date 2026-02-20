"""
Core training logic.

Provides ``train_one()`` — the single entry-point for fitting a model.
Imported by the CLI scripts in ``scripts/``.
"""

import warnings
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator
from tqdm.auto import tqdm


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
    scheduler_patience: int = 3,
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
    # 0. Setup Accelerator
    accelerator = Accelerator(
        mixed_precision="bf16",  # Force bf16 as in original config if supported, else "fp16" or "no"
        log_with="all",
        project_dir="logs_smoke_test" if dry_run else "logs",
    )

    # Make sure log directory exists
    log_dir = Path(accelerator.project_dir) / log_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # 1. Reproducibility
    torch.manual_seed(seed)
    # Accelerate handles device seeding

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

    # 3. Prepare with Accelerate
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # 4. Training Loop State
    best_val_loss = float("inf")
    patience_counter = 0
    start_epoch = 0

    csv_path = log_dir / "metrics.csv"
    with open(csv_path, "w") as f:
        f.write("epoch,train_loss,val_loss,val_mae,lr\n")

    if dry_run:
        max_epochs = 2

    print(f"Training {log_name} for max {max_epochs} epochs on {accelerator.device}...")

    for epoch in range(start_epoch, max_epochs):
        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        # Train Loop
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{max_epochs} [Train]",
            disable=not accelerator.is_local_main_process,
            leave=False,
        )

        for batch_idx, batch in enumerate(progress_bar):
            if dry_run and batch_idx >= 20:
                break

            dataset_images = batch["image"]
            targets = batch["target"]
            # Accelerate handles device placement

            # Forward
            optimizer.zero_grad()
            preds = model(dataset_images)
            loss = criterion(preds, targets.unsqueeze(1))

            # Backward
            accelerator.backward(loss)
            optimizer.step()

            # Metrics
            loss_val = loss.item()
            train_loss_sum += loss_val
            train_steps += 1

            progress_bar.set_postfix({"loss": f"{loss_val:.4f}"})

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
                dataset_images = batch["image"]
                targets = batch["target"]

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

        # Logging
        if accelerator.is_local_main_process:
            print(
                f"Epoch {epoch}: "
                f"Train Loss={avg_train_loss:.4f}, "
                f"Val Loss={avg_val_loss:.4f}, "
                f"Val MAE={avg_val_mae:.4f}, "
                f"LR={current_lr:.2e}"
            )

            with open(csv_path, "a") as f:
                f.write(
                    f"{epoch},{avg_train_loss},{avg_val_loss},{avg_val_mae},{current_lr}\n"
                )

            # Checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0

                checkpoint_dir = log_dir / "checkpoints"
                checkpoint_dir.mkdir(exist_ok=True)

                # Save model
                # Unwrap model for saving
                unwrapped_model = accelerator.unwrap_model(model)
                save_path = (
                    checkpoint_dir / f"best-e{epoch:02d}-vl{avg_val_loss:.4f}.pt"
                )

                # Torch save
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": unwrapped_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": avg_val_loss,
                        "model_name": log_name,
                    },
                    save_path,
                )

                # Save a copy as "best_weights.pt"
                torch.save(unwrapped_model.state_dict(), log_dir / "best_weights.pt")

            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Wait for all processes (if distributed)
        accelerator.wait_for_everyone()

    return accelerator
