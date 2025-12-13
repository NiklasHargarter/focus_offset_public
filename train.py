import os
import torch
from torch.utils.data import DataLoader, random_split
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

import config
from src.dataset.vsi_dataset import VSIDataset
from src.models.lightning_module import FocusOffsetRegressor

torch.set_float32_matmul_precision("medium")


def main() -> None:
    # 1. Setup Data
    L.seed_everything(42)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device strategy: {device_str}")

    index_path = config.get_index_path("train")
    if not index_path.exists():
        print(f"Training index not found at {index_path}. Run preprocess first.")
        return

    full_dataset = VSIDataset(mode="train")
    total_size = len(full_dataset)
    val_size = int(0.1 * total_size)
    train_size = total_size - val_size

    # Ensure consistent split
    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"Train samples: {train_size}, Val samples: {val_size}")

    train_loader = DataLoader(
        train_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count(),
        persistent_workers=True,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=os.cpu_count(),
        persistent_workers=True,
    )

    # 2. Setup Model
    model = FocusOffsetRegressor(
        arch_name=config.MODEL_ARCH, learning_rate=config.LEARNING_RATE
    )

    # 3. Setup Callbacks & Logger
    checkpoint_dir = config.CHECKPOINT_DIR / config.MODEL_ARCH
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Lightning uses strict path strings for checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=config.PATIENCE, mode="min", verbose=True
    )

    logger = CSVLogger(save_dir=str(checkpoint_dir), name="logs")

    # 4. Trainer
    trainer = L.Trainer(
        max_epochs=config.EPOCHS,
        precision="bf16-mixed",
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        log_every_n_steps=10,
    )

    print(f"Starting training for architecture: {config.MODEL_ARCH}")
    trainer.fit(model, train_loader, val_loader)

    # Save best model path for reference
    print(f"Best model path: {checkpoint_callback.best_model_path}")
    print("Training Complete.")


if __name__ == "__main__":
    main()
