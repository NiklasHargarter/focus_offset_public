import torch
import warnings

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

import config
from src.dataset.vsi_datamodule import VSIDataModule
from src.models.lightning_module import FocusOffsetRegressor

warnings.filterwarnings(
    "ignore", ".*Precision bf16-mixed is not supported by the model summary.*"
)
torch.set_float32_matmul_precision("medium")


def main() -> None:
    L.seed_everything(42)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device strategy: {device_str}")

    datamodule = VSIDataModule()

    model = FocusOffsetRegressor(
        arch_name=config.MODEL_ARCH, learning_rate=config.LEARNING_RATE
    )

    checkpoint_dir = config.CHECKPOINT_DIR / config.MODEL_ARCH
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=False,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=config.PATIENCE, mode="min", verbose=False
    )

    logger = TensorBoardLogger(save_dir=str(checkpoint_dir), name="logs")

    trainer = L.Trainer(
        max_epochs=config.EPOCHS,
        precision="bf16-mixed",
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        log_every_n_steps=50,
        enable_model_summary=False,
    )

    print(f"Starting training for architecture: {config.MODEL_ARCH}")
    trainer.fit(model, datamodule=datamodule)

    print(f"Best model path: {checkpoint_callback.best_model_path}")
    print("Training Complete.")


if __name__ == "__main__":
    main()
