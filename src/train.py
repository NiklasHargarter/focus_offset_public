import torch
import warnings
import lightning as L
from lightning.pytorch.cli import LightningCLI

from src.models.lightning_module import FocusOffsetRegressor

warnings.filterwarnings(
    "ignore", ".*Precision bf16-mixed is not supported by the model summary.*"
)
torch.set_float32_matmul_precision("medium")


def cli_main():
    LightningCLI(
        model_class=FocusOffsetRegressor,
        datamodule_class=L.LightningDataModule,
        subclass_mode_data=True,
        seed_everything_default=42,
        save_config_callback=None,  # Config will be saved with checkpoints automatically
        run=True,
    )


if __name__ == "__main__":
    cli_main()
