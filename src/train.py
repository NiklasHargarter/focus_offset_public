import os
import torch
import warnings
import lightning as L
from lightning.pytorch.cli import LightningCLI

from src.models.lightning_module import FocusOffsetRegressor

from src.dataset.vsi_datamodule import VSIDataModule

# General best practices for training performance
torch.set_float32_matmul_precision("medium")

warnings.filterwarnings(
    "ignore", ".*Precision bf16-mixed is not supported by the model summary.*"
)


class FocusCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--compile", action="store_true", default=False)

    def before_fit(self):
        if self.config["fit"].get("compile", True):
            print("Compiling model for optimized performance...")
            # We compile the backbone inside the lightning module
            self.model.backbone = torch.compile(self.model.backbone)


def cli_main():
    FocusCLI(
        model_class=FocusOffsetRegressor,
        datamodule_class=VSIDataModule,
        subclass_mode_data=False,
        seed_everything_default=42,
        save_config_callback=None,
        run=True,
    )


if __name__ == "__main__":
    cli_main()
