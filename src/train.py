import os
import torch
import warnings
import lightning as L
from lightning.pytorch.cli import LightningCLI

from src.models.lightning_module import FocusOffsetRegressor

# Best practices for performance & stability (Blackwell / RTX 5090)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision("medium")

warnings.filterwarnings(
    "ignore", ".*Precision bf16-mixed is not supported by the model summary.*"
)


class FocusCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--compile", action="store_true", default=True)

    def before_fit(self):
        if self.config["fit"].get("compile", True):
            print("Compiling backbone for maximum performance...")
            # We compile the backbone inside the lightning module
            self.model.backbone = torch.compile(self.model.backbone)


def cli_main():
    FocusCLI(
        model_class=FocusOffsetRegressor,
        datamodule_class=L.LightningDataModule,
        subclass_mode_data=True,
        seed_everything_default=42,
        save_config_callback=None,
        parser_kwargs={"default_config_files": ["default_config.yaml"]},
        run=True,
    )


if __name__ == "__main__":
    cli_main()
