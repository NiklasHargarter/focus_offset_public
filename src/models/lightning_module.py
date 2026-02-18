import lightning as L
import torch
import torch.nn as nn
from torchmetrics import MeanAbsoluteError

from src import config
from src.models.architectures import (
    MODEL_REGISTRY,
)


class FocusOffsetRegressor(L.LightningModule):
    """
    LightningModule for focus offset regression.
    Handles training, validation, and basic test metrics only.
    Detailed analysis belongs in dedicated evaluation scripts.
    """

    def __init__(
        self,
        model_name: str = "multimodal",
        learning_rate: float = config.LEARNING_RATE,
        weight_decay: float = config.WEIGHT_DECAY,
        scheduler_patience: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters()

        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model name: {model_name}. Available: {list(MODEL_REGISTRY.keys())}"
            )

        self.backbone = MODEL_REGISTRY[model_name]()
        self.criterion = nn.HuberLoss(delta=1.0)

        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        loss, _preds, _targets, images = self._common_eval_step(batch)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.size(0),
        )

        return loss

    def _common_eval_step(self, batch):
        images, targets = batch["image"], batch["target"]
        if images.device.type == "cuda":
            images = images.to(memory_format=torch.channels_last)

        preds = self(images)
        loss = self.criterion(preds, targets.unsqueeze(1))
        return loss, preds, targets, images

    def validation_step(self, batch, batch_idx):
        loss, preds, targets, images = self._common_eval_step(batch)

        self.val_mae(preds, targets.unsqueeze(1))
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.size(0),
        )
        self.log(
            "val_mae",
            self.val_mae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.size(0),
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, targets, images = self._common_eval_step(batch)

        self.test_mae(preds, targets.unsqueeze(1))
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, batch_size=images.size(0)
        )
        self.log(
            "test_mae",
            self.test_mae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.size(0),
        )
        return loss

    def predict_step(self, batch, batch_idx):
        """Return predictions, targets, and metadata for evaluation.

        Returning everything from a single forward pass avoids re-iterating
        the dataloader in eval scripts, keeping predictions and metadata
        perfectly aligned.
        """
        _loss, preds, _targets, _images = self._common_eval_step(batch)
        result = {"pred": preds, "target": batch["target"]}
        if "metadata" in batch:
            result["metadata"] = batch["metadata"]
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=self.hparams.scheduler_patience
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }
