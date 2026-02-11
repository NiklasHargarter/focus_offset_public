import torch
import torch.nn as nn
import lightning as L
import math
from torchmetrics import MeanAbsoluteError
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
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
    ):
        super().__init__()
        self.save_hyperparameters()

        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model name: {model_name}. Available: {list(MODEL_REGISTRY.keys())}"
            )

        self.backbone = MODEL_REGISTRY[model_name]()
        # Optimization: Use channels_last memory format
        self.backbone.to(memory_format=torch.channels_last)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.criterion = nn.HuberLoss(delta=1.0)

        # Metrics
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch["image"], batch["target"]
        if images.device.type == "cuda":
            images = images.to(memory_format=torch.channels_last)

        preds = self(images)
        loss = self.criterion(preds, targets.unsqueeze(1))

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # Log learning rate
        sch = self.lr_schedulers()
        if sch is not None:
            self.log(
                "lr", sch.get_last_lr()[0], on_step=True, on_epoch=False, prog_bar=False
            )

        return loss

    def _common_eval_step(self, batch):
        images, targets = batch["image"], batch["target"]
        if images.device.type == "cuda":
            images = images.to(memory_format=torch.channels_last)

        preds = self(images)
        loss = self.criterion(preds, targets.unsqueeze(1))
        return loss, preds, targets

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._common_eval_step(batch)

        self.val_mae(preds, targets.unsqueeze(1))
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_mae",
            self.val_mae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self._common_eval_step(batch)

        self.test_mae(preds, targets.unsqueeze(1))
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "test_mae",
            self.test_mae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def predict_step(self, batch, batch_idx):
        """Return raw predictions only. Eval scripts pair these with batch data."""
        images = batch["image"]
        if images.device.type == "cuda":
            images = images.to(memory_format=torch.channels_last)
        return self(images)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        max_epochs = self.trainer.max_epochs
        warmup_steps = 1
        main_steps = max(1, max_epochs - warmup_steps)

        def lr_lambda(epoch):
            if epoch < warmup_steps:
                return float(epoch + 1) / warmup_steps
            progress = float(epoch - warmup_steps) / main_steps
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
