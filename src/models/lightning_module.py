import torch
import torch.nn as nn
import lightning as L
import math


class FocusOffsetRegressor(L.LightningModule):
    """
    LightningModule for focus offset regression.
    Supports ablation studies with configurable input domains (RGB, FFT, DWT).
    """

    def __init__(
        self,
        pretrained: bool = False,
        transform_mode: str = "all",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        save_predictions: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        from src.models.architectures import ConvNeXtV2FocusRegressor

        self.backbone = ConvNeXtV2FocusRegressor(
            pretrained=pretrained, transform_mode=transform_mode
        )

        # Optimization: Use channels_last memory format
        self.backbone.to(memory_format=torch.channels_last)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.criterion = nn.HuberLoss(delta=1.0)

    def forward(self, x):
        return self.backbone(x)

    def _unpack_batch(self, batch):
        """Unpacks batch into images and targets, handling optional metadata."""
        if len(batch) == 3:
            images, targets, _ = batch
        else:
            images, targets = batch
        return images, targets

    def _shared_step(self, batch):
        images, targets = self._unpack_batch(batch)

        if images.device.type == "cuda":
            images = images.to(memory_format=torch.channels_last)

        targets = targets.unsqueeze(1)
        outputs = self(images)
        loss = self.criterion(outputs, targets)

        return loss, outputs, targets

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch[0].size(0),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs, targets = self._shared_step(batch)
        mae = torch.abs(outputs - targets).mean()

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch[0].size(0),
        )
        self.log(
            "val_mae",
            mae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch[0].size(0),
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, outputs, targets = self._shared_step(batch)
        abs_err = torch.abs(outputs - targets)

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].size(0),
        )
        self.log(
            "test_mae",
            abs_err.mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch[0].size(0),
        )

        if not hasattr(self, "test_step_outputs"):
            self.test_step_outputs = []
        self.test_step_outputs.append(abs_err.detach().cpu())

        return abs_err

    def on_test_epoch_end(self):
        if hasattr(self, "test_step_outputs"):
            all_errors = torch.cat(self.test_step_outputs)
            mae = all_errors.mean()
            std = all_errors.std()
            self.log("test_mae_final", mae)
            self.log("test_std", std)
            del self.test_step_outputs

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
