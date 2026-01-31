import torch
import torch.nn as nn
import lightning as L
import math


class FocusOffsetRegressor(L.LightningModule):
    """
    State-of-the-art LightningModule for focus offset regression.
    Supports full optimizer/scheduler injection, scalar metrics, and visual logging.
    """

    def __init__(
        self,
        backbone: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        save_predictions: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone
        # Use channels_last memory format for modern NVIDIA GPUs
        self.backbone.to(memory_format=torch.channels_last)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_predictions = save_predictions
        self.criterion = nn.HuberLoss(delta=1.0)

    def forward(self, x):
        return self.backbone(x)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """
        Efficient normalization hook.
        Moves float conversion to GPU to reduce PCIe bandwidth usage.
        """
        # Handle batches with and without metadata
        if len(batch) == 3:
            images, targets, metadata = batch
        else:
            images, targets = batch
            metadata = None

        # If data is still uint8, move to float efficiently on GPU
        if images.dtype == torch.uint8:
            images = images.float() / 255.0

        return (
            (images, targets, metadata) if metadata is not None else (images, targets)
        )

    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            images, targets, _ = batch
        else:
            images, targets = batch

        # Ensure channels_last for Tensor Cores
        if images.device.type == "cuda":
            images = images.to(memory_format=torch.channels_last)

        targets = targets.unsqueeze(1)
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        self.log(
            "train_loss",
            loss,
            on_step=False,  # Focus on epoch for cross-BS comparison
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=images.size(0),
        )
        return loss

    def test_step(self, batch, batch_idx):
        if len(batch) == 3:
            images, targets, _ = batch
        else:
            images, targets = batch

        if images.device.type == "cuda":
            images = images.to(memory_format=torch.channels_last)

        targets = targets.unsqueeze(1)
        outputs = self(images)
        loss = self.criterion(outputs, targets)

        abs_err = torch.abs(outputs - targets)
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "test_mae",
            abs_err.mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
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

    def validation_step(self, batch, batch_idx):
        if len(batch) == 3:
            images, targets, metadata = batch
        else:
            images, targets = batch
            metadata = None

        if images.device.type == "cuda":
            images = images.to(memory_format=torch.channels_last)

        targets = targets.unsqueeze(1)
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        mae = torch.abs(outputs - targets).mean()

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=images.size(0),
        )
        self.log(
            "val_mae",
            mae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=images.size(0),
        )

        # Optimization: Batch-transfer metadata to CPU to avoid graph-breaking .item() calls
        # We only do this if we are not in a benchmark (logger exists) and save_predictions is enabled
        if self.logger and self.save_predictions:
            pred_flat = outputs.squeeze(1).detach().float().cpu().numpy()
            target_flat = targets.squeeze(1).detach().float().cpu().numpy()

            # Efficiently move all metadata tensors to CPU at once
            xs = metadata["x"].detach().cpu().numpy()
            ys = metadata["y"].detach().cpu().numpy()
            zs = metadata["z_level"].detach().cpu().numpy()
            opt_zs = metadata["optimal_z"].detach().cpu().numpy()
            filenames = metadata["filename"]

            batch_data = []
            for i in range(len(pred_flat)):
                batch_data.append(
                    {
                        "filename": filenames[i],
                        "x": int(xs[i]),
                        "y": int(ys[i]),
                        "z_level": int(zs[i]),
                        "optimal_z": int(opt_zs[i]),
                        "prediction": float(pred_flat[i]),
                        "target": float(target_flat[i]),
                        "error": float(abs(pred_flat[i] - target_flat[i])),
                    }
                )

            if not hasattr(self, "validation_step_outputs"):
                self.validation_step_outputs = []
            self.validation_step_outputs.extend(batch_data)

        return loss

    def on_validation_epoch_end(self):
        """
        Aggregates per-sample validation results and saves them to a CSV.
        """
        if hasattr(self, "validation_step_outputs") and self.validation_step_outputs:
            import pandas as pd
            import os

            df = pd.DataFrame(self.validation_step_outputs)

            # Determine save directory
            if self.logger and self.logger.log_dir:
                save_dir = self.logger.log_dir
            else:
                save_dir = "logs"

            os.makedirs(save_dir, exist_ok=True)

            # Save CSV with epoch number
            filename = f"val_predictions_epoch_{self.current_epoch}.csv"
            save_path = os.path.join(save_dir, filename)
            df.to_csv(save_path, index=False)

            # Clear memory
            del self.validation_step_outputs

    def configure_optimizers(self):
        """
        Configures a simple AdamW optimizer and CosineAnnealingLR.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Warmup period (1 epoch for early peak protection)
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
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


class FocusEnsemble(L.LightningModule):
    """
    Ensemble wrapper that averages predictions from multiple FocusOffsetRegressor models.
    """

    def __init__(self, models: list[FocusOffsetRegressor]):
        super().__init__()
        self.models = nn.ModuleList(models)
        # Optimization: Use channels_last for all models
        for m in self.models:
            m.to(memory_format=torch.channels_last)

    def forward(self, x):
        # Average predictions from all models
        preds = torch.stack([model(x) for model in self.models])
        return torch.mean(preds, dim=0)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if len(batch) == 3:
            images, targets, metadata = batch
        else:
            images, targets = batch
            metadata = None

        if images.dtype == torch.uint8:
            images = images.float() / 255.0

        return (
            (images, targets, metadata) if metadata is not None else (images, targets)
        )

    def test_step(self, batch, batch_idx):
        if len(batch) == 3:
            images, targets, _ = batch
        else:
            images, targets = batch
        if images.device.type == "cuda":
            images = images.to(memory_format=torch.channels_last)

        targets = targets.unsqueeze(1)
        outputs = self(images)
        loss = nn.functional.huber_loss(outputs, targets)

        abs_err = torch.abs(outputs - targets)
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "test_mae",
            abs_err.mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
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
