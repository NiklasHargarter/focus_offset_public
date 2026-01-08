import torch
import torch.nn as nn
import lightning as L
import torchvision.utils


class FocusOffsetRegressor(L.LightningModule):
    """
    State-of-the-art LightningModule for focus offset regression.
    Supports full optimizer/scheduler injection, scalar metrics, and visual logging.
    """

    def __init__(
        self,
        backbone: nn.Module,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        targets = targets.unsqueeze(1)
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, targets = batch
        targets = targets.unsqueeze(1)
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        
        abs_err = torch.abs(outputs - targets)
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_mae", abs_err.mean(), on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        
        # Collect for std calculation
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
        images, targets = batch
        targets = targets.unsqueeze(1)
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Store a few samples from the FIRST batch for visual logging
        if batch_idx == 0:
            self.validation_step_outputs = {
                "images": images[:8].detach().cpu(),
                "targets": targets[:8].detach().cpu(),
                "outputs": outputs[:8].detach().cpu()
            }
        
        return loss

    def on_validation_epoch_end(self):
        """
        Log sample predictions overlaid on patches to TensorBoard at the end of each validation epoch.
        """
        if hasattr(self, "validation_step_outputs") and self.logger and hasattr(self.logger.experiment, "add_image"):
            from PIL import Image, ImageDraw
            import numpy as np

            samples = self.validation_step_outputs
            imgs = samples["images"]  # [N, 3, H, W]
            targets = samples["targets"]
            outputs = samples["outputs"]
            
            processed_patches = []
            
            for i in range(len(imgs)):
                # Convert tensor [0, 1] to PIL image [0, 255]
                img_np = (imgs[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                
                # Draw text overlay
                draw = ImageDraw.Draw(img_pil)
                
                pred = outputs[i].item()
                gt = targets[i].item()
                text = f"P:{pred:.2f} G:{gt:.2f}"
                
                # Draw a small shadow/background for readability
                draw.rectangle([2, 2, 110, 20], fill=(0, 0, 0, 150))
                draw.text((5, 5), text, fill=(255, 255, 255))
                
                # Convert back to tensor [0, 1]
                processed_patches.append(torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float() / 255.0)

            # Create and log the grid
            grid = torchvision.utils.make_grid(processed_patches, nrow=4)
            self.logger.experiment.add_image("val/annotated_samples", grid, self.global_step)
            
            # Clear the cache
            del self.validation_step_outputs

    def configure_optimizers(self):
        """
        Default optimizer configuration.
        This will be used when running manually (e.g., in benchmarks),
        but will be overridden by LightningCLI if an optimizer is provided in the config.
        """
        return torch.optim.Adam(self.parameters(), lr=1e-4)


# Note: configure_optimizers can be overridden or supplemented by LightningCLI 
# using the --optimizer and --lr_scheduler flags in the configuration.
