import torch.nn as nn
import torch.optim as optim
import lightning as L
from src.models.factory import get_model, ModelArch


class FocusOffsetRegressor(L.LightningModule):
    def __init__(self, arch_name: ModelArch, learning_rate: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = get_model(arch_name)
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        targets = targets.unsqueeze(1)
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        targets = targets.unsqueeze(1)
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        return outputs, targets.unsqueeze(1)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.learning_rate)
