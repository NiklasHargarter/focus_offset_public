import torch.nn as nn
import torchvision.models as models
from abc import ABC, abstractmethod


class BaseFocusRegressor(nn.Module, ABC):
    """
    Abstract base class for all focus regressors.
    Ensures that every model outputs a single scalar for regression.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass


class ResNetFocusRegressor(BaseFocusRegressor):
    def __init__(self, version: str = "resnet18", pretrained: bool = True):
        super().__init__()
        if version == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.resnet18(weights=weights)
        elif version == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported ResNet version: {version}")

        # Replace FC head
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.model(x)


class ViTFocusRegressor(BaseFocusRegressor):
    def __init__(self, version: str = "vit_b_16", pretrained: bool = True):
        super().__init__()
        if version == "vit_b_16":
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.vit_b_16(weights=weights)
        else:
            raise ValueError(f"Unsupported ViT version: {version}")

        # Replace classification head
        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.model(x)


class ConvNeXtFocusRegressor(BaseFocusRegressor):
    def __init__(self, version: str = "tiny", pretrained: bool = True):
        super().__init__()
        if version == "tiny":
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.convnext_tiny(weights=weights)
        else:
            raise ValueError(f"Unsupported ConvNeXt version: {version}")

        # Replace classification head
        last_layer_idx = len(self.model.classifier) - 1
        in_features = self.model.classifier[last_layer_idx].in_features
        self.model.classifier[last_layer_idx] = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.model(x)


class EfficientNetFocusRegressor(BaseFocusRegressor):
    def __init__(self, version: str = "b0", pretrained: bool = True):
        super().__init__()
        if version == "b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.efficientnet_b0(weights=weights)
        else:
            raise ValueError(f"Unsupported EfficientNet version: {version}")

        # Replace classification head
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.model(x)
