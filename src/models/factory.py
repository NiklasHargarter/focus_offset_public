import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    ViT_B_16_Weights,
)

from enum import Enum


class ModelArch(str, Enum):
    CONVNEXT_TINY = "convnext_tiny"
    RESNET18 = "resnet18"
    RESNET50 = "resnet50"
    EFFICIENTNET_B0 = "efficientnet_b0"
    MOBILENET_V3_LARGE = "mobilenet_v3_large"
    VIT_B_16 = "vit_b_16"


def get_model(arch_name: ModelArch):
    """
    Returns a configured model for regression (1 output).
    Supported architectures:
    - convnext_tiny
    - resnet18
    - resnet50
    - efficientnet_b0
    - mobilenet_v3_large
    - vit_b_16
    """
    print(f"Loading Model Architecture: {arch_name.value}...")

    match arch_name:
        case ModelArch.CONVNEXT_TINY:
            model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            last_layer_idx = len(model.classifier) - 1
            num_features = model.classifier[last_layer_idx].in_features
            model.classifier[last_layer_idx] = nn.Linear(num_features, 1)

        case ModelArch.RESNET18:
            model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 1)

        case ModelArch.RESNET50:
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 1)

        case ModelArch.EFFICIENTNET_B0:
            model = models.efficientnet_b0(
                weights=EfficientNet_B0_Weights.IMAGENET1K_V1
            )
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, 1)

        case ModelArch.MOBILENET_V3_LARGE:
            model = models.mobilenet_v3_large(
                weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1
            )
            last_layer_idx = len(model.classifier) - 1
            num_features = model.classifier[last_layer_idx].in_features
            model.classifier[last_layer_idx] = nn.Linear(num_features, 1)

        case ModelArch.VIT_B_16:
            model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

            num_features = model.heads.head.in_features
            model.heads.head = nn.Linear(num_features, 1)

        case _:
            raise ValueError(
                f"Unknown architecture: {arch_name}. Supported: {[m.value for m in ModelArch]}"
            )

    return model
