import torch.nn as nn
import torchvision.models as models
from torchvision.models import ConvNeXt_Tiny_Weights


def get_model(device):
    """
    Returns a configured ConvNeXt Tiny model for regression (1 output).
    """
    print("Loading ConvNeXt Tiny...")
    # Using default ImageNet weights
    model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

    # Replace head for regression
    # ConvNeXt classifier is a Sequential ended with Linear(in_features, num_classes)
    # Typically: model.classifier[2] is the final linear layer
    last_layer_idx = len(model.classifier) - 1
    num_features = model.classifier[last_layer_idx].in_features
    model.classifier[last_layer_idx] = nn.Linear(num_features, 1)

    return model.to(device)
