import numpy as np
import torch
from src.dataset.loader import get_transforms


def test_train_transforms():
    transforms = get_transforms("train")
    dummy_image = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)

    augmented = transforms(image=dummy_image)
    tensor = augmented["image"]

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.max() <= 1.0
    assert tensor.min() >= 0.0


def test_val_transforms():
    transforms = get_transforms("val")
    dummy_image = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)

    augmented = transforms(image=dummy_image)
    tensor = augmented["image"]

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.max() <= 1.0
    assert tensor.min() >= 0.0
