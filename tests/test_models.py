import pytest
import torch
from src.models.architectures import MODEL_REGISTRY


@pytest.mark.parametrize("model_name", ["rgb", "fft", "dwt", "multimodal"])
def test_ablation_model_forward(model_name):
    # Retrieve the model class from the registry
    model_class = MODEL_REGISTRY[model_name]
    model = model_class()
    model.eval()

    # Determine required channels based on the model type
    if model_name == "rgb":
        channels = 3
    elif model_name == "fft":
        channels = 1
    elif model_name == "dwt":
        channels = 3
    elif model_name == "multimodal":
        channels = 7
    else:
        channels = 3

    # The ablation models (like FFT/DWT models using ptwt/fft) actually
    # take in the raw RGB and convert internally in their forward passes.
    # Wait, let's verify if they take RGB and compute internally or take pre-computed.
    # From architectures.py: FFTModel takes x, computes `gray = x.mean(...)`, then fft.
    # Ah! They all take 3 channels (or at least standard shape) in their forward pass.
    # Let's check `architectures.py` again.

    # Wait, FFTModel:
    # def forward(self, x): gray = x.mean(dim=1, keepdim=True); ... then to resnet18(in_chans=1)
    # So `x` has 3 channels for ALL of them!

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 1), f"{model_name} output shape should be (2, 1)"


def test_production_model_forward():
    pytest.skip(
        "Skipping pretrained model forward test to avoid large network downloads."
    )
    model_class = MODEL_REGISTRY["rgb_convnext"]
    model = model_class()
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 1), "ConvNeXt output shape should be (2, 1)"
