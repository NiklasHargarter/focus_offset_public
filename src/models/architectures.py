import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
import timm
import ptwt


# ---------------------------------------------------------------------------
# Input Representations
# ---------------------------------------------------------------------------


class InputRepresentation(nn.Module, ABC):
    """Base class for input representations. Each declares its output channels."""

    out_channels: int

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform (B, 3, H, W) RGB input into (B, out_channels, H, W)."""
        ...


class RGBInput(InputRepresentation):
    """Pass-through RGB channels."""

    out_channels = 3

    def forward(self, x):
        return x


class FFTInput(InputRepresentation):
    """Log-magnitude of 2D FFT on grayscale."""

    out_channels = 1

    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)
        fft = torch.fft.fft2(gray)
        fft_mag = torch.log1p(torch.abs(fft))
        fft_mag = torch.fft.fftshift(fft_mag, dim=(-2, -1))
        # Normalize to roughly match RGB distribution
        fft_mag = (fft_mag - 4.5) / 1.0
        return fft_mag


class DWTInput(InputRepresentation):
    """Haar wavelet detail coefficients (LH, HL, HH)."""

    out_channels = 3

    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)
        coeffs = ptwt.wavedec2(gray, wavelet="haar", level=1)
        _, (LH, HL, HH) = coeffs
        dwt_details = torch.cat([LH, HL, HH], dim=1)
        # Scale to roughly unit variance
        dwt_details = dwt_details * 2.0
        # Upsample to match input resolution
        dwt_up = F.interpolate(dwt_details, size=x.shape[-2:], mode="nearest")
        return dwt_up


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


class BaseFocusRegressor(nn.Module, ABC):
    """Abstract base class for focus regressors."""

    @abstractmethod
    def forward(self, x): ...


class ConvNeXtV2FocusRegressor(BaseFocusRegressor):
    """
    ConvNeXt V2 backbone with composable input representations.
    Input channels are computed from the representation list.
    Prefer using the concrete subclasses below instead of this directly.
    """

    def __init__(self, representations: list[InputRepresentation]):
        super().__init__()
        self.representations = nn.ModuleList(representations)
        in_chans = sum(r.out_channels for r in representations)

        self.model = timm.create_model(
            "convnextv2_tiny",
            pretrained=False,
            num_classes=1,
            in_chans=in_chans,
        )

    def forward(self, x):
        features = [r(x) for r in self.representations]
        x = torch.cat(features, dim=1)
        return self.model(x)


# ---------------------------------------------------------------------------
# Concrete Models — use these instead of composing manually
# ---------------------------------------------------------------------------


class RGBModel(ConvNeXtV2FocusRegressor):
    """RGB-only baseline (3 channels)."""

    def __init__(self):
        super().__init__(representations=[RGBInput()])


class FFTModel(ConvNeXtV2FocusRegressor):
    """FFT-only model (1 channel)."""

    def __init__(self):
        super().__init__(representations=[FFTInput()])


class DWTModel(ConvNeXtV2FocusRegressor):
    """DWT-only model (3 channels)."""

    def __init__(self):
        super().__init__(representations=[DWTInput()])


class MultimodalModel(ConvNeXtV2FocusRegressor):
    """RGB + FFT + DWT ensemble (7 channels)."""

    def __init__(self):
        super().__init__(representations=[RGBInput(), FFTInput(), DWTInput()])


# Registry for model lookup by name
MODEL_REGISTRY = {
    "rgb": RGBModel,
    "fft": FFTModel,
    "dwt": DWTModel,
    "multimodal": MultimodalModel,
}
