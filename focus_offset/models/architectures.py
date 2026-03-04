import ptwt

import timm
import torch
import torch.fft
import torch.nn as nn
import torchvision.transforms.functional as tvF

# ---------------------------------------------------------------------------
# Transformation functions for WSI focus prediction
# ---------------------------------------------------------------------------


def _normalize_log_fft(magnitude: torch.Tensor) -> torch.Tensor:
    """
    Safely applies log-scaling and min-max normalization using native PyTorch.
    Input expected: (..., H, W)
    """
    log_mag = torch.log1p(magnitude)

    # Min-Max normalization per image/channel
    min_val = torch.amin(log_mag, dim=[-2, -1], keepdim=True)
    max_val = torch.amax(log_mag, dim=[-2, -1], keepdim=True)

    range_val = max_val - min_val
    return (log_mag - min_val) / (range_val + 1e-8)


def rgb_fft(tensor: torch.Tensor) -> torch.Tensor:
    """Input: (B, 3, H, W) -> Output: (B, 3, H, W)"""
    fft_complex = torch.fft.fft2(tensor, norm="forward")
    fft_shifted = torch.fft.fftshift(fft_complex, dim=(-2, -1))
    magnitude = torch.abs(fft_shifted)
    return _normalize_log_fft(magnitude)


def grayscale_fft(tensor: torch.Tensor) -> torch.Tensor:
    """Input: (B, 3, H, W) -> Output: (B, 1, H, W)"""
    gray = tvF.rgb_to_grayscale(tensor, num_output_channels=1)
    fft_complex = torch.fft.fft2(gray, norm="forward")
    fft_shifted = torch.fft.fftshift(fft_complex, dim=(-2, -1))
    magnitude = torch.abs(fft_shifted)
    return _normalize_log_fft(magnitude)


def fourier_domain_only_features(tensor: torch.Tensor) -> torch.Tensor:
    """Input: (B, 3, H, W) -> Output: (B, 2, H, W)
    Channel 0: Log-normalized Magnitude
    Channel 1: Normalized Angle
    """
    gray = tvF.rgb_to_grayscale(tensor, num_output_channels=1)
    fft_complex = torch.fft.fft2(gray, norm="forward")
    fft_shifted = torch.fft.fftshift(fft_complex, dim=(-2, -1))

    magnitude = torch.abs(fft_shifted)
    normalized_mag = _normalize_log_fft(magnitude)

    angle = torch.angle(fft_shifted)
    # Angle is strictly in [-pi, pi]. Mathematically map to [0, 1] without dynamic reduction
    normalized_angle = (angle + torch.pi) / (2.0 * torch.pi)

    return torch.cat([normalized_mag, normalized_angle], dim=1)


def two_domain_input_features(tensor: torch.Tensor) -> torch.Tensor:
    """Input: (B, 3, H, W) -> Output: (B, 3, H, W)
    Channel 0: Spatial Intensity (Grayscale)
    Channel 1: Log-normalized Magnitude
    Channel 2: Normalized Angle
    """
    gray = tvF.rgb_to_grayscale(tensor, num_output_channels=1)
    fft_complex = torch.fft.fft2(gray, norm="forward")
    fft_shifted = torch.fft.fftshift(fft_complex, dim=(-2, -1))

    magnitude = torch.abs(fft_shifted)
    normalized_mag = _normalize_log_fft(magnitude)

    angle = torch.angle(fft_shifted)
    normalized_angle = (angle + torch.pi) / (2.0 * torch.pi)

    return torch.cat([gray, normalized_mag, normalized_angle], dim=1)


class RGBModel(nn.Module):
    """RGB-only ablation (3 channels, from scratch)."""

    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "resnet18", pretrained=False, num_classes=1, in_chans=3, drop_rate=0.2
        )

    def forward(self, x):
        return self.model(x)


class DWTModel(nn.Module):
    """
    Optimized DWT ablation (4 subbands * 3 RGB channels = 12 channels).
    Feeds directly into a modified ResNet without destructive interpolation.
    """

    def __init__(self):
        super().__init__()
        # Normalizes the 12 DWT channels
        self.norm = nn.InstanceNorm2d(12, affine=True)

        # Create the base ResNet
        self.model = timm.create_model(
            "resnet18", pretrained=False, num_classes=1, in_chans=12, drop_rate=0.2
        )

        # FIX: Modify the ResNet stem.
        # Standard ResNet uses a 7x7 stride-2 conv to downsample 224 -> 112.
        # Since DWT already downsampled to 112, we replace the stem with a
        # 3x3 stride-1 conv. This preserves spatial dimensions and saves compute.
        self.model.conv1 = nn.Conv2d(
            12, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, x):
        # x shape: (B, 3, H, W) e.g., (B, 3, 224, 224)

        # Compute Wavelet Transform
        LL, (LH, HL, HH) = ptwt.wavedec2(x, wavelet="haar", level=1)

        # Concatenate subbands along the channel dimension
        # Output shape: (B, 12, H/2, W/2) e.g., (B, 12, 112, 112)
        dwt_all = torch.cat([LL, LH, HL, HH], dim=1)

        # Normalize and pass directly to the modified ResNet
        # No interpolation needed!
        return self.model(self.norm(dwt_all))


# ---------------------------------------------------------------------------
# New FFT Ablation variants
# ---------------------------------------------------------------------------


class RGBFFTModel(nn.Module):
    """3-Channel RGB Log-FFT ablation."""

    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "resnet18", pretrained=False, num_classes=1, in_chans=3, drop_rate=0.2
        )
        self.norm = nn.InstanceNorm2d(3, affine=True)

    def forward(self, x):
        with torch.no_grad():
            fft = rgb_fft(x)
        return self.model(self.norm(fft))


class GrayscaleFFTModel(nn.Module):
    """1-Channel Grayscale Log-FFT ablation (Improved)."""

    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "resnet18", pretrained=False, num_classes=1, in_chans=1, drop_rate=0.2
        )
        self.norm = nn.InstanceNorm2d(1, affine=True)

    def forward(self, x):
        with torch.no_grad():
            fft = grayscale_fft(x)
        return self.model(self.norm(fft))


class FourierDomainModel(nn.Module):
    """2-Channel Fourier Domain (Magnitude + Phase) ablation."""

    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "resnet18", pretrained=False, num_classes=1, in_chans=2, drop_rate=0.2
        )
        self.norm = nn.InstanceNorm2d(2, affine=True)

    def forward(self, x):
        with torch.no_grad():
            features = fourier_domain_only_features(x)
        return self.model(self.norm(features))


class TwoDomainModel(nn.Module):
    """3-Channel Two Domain (Spatial Intensity + Magnitude + Phase) ablation."""

    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "resnet18", pretrained=False, num_classes=1, in_chans=3, drop_rate=0.2
        )
        self.norm = nn.InstanceNorm2d(3, affine=True)

    def forward(self, x):
        with torch.no_grad():
            features = two_domain_input_features(x)
        return self.model(self.norm(features))


# Registry for model lookup by name
MODEL_REGISTRY = {
    "rgb": RGBModel,
    "dwt": DWTModel,
    "rgb_fft": RGBFFTModel,
    "grayscale_fft": GrayscaleFFTModel,
    "fourier_domain": FourierDomainModel,
    "two_domain": TwoDomainModel,
}
