import ptwt
import numpy as np
import timm
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from skimage.color import separate_stains, hed_from_rgb

# ---------------------------------------------------------------------------
# Transformation functions for WSI focus prediction
# ---------------------------------------------------------------------------

_RADII_CACHE = {}


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
    normalized = torch.where(
        range_val > 0, (log_mag - min_val) / range_val, torch.zeros_like(log_mag)
    )
    return normalized


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


def hematoxylin_fft(tensor: torch.Tensor) -> torch.Tensor:
    """Input: (B, 3, H, W) -> Output: (B, 1, H, W)"""
    # Exact batch expectation logic
    results = []
    for t in tensor:
        clamped = torch.clamp(t, 0.0, 1.0)
        np_patch = clamped.permute(1, 2, 0).cpu().numpy()
        hed_patch = separate_stains(np_patch, hed_from_rgb)
        h_channel = torch.from_numpy(hed_patch[:, :, 0]).float().unsqueeze(0)
        results.append(h_channel.to(tensor.device))

    h_channels = torch.stack(results)
    fft_complex = torch.fft.fft2(h_channels, norm="forward")
    fft_shifted = torch.fft.fftshift(fft_complex, dim=(-2, -1))
    magnitude = torch.abs(fft_shifted)
    return _normalize_log_fft(magnitude)


def radial_profile_1d(tensor: torch.Tensor) -> torch.Tensor:
    """Input: (B, 3, H, W) -> Output: (B, 1, R)"""
    gray = tvF.rgb_to_grayscale(tensor, num_output_channels=1)
    fft_complex = torch.fft.fft2(gray, norm="forward")
    fft_shifted = torch.fft.fftshift(fft_complex, dim=(-2, -1))
    magnitudes = torch.abs(fft_shifted).squeeze(1)  # (B, H, W)

    b, h, w = magnitudes.shape
    device = tensor.device
    cache_key = (h, w, device)

    if cache_key not in _RADII_CACHE:
        center_y, center_x = h // 2, w // 2
        y, x = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        radii = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2).long().flatten()
        _RADII_CACHE[cache_key] = (radii, center_x, center_y)

    radii_flat, center_x, center_y = _RADII_CACHE[cache_key]
    max_radius = min(center_x, center_y)

    profiles = []
    for i in range(b):
        mag_flat = magnitudes[i].flatten()
        ring_sums = torch.bincount(radii_flat, weights=mag_flat)
        ring_counts = torch.bincount(radii_flat)

        radial_mean = torch.where(
            ring_counts > 0, ring_sums / ring_counts, torch.zeros_like(ring_sums)
        )
        radial_profile = radial_mean[:max_radius]
        profiles.append(radial_profile.unsqueeze(0))

    profiles_tensor = torch.stack(profiles)
    return _normalize_log_fft(profiles_tensor)  # (B, 1, Max_Radius)


class RGBModel(nn.Module):
    """RGB-only ablation (3 channels, from scratch)."""

    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "resnet18", pretrained=False, num_classes=1, in_chans=3
        )

    def forward(self, x):
        return self.model(x)


class DWTModel(nn.Module):
    """DWT-only ablation (3 channels, from scratch)."""

    def __init__(self):
        super().__init__()
        self.norm = nn.InstanceNorm2d(3, affine=True)
        self.model = timm.create_model(
            "resnet18", pretrained=False, num_classes=1, in_chans=3
        )

    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)
        target_size = x.shape[-2:]
        _, (LH, HL, HH) = ptwt.wavedec2(gray, wavelet="haar", level=1)
        dwt_details = torch.cat([LH, HL, HH], dim=1)
        dwt_up = F.interpolate(
            dwt_details, size=target_size, mode="bilinear", align_corners=False
        )
        return self.model(self.norm(dwt_up))


# ---------------------------------------------------------------------------
# New FFT Ablation variants
# ---------------------------------------------------------------------------


class RGBFFTModel(nn.Module):
    """3-Channel RGB Log-FFT ablation."""

    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "resnet18", pretrained=False, num_classes=1, in_chans=3
        )
        self.norm = nn.InstanceNorm2d(3, affine=True)

    def forward(self, x):
        fft = rgb_fft(x)
        return self.model(self.norm(fft))


class GrayscaleFFTModel(nn.Module):
    """1-Channel Grayscale Log-FFT ablation (Improved)."""

    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "resnet18", pretrained=False, num_classes=1, in_chans=1
        )
        self.norm = nn.InstanceNorm2d(1, affine=True)

    def forward(self, x):
        fft = grayscale_fft(x)
        return self.model(self.norm(fft))


class HematoxylinFFTModel(nn.Module):
    """1-Channel Hematoxylin (Color Deconvolution) Log-FFT ablation."""

    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "resnet18", pretrained=False, num_classes=1, in_chans=1
        )
        self.norm = nn.InstanceNorm2d(1, affine=True)

    def forward(self, x):
        fft = hematoxylin_fft(x)
        return self.model(self.norm(fft))


class RadialProfileModel(nn.Module):
    """1D Radial Profile ablation (using 1D CNN)."""

    def __init__(self):
        super().__init__()
        # 1D Radial Profile output is (1, 1, 112) for 224x224 input
        self.model = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        profiles = radial_profile_1d(x)
        return self.model(profiles)


# Registry for model lookup by name
MODEL_REGISTRY = {
    "rgb": RGBModel,
    "dwt": DWTModel,
    "rgb_fft": RGBFFTModel,
    "grayscale_fft": GrayscaleFFTModel,
    "hematoxylin_fft": HematoxylinFFTModel,
    "radial_profile": RadialProfileModel,
}
