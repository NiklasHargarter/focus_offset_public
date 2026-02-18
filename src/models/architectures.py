import ptwt
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Ablation models (ResNet-18 from scratch, fair comparison)
#
# All variants start from identical random initialization so performance
# differences reflect the input signal, not pretraining advantage.
# ---------------------------------------------------------------------------


class RGBModel(nn.Module):
    """RGB-only ablation (3 channels, from scratch)."""

    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "resnet18", pretrained=False, num_classes=1, in_chans=3
        )

    def forward(self, x):
        return self.model(x)


class FFTModel(nn.Module):
    """FFT-only ablation (1 channel, from scratch)."""

    def __init__(self):
        super().__init__()
        self.norm = nn.InstanceNorm2d(1, affine=True)
        self.model = timm.create_model(
            "resnet18", pretrained=False, num_classes=1, in_chans=1
        )

    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)
        fft = torch.fft.fft2(gray)
        fft_mag = torch.log1p(torch.abs(fft))
        fft_mag = torch.fft.fftshift(fft_mag, dim=(-2, -1))
        return self.model(self.norm(fft_mag))


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


class MultimodalModel(nn.Module):
    """RGB + FFT + DWT ablation (7 channels, from scratch)."""

    def __init__(self):
        super().__init__()
        self.fft_norm = nn.InstanceNorm2d(1, affine=True)
        self.dwt_norm = nn.InstanceNorm2d(3, affine=True)
        self.model = timm.create_model(
            "resnet18", pretrained=False, num_classes=1, in_chans=7
        )

    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)
        target_size = x.shape[-2:]

        # FFT branch
        fft = torch.fft.fft2(gray)
        fft_mag = torch.log1p(torch.abs(fft))
        fft_mag = torch.fft.fftshift(fft_mag, dim=(-2, -1))

        # DWT branch
        _, (LH, HL, HH) = ptwt.wavedec2(gray, wavelet="haar", level=1)
        dwt_details = torch.cat([LH, HL, HH], dim=1)
        dwt_up = F.interpolate(
            dwt_details, size=target_size, mode="bilinear", align_corners=False
        )

        features = torch.cat([x, self.fft_norm(fft_mag), self.dwt_norm(dwt_up)], dim=1)
        return self.model(features)


# ---------------------------------------------------------------------------
# Production model (ConvNeXt V2 Nano pretrained, RGB only)
# ---------------------------------------------------------------------------


class RGBConvNeXtModel(nn.Module):
    """RGB-only production model (3 channels, pretrained ConvNeXt V2 Nano).

    Applies ImageNet normalization internally because the backbone was
    pretrained on ImageNet-normalised inputs.  All other models in this
    project train from scratch and expect stain-agnostic [0, 1] inputs.
    """

    # ImageNet statistics (registered as buffers so they follow .to(device))
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self):
        super().__init__()
        self.register_buffer("_img_mean", torch.tensor(self.MEAN).view(1, 3, 1, 1))
        self.register_buffer("_img_std", torch.tensor(self.STD).view(1, 3, 1, 1))
        self.model = timm.create_model(
            "convnextv2_nano.fcmae_ft_in22k_in1k",
            pretrained=True,
            num_classes=1,
            in_chans=3,
            drop_rate=0.1,
        )

    def forward(self, x):
        x = (x - self._img_mean) / self._img_std
        return self.model(x)


# Registry for model lookup by name
MODEL_REGISTRY = {
    "rgb": RGBModel,
    "fft": FFTModel,
    "dwt": DWTModel,
    "multimodal": MultimodalModel,
    "rgb_convnext": RGBConvNeXtModel,
}
