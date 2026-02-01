import torch.nn as nn
from abc import ABC, abstractmethod
import torch.nn.functional as F
import torch
import timm


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


def haar_dwt(x):
    """
    Simple 2D Haar DWT implementation in PyTorch.
    x: (B, C, H, W)
    Returns: (B, C*4, H/2, W/2) corresponding to LL, LH, HL, HH
    """
    b, c, h, w = x.shape

    # Reshape to (b, c, h/2, 2, w/2, 2)
    # Pad if dimensions are odd
    if h % 2 != 0 or w % 2 != 0:
        x = F.pad(x, (0, w % 2, 0, h % 2))
        b, c, h, w = x.shape

    x_eshaped = x.view(b, c, h // 2, 2, w // 2, 2)

    # Average and difference along rows (last dim)
    x0 = x_eshaped[..., 0]
    x1 = x_eshaped[..., 1]
    L = x0 + x1
    H = x0 - x1

    # Now columns
    LL = L[:, :, :, 0, :] + L[:, :, :, 1, :]
    LH = L[:, :, :, 0, :] - L[:, :, :, 1, :]
    HL = H[:, :, :, 0, :] + H[:, :, :, 1, :]
    HH = H[:, :, :, 0, :] - H[:, :, :, 1, :]

    # Stack channels: (b, c, 4, h/2, w/2) -> (b, c*4, h/2, w/2)
    out = torch.stack([LL, LH, HL, HH], dim=2)
    out = out.reshape(b, c * 4, h // 2, w // 2)

    return out * 0.5


class InputTransformLayer(nn.Module):
    def __init__(self, mode: str = "all"):
        """
        Selective input transformation layer.
        mode: 'fft', 'dwt', or 'all'
        """
        super().__init__()
        self.mode = mode

    def forward(self, x):
        # x: (B, 3, H, W)
        # Convert to grayscale for transforms (B, 1, H, W)
        gray = x.mean(dim=1, keepdim=True)
        features = [x]

        # 1. FFT
        if self.mode in ["all", "fft"]:
            fft = torch.fft.fft2(gray)
            fft_mag = torch.log1p(torch.abs(fft))
            fft_mag = torch.fft.fftshift(fft_mag, dim=(-2, -1))

            # FFT Normalization (Empirical values from diagnostics)
            # Shift and scale to match RGB distribution (roughly mean 0, std 1)
            fft_mag = (fft_mag - 4.5) / 1.0
            features.append(fft_mag)

        # 2. DWT
        if self.mode in ["all", "dwt"]:
            dwt = haar_dwt(gray)
            dwt_details = dwt[:, 1:4]  # Take LH, HL, HH

            # DWT Normalization (Details are already zero-centered, but std is ~0.5)
            dwt_details = dwt_details * 2.0

            # Upsample DWT to match input resolution for concatenation
            dwt_up = F.interpolate(dwt_details, size=x.shape[-2:], mode="nearest")
            features.append(dwt_up)

        # Concatenate spatial and enabled frequency features
        return torch.cat(features, dim=1)


# ... (keep other classes) ...


class ConvNeXtV2FocusRegressor(BaseFocusRegressor):
    """
    State-of-the-art ConvNeXt V2 (with GRN).
    Supports multiple ablation modes: 'none', 'fft', 'dwt', 'all'.
    """

    def __init__(self, pretrained: bool = False, transform_mode: str = "all"):
        super().__init__()
        self.transform_mode = transform_mode

        # Number of input channels depends on the mode
        if transform_mode == "none":
            in_chans = 3
        elif transform_mode == "fft":
            in_chans = 4  # RGB (3) + FFT (1)
        elif transform_mode == "dwt":
            in_chans = 6  # RGB (3) + DWT (3)
        elif transform_mode == "all":
            in_chans = 7  # RGB (3) + FFT (1) + DWT (3)
        else:
            raise ValueError(f"Unknown transform_mode: {transform_mode}")

        # We train from scratch (pretrained=False) to simplify the integration
        # of non-RGB channels and better adapt to the microscopy domain.
        self.model = timm.create_model(
            "convnextv2_tiny",
            pretrained=False,
            num_classes=1,
            in_chans=in_chans,
        )

        if self.transform_mode != "none":
            self.transform_layer = InputTransformLayer(mode=self.transform_mode)

    def forward(self, x):
        if self.transform_mode != "none":
            x = self.transform_layer(x)
        return self.model(x)
