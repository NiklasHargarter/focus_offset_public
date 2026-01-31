import torch.nn as nn
from abc import ABC, abstractmethod
import torch.nn.functional as F
import torch


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
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # x: (B, 3, H, W)
        
        # Convert to grayscale for transforms (B, 1, H, W)
        # Using simple average or Rec. 601 coefficients would be fine. 
        # Weighted average (0.299R + 0.587G + 0.114B) is standard for perception,
        # but simple mean is sufficient for structural focus.
        gray = x.mean(dim=1, keepdim=True)
        
        # 1. FFT
        # Compute 2D FFT, take log magnitude
        fft = torch.fft.fft2(gray)
        fft_mag = torch.log1p(torch.abs(fft))
        # Shift so zero freq is in center
        fft_mag = torch.fft.fftshift(fft_mag, dim=(-2, -1))
        # (B, 1, H, W)
        
        # 2. DWT
        # (B, 4, H/2, W/2) corresponding to LL, LH, HL, HH
        dwt = haar_dwt(gray)
        
        # We want 3 channels for DWT. Usually high-frequency details are most relevant for focus.
        # LL is low-res approximation (redundant with RGB).
        # LH, HL, HH are the details.
        # dwt[:, 0:1] is LL, dwt[:, 1:2] is LH, etc.
        dwt_details = dwt[:, 1:4] # Take LH, HL, HH
        
        # Upsample DWT to match input resolution for concatenation
        dwt_up = F.interpolate(dwt_details, size=x.shape[-2:], mode='nearest')
        
        # Concatenate: 3 (RGB) + 1 (FFT) + 3 (DWT) = 7 channels
        return torch.cat([x, fft_mag, dwt_up], dim=1)


import timm

# ... (keep other classes) ...

class ConvNeXtV2FocusRegressor(BaseFocusRegressor):
    """
    State-of-the-art ConvNeXt V2 (with GRN).
    Uses `timm` implementation.
    Includes multi-domain input (Spatial + FFT + DWT).
    Hardcoded to 'tiny' version.
    """

    def __init__(self, pretrained: bool = True, use_transforms: bool = True):
        super().__init__()
        self.use_transforms = use_transforms
        
        # Load ConvNeXt V2 Tiny from timm
        # 'convnextv2_tiny.fcmae_ft_in1k' is the standard fine-tuned checkpoint
        model_name = "convnextv2_tiny.fcmae_ft_in1k" if pretrained else "convnextv2_tiny"
        
        # Initialize model with 1 output class (regression)
        # timm handles the head replacement automatically
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=1
        )

        # Modify first layer if using transforms
        if self.use_transforms:
            self.transform_layer = InputTransformLayer()
            
            # In timm, the stem is usually model.stem[0] for ConvNeXt
            # Let's inspect: ConvNeXtV2 uses a stem which is a Conv2d
            # Accessing the first conv layer:
            stem_conv = self.model.stem[0]
            
            # New input channels: 3 (RGB) + 1 (FFT) + 3 (DWT) = 7
            new_in_channels = 7
            
            new_conv = nn.Conv2d(
                new_in_channels,
                stem_conv.out_channels,
                kernel_size=stem_conv.kernel_size,
                stride=stem_conv.stride,
                padding=stem_conv.padding,
                bias=stem_conv.bias is not None
            )
            
            # Initialize weights
            with torch.no_grad():
                # Copy original spatial weights to the first 3 channels
                new_conv.weight[:, :3] = stem_conv.weight
                # Initialize the rest with Kaiming Normal
                nn.init.kaiming_normal_(new_conv.weight[:, 3:], mode='fan_out', nonlinearity='relu')
                
                if stem_conv.bias is not None:
                    new_conv.bias = stem_conv.bias
            
            self.model.stem[0] = new_conv

    def forward(self, x):
        if self.use_transforms:
            x = self.transform_layer(x)
        return self.model(x)
