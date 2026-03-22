import torch
import torch.nn as nn


class SyntheticConvModel(nn.Module):
    """
    A single depthwise Conv2d model for supervised patch-to-patch training.
    Relation: PatchIn - PatchOut = k - 1
    """

    def __init__(self, kernel_size: int, groups: int = 3, weight_init: str = "uniform"):
        super().__init__()
        # groups=3: Depthwise (independent RGB)
        # groups=1: Full (mixed RGB)
        self.conv = nn.Conv2d(3, 3, kernel_size=kernel_size, groups=groups, bias=False)

        if weight_init == "uniform":
            # Initialize with a uniform flat kernel (1.0 / area)
            # This provides a stable starting point (mean/blob) for the optimizer.
            nn.init.constant_(self.conv.weight, 1.0 / (kernel_size * kernel_size))
        elif weight_init == "ident":
            # Identity: Delta at center (k // 2, k // 2)
            nn.init.zeros_(self.conv.weight)
            center = kernel_size // 2
            with torch.no_grad():
                for i in range(3):
                    if groups == 3:
                        self.conv.weight[i, 0, center, center] = 1.0
                    else:
                        self.conv.weight[i, i, center, center] = 1.0
        elif weight_init == "random":
            # Normal distribution with mean 1/k^2
            mu = 1.0 / (kernel_size * kernel_size)
            nn.init.normal_(self.conv.weight, mean=mu, std=mu * 0.1)
        else:
            raise ValueError(f"Unknown weight_init: {weight_init}")

    def forward(self, x):
        return self.conv(x)
