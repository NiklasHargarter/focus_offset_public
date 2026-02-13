import numpy as np
from histolab.filters.image_filters import Compose, OtsuThreshold, RgbToGrayscale
from histolab.filters.morphological_filters import (
    BinaryDilation,
    RemoveSmallHoles,
    RemoveSmallObjects,
)
from PIL import Image

_tissue_pipeline = Compose(
    [
        RgbToGrayscale(),
        OtsuThreshold(),
        BinaryDilation(disk_size=5),
        RemoveSmallHoles(area_threshold=3000),
        RemoveSmallObjects(min_size=3000),
    ]
)


def detect_tissue_mask(
    *,
    thumbnail_rgb: np.ndarray,
) -> np.ndarray:
    """Generate a binary tissue mask from a downsampled RGB thumbnail.

    Args:
        thumbnail_rgb: Downsampled RGB image (H, W, 3) uint8.

    Returns:
        mask as a uint8 binary image (H, W) with tissue=255.
    """
    if thumbnail_rgb is None or thumbnail_rgb.size == 0:
        raise ValueError("thumbnail_rgb must be a non-empty RGB image")

    if thumbnail_rgb.ndim != 3 or thumbnail_rgb.shape[2] != 3:
        raise ValueError("thumbnail_rgb must be an RGB image (H, W, 3)")

    pil_img = Image.fromarray(thumbnail_rgb)
    bool_mask = _tissue_pipeline(pil_img)
    return (bool_mask * 255).astype(np.uint8)