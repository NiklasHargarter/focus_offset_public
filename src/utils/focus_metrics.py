import cv2
import numpy as np


def compute_brenner_gradient(image: np.ndarray) -> float:
    """
    Standard Brenner Gradient focus metric.

    Source: Brenner, J. F., et al. (1976). "An automated microscope for cytologic research."
    Formula: sum((I(x+2, y) - I(x, y))^2)

    Uses float64 internally to prevent overflow on large Super Patches or ROIs.
    """
    if image.size == 0:
        return 0.0

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    gray = gray.astype(np.float64)

    diff = gray[:, 2:] - gray[:, :-2]

    return float(np.sum(diff**2))
