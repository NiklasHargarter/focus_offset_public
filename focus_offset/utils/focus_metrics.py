import cv2
import numpy as np


def compute_focus_score(image: np.ndarray) -> float:
    """
    Compute Focus Score using the Variance of Laplacian metric.

    This is a fast, robust, and industry-standard method for measuring
    image sharpness/focus. Returns a higher score for sharper images.
    """
    if image.size == 0:
        return 0.0

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Laplacian of the image, then compute its variance
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
