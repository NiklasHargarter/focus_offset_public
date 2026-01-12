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

    # Conversion to float64 is critical to handle differences properly 
    # and to avoid overflow during summation of squared differences.
    gray = gray.astype(np.float64)

    # Standard implementation: compare pixel with its neighbor 2 pixels away.
    # We slice [:, 2:] and [:, :-2] to avoid wrap-around noise from np.roll.
    diff = gray[:, 2:] - gray[:, :-2]
    
    return float(np.sum(diff**2))


def calculate_stability_metrics(patches: list, width: int, height: int, patch_size: int):
    """
    Calculate Total Variation and Outlier Ratio for a Z-map.
    Useful for assessing the 'smoothness' of focus estimation across a slide.
    """
    if not patches:
        return 0.0, 0.0

    xs = sorted(list(set(p.x for p in patches)))
    ys = sorted(list(set(p.y for p in patches)))
    x_map = {x: i for i, x in enumerate(xs)}
    y_map = {y: j for j, y in enumerate(ys)}

    z_grid = np.full((len(ys), len(xs)), np.nan)
    for p in patches:
        z_grid[y_map[p.y], x_map[p.x]] = p.z

    diff_h = np.abs(np.diff(z_grid, axis=1))
    diff_v = np.abs(np.diff(z_grid, axis=0))

    tv = ((np.nanmean(diff_h) + np.nanmean(diff_v)) / 2 if diff_h.size > 0 and diff_v.size > 0 else 0.0)
    
    outliers = np.count_nonzero(diff_h > 2) + np.count_nonzero(diff_v > 2)
    total_diffs = diff_h.size + diff_v.size
    outlier_ratio = outliers / total_diffs if total_diffs > 0 else 0.0

    return float(tv), float(outlier_ratio)
