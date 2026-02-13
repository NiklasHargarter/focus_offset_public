import numpy as np


def generate_grid(
    width: int,
    height: int,
    patch_size: int,
    stride: int,
) -> list[tuple[int, int]]:
    """Return all (x, y) top-left positions on a regular stride grid."""
    xs = range(0, width - patch_size + 1, stride)
    ys = range(0, height - patch_size + 1, stride)
    return [(x, y) for y in ys for x in xs]


def filter_by_tissue_coverage(
    positions: list[tuple[int, int]],
    mask: np.ndarray,
    patch_size: int,
    min_coverage: float,
    mask_downscale: int,
) -> list[tuple[int, int]]:
    """Keep only positions whose tissue coverage meets the threshold.

    Uses an integral image for O(1) coverage lookup per position.
    """
    m_h, m_w = mask.shape
    if m_h == 0 or m_w == 0 or not positions:
        return []

    mw = max(1, patch_size // mask_downscale)
    mh = max(1, patch_size // mask_downscale)

    binary = (mask > 0).astype(np.float64)
    integral = np.zeros((m_h + 1, m_w + 1), dtype=np.float64)
    np.cumsum(binary, axis=0, out=integral[1:, 1:])
    np.cumsum(integral[1:, 1:], axis=1, out=integral[1:, 1:])

    accepted: list[tuple[int, int]] = []
    for x_raw, y_raw in positions:
        mx = x_raw // mask_downscale
        my = y_raw // mask_downscale
        x1 = min(mx + mw, m_w)
        y1 = min(my + mh, m_h)
        if x1 <= mx or y1 <= my:
            continue

        tissue_sum = (
            integral[y1, x1]
            - integral[my, x1]
            - integral[y1, mx]
            + integral[my, mx]
        )
        area = (y1 - my) * (x1 - mx)
        if tissue_sum / area >= min_coverage:
            accepted.append((x_raw, y_raw))

    return accepted
