import numpy as np
from src.dataset.vsi_prep.preprocess import load_master_index


def fit_plane_robust(X, Y, Z, iterations=3, threshold=3.0):
    mask = np.ones_like(Z, dtype=bool)
    for _ in range(iterations):
        if np.sum(mask) < 4:
            break
        A = np.column_stack([X[mask], Y[mask], np.ones_like(X[mask])])
        params, _, _, _ = np.linalg.lstsq(A, Z[mask], rcond=None)

        Z_pred = np.column_stack([X, Y, np.ones_like(X)]) @ params
        errors = np.abs(Z - Z_pred)
        mask = errors < threshold

    return params, errors


def analyze_tilt(dataset_name="ZStack_HE", patch_size=224):
    master = load_master_index(dataset_name, patch_size)
    if not master:
        print("Index not found or failed to load.")
        return

    print(f"Analyzing {dataset_name} for tilt outliers (Robust Fit)...")

    total_removed = 0
    total_before = 0
    threshold = 4.0  # Slices

    for slide in master.file_registry:
        patches = slide.patches
        if len(patches) < 10:
            continue

        total_before += len(patches)
        X = patches[:, 0]
        Y = patches[:, 1]
        Z = patches[:, 2]

        params, errors = fit_plane_robust(X, Y, Z, iterations=5, threshold=threshold)

        outliers = errors > threshold
        num_outliers = np.sum(outliers)

        total_removed += num_outliers

        if num_outliers > 0.3 * len(patches):
            print(f"Slide: {slide.name} - High outlier count!")
            print(f"  Total patches: {len(patches)}")
            print(f"  Mean error: {np.mean(errors):.2f}, Std: {np.std(errors):.2f}")
            print(
                f"  Outliers (> {threshold}): {num_outliers} ({num_outliers / len(patches) * 100:.1f}%)"
            )
        elif num_outliers > 0:
            # print(f"Slide: {slide.name} - {num_outliers} outliers")
            pass

    print(f"\nSummary (Threshold={threshold}):")
    print(f"  Total patches before: {total_before}")
    print(
        f"  Total outliers found: {total_removed} ({total_removed / total_before * 100:.2f}%)"
    )


if __name__ == "__main__":
    analyze_tilt()
