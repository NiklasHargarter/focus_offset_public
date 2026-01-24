import numpy as np
import matplotlib.pyplot as plt
import slideio
import cv2
from src import config
from src.dataset.vsi_prep.preprocess import load_master_index
from src.dataset.vsi_prep.filter_outliers import fit_plane_robust


def compare_focus_quality(slide_name, dataset_name="ZStack_HE"):
    master = load_master_index(dataset_name, 224)
    slide = next(s for s in master.file_registry if s.name == slide_name)

    patches = slide.patches
    X, Y, Z_labels = patches[:, 0], patches[:, 1], patches[:, 2]
    _, inlier_mask = fit_plane_robust(X, Y, Z_labels, threshold=4.0)

    outlier_idx = np.where(~inlier_mask)[0][0]  # First outlier
    inlier_idx = np.where(inlier_mask)[0][0]  # First inlier

    raw_path = config.get_vsi_raw_dir(dataset_name) / slide_name
    slide_obj = slideio.open_slide(str(raw_path), "VSI")
    scene = slide_obj.get_scene(0)
    num_z = scene.num_z_slices

    def get_focus_curve(idx):
        px, py = int(X[idx]), int(Y[idx])
        curve = []
        for z in range(num_z):
            # Read small patch
            # We use a 448x448 raw area for 224 output
            patch = scene.read_block(
                rect=(px, py, 448, 448), size=(224, 224), slices=(z, z + 1)
            )
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            score = np.sum(
                (gray[:, 2:].astype(float) - gray[:, :-2].astype(float)) ** 2
            )
            curve.append(score)
        return np.array(curve)

    print(f"Sampling curves for {slide_name}...")
    inlier_curve = get_focus_curve(inlier_idx)
    outlier_curve = get_focus_curve(outlier_idx)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(
        range(num_z), inlier_curve / inlier_curve.max(), "g-o", label="Inlier (Clean)"
    )
    ax1.axvline(
        Z_labels[inlier_idx],
        color="g",
        linestyle="--",
        label=f"Label: {Z_labels[inlier_idx]}",
    )
    ax1.set_title("Inlier: Clear Focus Peak")
    ax1.legend()

    ax2.plot(
        range(num_z),
        outlier_curve / outlier_curve.max(),
        "r-o",
        label="Outlier (Messy)",
    )
    ax2.axvline(
        Z_labels[outlier_idx],
        color="r",
        linestyle="--",
        label=f"Label: {Z_labels[outlier_idx]}",
    )
    ax2.set_title("Outlier: Ambiguous or Wrong Peak")
    ax2.legend()

    plt.savefig("visualizations/diagnostics/focus_curve_comparison.png")
    print("Graph saved to visualizations/diagnostics/focus_curve_comparison.png")


if __name__ == "__main__":
    compare_focus_quality("030_1_HE_stack.vsi")
