import matplotlib.pyplot as plt
from src.dataset.vsi_prep.preprocess import load_master_index
from src.dataset.vsi_prep.filter_outliers import fit_plane_robust
from pathlib import Path


def plot_multi_sample_check(slide_name, dataset_name="ZStack_HE"):
    master = load_master_index(dataset_name, 224)
    slide = next(s for s in master.file_registry if s.name == slide_name)

    patches = slide.patches
    X, Y, Z = patches[:, 0], patches[:, 1], patches[:, 2]

    # Global Fit
    params_global, errors_global = fit_plane_robust(X, Y, Z, threshold=4.0)

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # 1. Spatial layout colored by Z (Absolute)
    sc1 = ax1.scatter(X, Y, c=Z, cmap="viridis", s=5)
    plt.colorbar(sc1, ax=ax1, label="Actual Z level")
    ax1.set_title(f"Absolute Focus Levels: {slide_name}")
    ax1.invert_yaxis()

    # 2. Spatial layout colored by Error (Difference from Plane)
    sc2 = ax2.scatter(X, Y, c=errors_global, vmin=0, vmax=10, cmap="Reds", s=5)
    plt.colorbar(sc2, ax=ax2, label="Error from Global Plane")
    ax2.set_title("Errors relative to ONE Global Plane")
    ax2.invert_yaxis()

    out_dir = Path("visualizations/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slide_name}_multi_sample_check.png"
    plt.savefig(out_path)
    print(f"Diagnostic plot saved to {out_path}")


if __name__ == "__main__":
    # Slide 072 often has multiple sections
    plot_multi_sample_check("072_All_HE_stack.vsi")
