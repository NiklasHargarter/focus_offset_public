"""
Visualize a kernel comparison suite.

Generates:
  1. Kernel heatmap grid (3 channels × N experiments)
  2. Radial profile overlay of all 4 kernels
  3. Training curves overlay

Usage:
  python -m synthetic_data.scripts.visualize_comparison <suite_dir>
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_kernel(checkpoint_path: Path) -> np.ndarray:
    """Load a checkpoint and return the green-channel kernel as 2D array."""
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    w = sd["conv.weight"]  # [3, 1, K, K] or [3, 3, K, K]
    if w.shape[1] == 1:
        return w[1, 0].numpy()  # green channel, depthwise
    return w[1, 1].numpy()  # green channel, full


def _load_all_channels(checkpoint_path: Path) -> list[np.ndarray]:
    """Load a checkpoint and return [R, G, B] kernels."""
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    w = sd["conv.weight"]  # [3, 1, K, K] or [3, 3, K, K]
    if w.shape[1] == 1:
        return [w[c, 0].numpy() for c in range(3)]
    return [w[c, c].numpy() for c in range(3)]


def _radial_profile(kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the radially-averaged profile from the kernel center."""
    k = kernel.shape[0]
    center = k // 2
    y, x = np.ogrid[:k, :k]
    r = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    r_int = r.astype(int)
    max_r = int(r_int.max()) + 1
    radii = np.arange(max_r)
    profile = np.array([kernel[r_int == ri].mean() for ri in radii])
    return radii, profile


def _parse_label(exp_name: str) -> str:
    """Extract a human-readable label like 'HE +10' from experiment dir name."""
    parts = exp_name.split("_")
    # Find dataset and offset from the naming convention:
    # conv_k63_off10_ZStack_HE_g3_random
    offset = ""
    dataset = ""
    for i, p in enumerate(parts):
        if p.startswith("off"):
            val = p[3:]
            offset = f"+{val}" if not val.startswith("-") else val
        if p == "ZStack" and i + 1 < len(parts):
            dataset = parts[i + 1]  # HE or IHC
    return f"{dataset} {offset}" if dataset else exp_name


def _discover_experiments(suite_dir: Path) -> list[tuple[str, Path]]:
    """Return sorted list of (label, exp_dir) for all experiments in a suite."""
    experiments = []
    for d in sorted(suite_dir.iterdir()):
        if d.is_dir() and (d / "best_model.pt").exists():
            label = _parse_label(d.name)
            experiments.append((label, d))
    return experiments


# ---------------------------------------------------------------------------
# Plot 1: Kernel Heatmap Grid
# ---------------------------------------------------------------------------

CHANNELS = ["R", "G", "B"]
CHANNEL_COLORS = ["#E24A33", "#228B22", "#348ABD"]


def plot_kernel_grid(suite_dir: Path, checkpoint: str = "best_model.pt"):
    """Grid of kernel heatmaps: rows = R/G/B channels, cols = experiments."""
    experiments = _discover_experiments(suite_dir)
    if not experiments:
        print("No experiments found.")
        return

    # Load all channels for each experiment
    exp_channels = [
        (label, _load_all_channels(d / checkpoint)) for label, d in experiments
    ]
    n_exp = len(exp_channels)

    sns.set_theme(style="white")
    fig, axes = plt.subplots(3, n_exp, figsize=(3.5 * n_exp, 9))
    if n_exp == 1:
        axes = axes.reshape(3, 1)

    for col, (label, chs) in enumerate(exp_channels):
        for row in range(3):
            ax = axes[row, col]
            kernel = chs[row]
            lim = max(float(np.abs(kernel).max()), 1e-12)
            ax.imshow(kernel, cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.axis("off")
            if row == 0:
                ax.set_title(label, fontsize=12, fontweight="bold")


    for row in range(3):
        bbox = axes[row, 0].get_position()
        fig.text(
            bbox.x0 - 0.02, bbox.y0 + bbox.height / 2,
            CHANNELS[row], fontsize=14, fontweight="bold",
            color=CHANNEL_COLORS[row], ha="right", va="center",
        )

    fig.suptitle("Learned Kernels", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0.03, 0, 1, 0.96])
    out = suite_dir / "kernel_grid.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2: Radial Profile Overlay
# ---------------------------------------------------------------------------

def plot_radial_profiles(suite_dir: Path, checkpoint: str = "best_model.pt"):
    """Overlay radial profiles of all kernels on one plot."""
    experiments = _discover_experiments(suite_dir)
    if not experiments:
        return

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = sns.color_palette("husl", len(experiments))
    for (label, d), color in zip(experiments, colors):
        kernel = _load_kernel(d / checkpoint)
        radii, profile = _radial_profile(kernel)
        ax.plot(radii, profile, label=label, color=color, linewidth=2)

    ax.set_xlabel("Radius (pixels)", fontsize=12)
    ax.set_ylabel("Kernel Weight", fontsize=12)
    ax.set_title("Radial Kernel Profiles", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    out = suite_dir / "radial_profiles.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3: Training Curves Overlay
# ---------------------------------------------------------------------------

def plot_training_curves(suite_dir: Path):
    """Overlay val_loss vs epoch for all experiments."""
    experiments = _discover_experiments(suite_dir)
    if not experiments:
        return

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))

    colors = sns.color_palette("husl", len(experiments))
    for (label, d), color in zip(experiments, colors):
        csv_path = d / "metrics_synthetic.csv"
        if not csv_path.exists():
            continue
        epochs, val_losses = [], []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                val_losses.append(float(row["val_loss"]))
        ax.plot(epochs, val_losses, label=label, color=color, linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.set_title("Training Curves", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    out = suite_dir / "training_curves.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def visualize_comparison(suite_dir: Path, checkpoint: str = "best_model.pt"):
    """Generate all comparison visualizations for a suite directory."""
    suite_dir = Path(suite_dir)
    if not suite_dir.exists():
        raise FileNotFoundError(f"Suite directory not found: {suite_dir}")

    print(f"Generating visualizations for: {suite_dir}\n")
    plot_kernel_grid(suite_dir, checkpoint)
    plot_radial_profiles(suite_dir, checkpoint)
    plot_training_curves(suite_dir)
    print(f"\nAll visualizations saved to: {suite_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison visualizations for a kernel comparison suite."
    )
    parser.add_argument("suite_dir", type=str, help="Path to the comparison suite directory")
    parser.add_argument(
        "--checkpoint", type=str, default="best_model.pt",
        help="Checkpoint name to use for kernel plots (default: best_model.pt)",
    )
    args = parser.parse_args()
    visualize_comparison(Path(args.suite_dir), args.checkpoint)


if __name__ == "__main__":
    main()
