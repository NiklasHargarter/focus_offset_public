import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from synthetic_data.plot.plot_comparison import create_kernel_comparison_from_checkpoints
from synthetic_data.config import SyntheticConfig
from synthetic_data.scripts.train_synthetic import get_teacher_kernel


def _checkpoint_label(path: Path) -> str:
    if path.name == "initial_model.pt":
        return "init"
    if path.name == "best_model.pt":
        return "best"

    match = re.match(r"model_epoch_(\d+)\.pt", path.name)
    if match:
        return f"e{int(match.group(1))}"
    return path.stem


def _checkpoint_order(path: Path) -> int:
    if path.name == "initial_model.pt":
        return 0

    match = re.match(r"model_epoch_(\d+)\.pt", path.name)
    if match:
        return int(match.group(1))

    if path.name == "best_model.pt":
        return 10**9

    return 10**8


def _kernels_for_display(weight: torch.Tensor) -> np.ndarray:
    w = weight.detach().cpu().numpy()
    if w.shape[1] == 1:
        return np.stack([w[0, 0, :, :], w[1, 0, :, :], w[2, 0, :, :]], axis=0)
    return np.stack([w[0, 0, :, :], w[1, 1, :, :], w[2, 2, :, :]], axis=0)


def _list_progression_checkpoints(exp_dir: Path) -> list[Path]:
    checkpoints = [exp_dir / "initial_model.pt"]

    epoch_checkpoints = sorted(
        exp_dir.glob("model_epoch_*.pt"),
        key=_checkpoint_order,
    )
    checkpoints.extend(epoch_checkpoints)

    best_path = exp_dir / "best_model.pt"
    if best_path.exists():
        checkpoints.append(best_path)

    return [p for p in checkpoints if p.exists()]


def create_kernel_progression_for_experiment(
    exp_dir: Path,
    output_name: str = "kernel_progression.png",
):
    exp_dir = Path(exp_dir)
    checkpoints = _list_progression_checkpoints(exp_dir)

    if not checkpoints:
        print(f"No checkpoints found in: {exp_dir}")
        return None

    items: list[tuple[str, np.ndarray]] = []
    for cp in checkpoints:
        state_dict = torch.load(cp, map_location="cpu", weights_only=True)
        kernels = _kernels_for_display(state_dict["conv.weight"])
        items.append((_checkpoint_label(cp), kernels))

    vmax = max(float(np.abs(k).max()) for _, k in items)
    vmax = max(vmax, 1e-12)

    sns.set_theme(style="white")
    rows, cols = 3, len(items)
    fig, axes = plt.subplots(rows, cols, figsize=(2.8 * cols, 7.8))

    if cols == 1:
        axes = np.array(axes).reshape(3, 1)

    channel_names = ["R", "G", "B"]
    for c, (label, kernels) in enumerate(items):
        for r in range(3):
            ax = axes[r, c]
            sns.heatmap(
                kernels[r],
                ax=ax,
                cmap="viridis",
                center=0,
                vmin=-vmax,
                vmax=vmax,
                cbar=False,
                xticklabels=False,
                yticklabels=False,
            )
            if r == 0:
                ax.set_title(label, fontsize=10)
            if c == 0:
                ax.set_ylabel(channel_names[r], fontsize=10, rotation=0, labelpad=12)
            ax.axis("off")

    fig.suptitle(exp_dir.name, fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_path = exp_dir / output_name
    plt.savefig(save_path, dpi=180)
    plt.close(fig)
    print(f"Kernel progression saved to: {save_path}")
    return save_path


def create_suite_overview(suite_dir: Path):
    suite_dir = Path(suite_dir)
    exp_dirs = sorted([d for d in suite_dir.iterdir() if d.is_dir()])
    if not exp_dirs:
        print(f"No experiment directories found in: {suite_dir}")
        return

    # Per-init progression (init -> epochs -> best)
    for exp_dir in exp_dirs:
        create_kernel_progression_for_experiment(exp_dir)

    # Cross-init comparisons for key checkpoints
    create_kernel_comparison_from_checkpoints(
        suite_dir=suite_dir,
        checkpoint_name="initial_model.pt",
        output_name="suite_compare_initial.png",
    )

    create_kernel_comparison_from_checkpoints(
        suite_dir=suite_dir,
        checkpoint_name="best_model.pt",
        output_name="suite_compare_best.png",
    )

    epoch_numbers: set[int] = set()
    for exp_dir in exp_dirs:
        for cp in exp_dir.glob("model_epoch_*.pt"):
            match = re.match(r"model_epoch_(\d+)\.pt", cp.name)
            if match:
                epoch_numbers.add(int(match.group(1)))

    for epoch in sorted(epoch_numbers):
        cp_name = f"model_epoch_{epoch}.pt"
        create_kernel_comparison_from_checkpoints(
            suite_dir=suite_dir,
            checkpoint_name=cp_name,
            output_name=f"suite_compare_{cp_name.replace('.pt', '.png')}",
        )

    create_teacher_checkpoint_comparison(
        suite_dir=suite_dir,
        checkpoint_name="initial_model.pt",
        output_name="suite_compare_teacher_initial.png",
    )
    create_teacher_checkpoint_comparison(
        suite_dir=suite_dir,
        checkpoint_name="best_model.pt",
        output_name="suite_compare_teacher_best.png",
    )


def create_teacher_checkpoint_comparison(
    suite_dir: Path,
    checkpoint_name: str,
    output_name: str,
):
    suite_dir = Path(suite_dir)
    exp_dirs = sorted([d for d in suite_dir.iterdir() if d.is_dir()])
    if not exp_dirs:
        print(f"No experiment directories found in: {suite_dir}")
        return

    config = SyntheticConfig()
    teacher = get_teacher_kernel(
        k_size=config.kernel_size,
        radius=config.simulation_radius,
        device="cpu",
    )

    items: list[tuple[str, np.ndarray]] = [("teacher", _kernels_for_display(teacher))]

    short_label = (
        "initial" if checkpoint_name == "initial_model.pt" else checkpoint_name
    )

    for exp_dir in exp_dirs:
        cp_path = exp_dir / checkpoint_name
        if not cp_path.exists():
            continue
        state_dict = torch.load(cp_path, map_location="cpu", weights_only=True)
        kernels = _kernels_for_display(state_dict["conv.weight"])
        label = exp_dir.name.split("_")[-1]
        items.append((f"{short_label} ({label})", kernels))

    if len(items) <= 1:
        print(f"No checkpoints named '{checkpoint_name}' found in: {suite_dir}")
        return

    sns.set_theme(style="white")
    rows, cols = 3, len(items)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 7.8))

    if cols == 1:
        axes = np.array(axes).reshape(3, 1)

    channel_names = ["R", "G", "B"]
    for c, (label, kernels) in enumerate(items):
        for r in range(3):
            ax = axes[r, c]
            kernel = kernels[r]
            k_min = float(kernel.min())
            k_max = float(kernel.max())

            # Essential only: per-kernel full-range scaling.
            has_negative = k_min < 0
            if has_negative:
                lim = max(abs(k_min), abs(k_max), 1e-12)
                vmin, vmax = -lim, lim
                cmap = "coolwarm"
                center = 0
            else:
                vmin, vmax = k_min, max(k_max, k_min + 1e-12)
                cmap = "magma"
                center = None

            sns.heatmap(
                kernel,
                ax=ax,
                cmap=cmap,
                center=center,
                vmin=vmin,
                vmax=vmax,
                cbar=False,
                xticklabels=False,
                yticklabels=False,
            )
            if r == 0:
                ax.set_title(label, fontsize=10)
            if c == 0:
                ax.set_ylabel(channel_names[r], fontsize=10, rotation=0, labelpad=12)
            ax.axis("off")

    fig.suptitle(
        f"Teacher vs {short_label} Kernels", fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    save_path = suite_dir / output_name
    plt.savefig(save_path, dpi=180)
    plt.close(fig)
    print(f"Teacher-checkpoint comparison saved to: {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate simulation-focused kernel visualizations for a synthetic experiment suite."
    )
    parser.add_argument("suite_dir", type=str, help="Path to suite directory in logs/")
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir)
    if not suite_dir.exists():
        raise FileNotFoundError(f"Suite directory not found: {suite_dir}")

    create_suite_overview(suite_dir)


if __name__ == "__main__":
    main()
