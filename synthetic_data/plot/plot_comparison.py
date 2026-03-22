from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F

from synthetic_data.model import SyntheticConvModel


def _load_model_from_checkpoint(cp_path: Path) -> SyntheticConvModel:
    state_dict = torch.load(cp_path, map_location="cpu", weights_only=True)
    weight = state_dict["conv.weight"]
    kernel_size = weight.shape[-1]
    groups = 3 if weight.shape[1] == 1 else 1

    model = SyntheticConvModel(kernel_size=kernel_size, groups=groups)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _kernels_for_display(weight: torch.Tensor) -> np.ndarray:
    w = weight.detach().cpu().numpy()
    if w.shape[1] == 1:
        return np.stack([w[0, 0, :, :], w[1, 0, :, :], w[2, 0, :, :]], axis=0)
    return np.stack([w[0, 0, :, :], w[1, 1, :, :], w[2, 2, :, :]], axis=0)


def _plot_image_with_seaborn(ax, image: np.ndarray, title: str):
    if image.ndim == 3:
        image_2d = image.mean(axis=2)
    else:
        image_2d = image

    sns.heatmap(
        image_2d,
        ax=ax,
        cmap="mako",
        vmin=0.0,
        vmax=1.0,
        cbar=False,
        xticklabels=False,
        yticklabels=False,
    )
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def create_kernel_comparison_from_checkpoints(
    suite_dir: Path,
    checkpoint_name: str,
    output_name: str,
    fixed_vmax: float | None = None,
):
    suite_dir = Path(suite_dir)
    exp_dirs = sorted([d for d in suite_dir.iterdir() if d.is_dir()])

    items: list[tuple[str, np.ndarray]] = []
    for exp_dir in exp_dirs:
        cp_path = exp_dir / checkpoint_name
        if not cp_path.exists():
            continue
        model = _load_model_from_checkpoint(cp_path)
        kernels = _kernels_for_display(model.conv.weight)
        label = exp_dir.name.split("_")[-1]
        items.append((label, kernels))

    if not items:
        print(f"No checkpoints named '{checkpoint_name}' in {suite_dir}")
        return

    channel_names = ["R", "G", "B"]
    rows = len(items)
    cols = 3
    sns.set_theme(style="white")

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for r, (label, kernels) in enumerate(items):
        for c in range(cols):
            if fixed_vmax is None:
                sns.heatmap(kernels[c], ax=axes[r, c], cmap="viridis", center=0)
            else:
                sns.heatmap(
                    kernels[c],
                    ax=axes[r, c],
                    cmap="viridis",
                    center=0,
                    vmin=-fixed_vmax,
                    vmax=fixed_vmax,
                )
            axes[r, c].set_title(
                f"{label} {channel_names[c]} ({checkpoint_name})", fontsize=10
            )
            axes[r, c].axis("off")

    plt.tight_layout()
    save_path = suite_dir / output_name
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Kernel comparison saved to: {save_path}")


def create_kernel_delta_comparison_from_checkpoints(
    suite_dir: Path,
    checkpoint_name: str,
    output_name: str,
    config,
    fixed_vmax: float | None = None,
):
    from synthetic_data.scripts.train_synthetic import get_teacher_kernel

    suite_dir = Path(suite_dir)
    exp_dirs = sorted([d for d in suite_dir.iterdir() if d.is_dir()])

    teacher = get_teacher_kernel(
        k_size=config.kernel_size,
        radius=config.simulation_radius,
        device="cpu",
    )

    items: list[tuple[str, np.ndarray]] = []
    for exp_dir in exp_dirs:
        cp_path = exp_dir / checkpoint_name
        if not cp_path.exists():
            continue
        model = _load_model_from_checkpoint(cp_path)
        delta = _kernels_for_display(model.conv.weight - teacher)
        label = exp_dir.name.split("_")[-1]
        items.append((label, delta))

    if not items:
        print(f"No checkpoints named '{checkpoint_name}' in {suite_dir}")
        return

    channel_names = ["R", "G", "B"]
    rows = len(items)
    cols = 3
    sns.set_theme(style="white")

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    vmax = max(float(np.abs(delta).max()) for _, delta in items)
    if fixed_vmax is not None:
        vmax = fixed_vmax
    vmax = max(vmax, 1e-12)

    for r, (label, delta) in enumerate(items):
        for c in range(cols):
            sns.heatmap(
                delta[c],
                ax=axes[r, c],
                cmap="coolwarm",
                center=0,
                vmin=-vmax,
                vmax=vmax,
            )
            axes[r, c].set_title(
                f"Δ {label} {channel_names[c]} ({checkpoint_name})", fontsize=10
            )
            axes[r, c].axis("off")

    plt.tight_layout()
    save_path = suite_dir / output_name
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Kernel delta comparison saved to: {save_path}")


def create_kernel_step_delta_from_checkpoints(
    suite_dir: Path,
    checkpoint_name: str,
    prev_checkpoint_name: str,
    output_name: str,
    fixed_vmax: float | None = None,
):
    suite_dir = Path(suite_dir)
    exp_dirs = sorted([d for d in suite_dir.iterdir() if d.is_dir()])

    items: list[tuple[str, np.ndarray]] = []
    for exp_dir in exp_dirs:
        cur_path = exp_dir / checkpoint_name
        prev_path = exp_dir / prev_checkpoint_name
        if not cur_path.exists() or not prev_path.exists():
            continue

        cur_model = _load_model_from_checkpoint(cur_path)
        prev_model = _load_model_from_checkpoint(prev_path)

        delta = _kernels_for_display(cur_model.conv.weight - prev_model.conv.weight)
        label = exp_dir.name.split("_")[-1]
        items.append((label, delta))

    if not items:
        print(
            f"No checkpoint pairs found for '{prev_checkpoint_name}' -> '{checkpoint_name}' in {suite_dir}"
        )
        return

    channel_names = ["R", "G", "B"]
    rows = len(items)
    cols = 3
    sns.set_theme(style="white")

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    vmax = max(float(np.abs(delta).max()) for _, delta in items)
    if fixed_vmax is not None:
        vmax = fixed_vmax
    vmax = max(vmax, 1e-12)

    for r, (label, delta) in enumerate(items):
        for c in range(cols):
            sns.heatmap(
                delta[c],
                ax=axes[r, c],
                cmap="coolwarm",
                center=0,
                vmin=-vmax,
                vmax=vmax,
            )
            axes[r, c].set_title(
                f"Δstep {label} {channel_names[c]} ({prev_checkpoint_name}→{checkpoint_name})",
                fontsize=9,
            )
            axes[r, c].axis("off")

    plt.tight_layout()
    save_path = suite_dir / output_name
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Kernel step-delta comparison saved to: {save_path}")


def create_prediction_comparison_from_checkpoints(
    suite_dir: Path,
    checkpoint_name: str,
    output_name: str,
    config,
):
    from synthetic_data.dataset import get_synthetic_dataloaders

    suite_dir = Path(suite_dir)
    exp_dirs = sorted([d for d in suite_dir.iterdir() if d.is_dir()])

    _, val_loader = get_synthetic_dataloaders(config, num_workers=0)
    batch = next(iter(val_loader))
    inp = batch["input"]

    if config.simulation_mode:
        from synthetic_data.scripts.train_synthetic import get_teacher_kernel

        teacher_kernel = get_teacher_kernel(
            k_size=config.kernel_size,
            radius=config.simulation_radius,
            device="cpu",
        )
        with torch.no_grad():
            tar = F.conv2d(inp, teacher_kernel, groups=3, padding=0)
    else:
        tar = batch["target"]

    img_in = np.clip(np.transpose(inp[0].cpu().numpy(), (1, 2, 0)), 0, 1)
    img_tar = np.clip(np.transpose(tar[0].cpu().numpy(), (1, 2, 0)), 0, 1)

    preds: list[tuple[str, np.ndarray]] = []
    for exp_dir in exp_dirs:
        cp_path = exp_dir / checkpoint_name
        if not cp_path.exists():
            continue
        model = _load_model_from_checkpoint(cp_path)
        with torch.no_grad():
            pred = model(inp)
        img_pred = np.clip(np.transpose(pred[0].cpu().numpy(), (1, 2, 0)), 0, 1)
        label = exp_dir.name.split("_")[-1]
        preds.append((label, img_pred))

    if not preds:
        print(f"No checkpoints named '{checkpoint_name}' in {suite_dir}")
        return

    total = 2 + len(preds)
    fig, axes = plt.subplots(1, total, figsize=(5 * total, 5))

    _plot_image_with_seaborn(axes[0], img_in, "Input")
    _plot_image_with_seaborn(axes[1], img_tar, "Target")

    for i, (label, img_pred) in enumerate(preds):
        _plot_image_with_seaborn(axes[2 + i], img_pred, f"Pred ({label})")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(checkpoint_name, fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_path = suite_dir / output_name
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Prediction comparison saved to: {save_path}")


if __name__ == "__main__":
    pass
