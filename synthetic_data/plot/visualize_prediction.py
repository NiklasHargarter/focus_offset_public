import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import argparse
from pathlib import Path
from synthetic_data.config import SyntheticConfig
from synthetic_data.dataset import get_synthetic_dataloaders
from synthetic_data.model import SyntheticConvModel
from synthetic_data.scripts.train_synthetic import get_teacher_kernel


def visualize_prediction(
    model=None,
    batch=None,
    config=None,
    epoch=None,
    device=None,
    log_dir=None,
):
    if config is None:
        config = SyntheticConfig()

    if log_dir is None:
        log_dir = config.log_dir

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    # Get the first batch from the val loader if no batch provided
    if batch is None:
        _, val_loader = get_synthetic_dataloaders(config, num_workers=0)
        batch = next(iter(val_loader))

    inputs = batch["input"].to(device)

    # Run forward pass on first sample only
    with torch.no_grad():
        preds = model(inputs)

    # Save prediction PNG for sample 0
    if log_dir is not None and epoch is not None:
        log_dir = Path(log_dir)
        pred_img = np.clip(np.transpose(preds[0].cpu().numpy(), (1, 2, 0)), 0, 1)
        save_path = log_dir / f"pred_epoch_{epoch}.png"
        pred_gray = pred_img.mean(axis=2)
        sns.set_theme(style="white")
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(
            pred_gray,
            ax=ax,
            cmap="mako",
            cbar=False,
            xticklabels=False,
            yticklabels=False,
        )
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    return preds


def _to_rgb_image(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()
    return np.clip(np.transpose(arr, (1, 2, 0)), 0.0, 1.0)


def _load_model_from_checkpoint(checkpoint_path: Path, device: str):
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    weight = state_dict["conv.weight"]
    kernel_size = int(weight.shape[-1])
    groups = 3 if int(weight.shape[1]) == 1 else 1
    model = SyntheticConvModel(kernel_size=kernel_size, groups=groups)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def visualize_kernel_use(
    config: SyntheticConfig | None = None,
    checkpoint_path: str | None = None,
    device: str | None = None,
    output_name: str = "kernel_use_simple.png",
):
    """Simple visualization of synthetic kernel usage on one sample.

    Panels (single-checkpoint mode):
    1) Input patch
    2) Teacher kernel (R channel)
    3) Teacher-blurred target
    4) Model prediction (if checkpoint given)
    """
    if config is None:
        config = SyntheticConfig()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, _ = get_synthetic_dataloaders(config, num_workers=0)
    batch = next(iter(train_loader))
    inputs = batch["input"].to(device)

    teacher_kernel = get_teacher_kernel(
        k_size=config.kernel_size,
        radius=config.simulation_radius,
        device=device,
    )

    with torch.no_grad():
        teacher_target = F.conv2d(inputs, teacher_kernel, groups=3, padding=0)

    model = None
    if checkpoint_path:
        cp_path = Path(checkpoint_path)
        if cp_path.exists():
            model = _load_model_from_checkpoint(cp_path, device)
        else:
            print(f"Warning: checkpoint not found: {cp_path}")

    input_img = _to_rgb_image(inputs[0])
    target_img = _to_rgb_image(teacher_target[0])
    kernel_img = teacher_kernel[0, 0].detach().cpu().numpy()

    sns.set_theme(style="white")

    if model is None:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

        axes[0].imshow(input_img)
        axes[0].set_title("Input")
        axes[0].axis("off")

        sns.heatmap(
            kernel_img,
            ax=axes[1],
            cmap="viridis",
            cbar=False,
            xticklabels=False,
            yticklabels=False,
        )
        axes[1].set_title("Teacher Kernel (R)")
        axes[1].axis("off")

        axes[2].imshow(target_img)
        axes[2].set_title("Teacher Blur Target")
        axes[2].axis("off")
    else:
        with torch.no_grad():
            preds = model(inputs)

        pred_img = _to_rgb_image(preds[0])

        fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))

        axes[0].imshow(input_img)
        axes[0].set_title("Input")
        axes[0].axis("off")

        sns.heatmap(
            kernel_img,
            ax=axes[1],
            cmap="viridis",
            cbar=False,
            xticklabels=False,
            yticklabels=False,
        )
        axes[1].set_title("Teacher Kernel (R)")
        axes[1].axis("off")

        axes[2].imshow(target_img)
        axes[2].set_title("Teacher Blur Target")
        axes[2].axis("off")

        axes[3].imshow(pred_img)
        axes[3].set_title("Model Prediction")
        axes[3].axis("off")

    save_dir = Path(config.log_dir)
    if checkpoint_path:
        cp_parent = Path(checkpoint_path).parent
        if cp_parent.exists():
            save_dir = cp_parent
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / output_name
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"Saved simple kernel-use visualization: {save_path}")
    return save_path


def visualize_suite_predictions(
    suite_dir: str,
    config: SyntheticConfig | None = None,
    checkpoint_name: str = "best_model.pt",
    output_name: str = "suite_prediction_panel.png",
):
    """Create one panel with Input + Teacher + one prediction per init run."""
    if config is None:
        config = SyntheticConfig()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    suite_path = Path(suite_dir)

    train_loader, _ = get_synthetic_dataloaders(config, num_workers=0)
    batch = next(iter(train_loader))
    inputs = batch["input"].to(device)

    teacher_kernel = get_teacher_kernel(
        k_size=config.kernel_size,
        radius=config.simulation_radius,
        device=device,
    )
    with torch.no_grad():
        teacher_target = F.conv2d(inputs, teacher_kernel, groups=3, padding=0)

    input_img = _to_rgb_image(inputs[0])
    teacher_img = _to_rgb_image(teacher_target[0])

    predictions: list[tuple[str, np.ndarray]] = []
    for exp_dir in sorted([d for d in suite_path.iterdir() if d.is_dir()]):
        cp = exp_dir / checkpoint_name
        if not cp.exists():
            continue
        model = _load_model_from_checkpoint(cp, device)
        with torch.no_grad():
            pred = model(inputs)
        label = exp_dir.name.split("_")[-1]
        predictions.append((label, _to_rgb_image(pred[0])))

    if not predictions:
        raise FileNotFoundError(
            f"No checkpoints named '{checkpoint_name}' found under {suite_path}"
        )

    total = 2 + len(predictions)
    fig, axes = plt.subplots(1, total, figsize=(4.2 * total, 4.5))

    axes[0].imshow(input_img)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(teacher_img)
    axes[1].set_title("Teacher")
    axes[1].axis("off")

    for i, (label, pred_img) in enumerate(predictions):
        axes[2 + i].imshow(pred_img)
        axes[2 + i].set_title(f"Pred ({label})")
        axes[2 + i].axis("off")

    plt.tight_layout()
    out_path = suite_path / output_name
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved suite prediction panel: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Simple synthetic kernel-use visualization")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path (.pt) for prediction/error panels",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="kernel_use_simple.png",
        help="Output filename inside experiment log dir",
    )
    parser.add_argument(
        "--suite-dir",
        type=str,
        default=None,
        help="Optional suite dir for combined panel: input + teacher + 3 predictions",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="best_model.pt",
        help="Checkpoint name used when --suite-dir is provided",
    )
    args = parser.parse_args()

    if args.suite_dir:
        visualize_suite_predictions(
            suite_dir=args.suite_dir,
            checkpoint_name=args.checkpoint_name,
            output_name=args.output_name,
        )
    else:
        visualize_kernel_use(checkpoint_path=args.checkpoint, output_name=args.output_name)


if __name__ == "__main__":
    main()
