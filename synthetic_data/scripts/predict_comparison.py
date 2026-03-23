"""
Generate synthetic predictions using the best models from a comparison suite.

For each trained model, loads a few validation patches, applies the learned
kernel, and saves side-by-side panels: Input → Prediction → Ground Truth → Residual.

Usage:
  python -m synthetic_data.scripts.predict_comparison <suite_dir>
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from synthetic_data.config import SyntheticConfig
from synthetic_data.dataset import get_synthetic_dataloaders
from synthetic_data.model import SyntheticConvModel


def _load_model(checkpoint_path: Path) -> SyntheticConvModel:
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    w = sd["conv.weight"]
    k = int(w.shape[-1])
    g = 3 if int(w.shape[1]) == 1 else 1
    model = SyntheticConvModel(kernel_size=k, groups=g)
    model.load_state_dict(sd)
    model.eval()
    return model


def _to_rgb(t: torch.Tensor) -> np.ndarray:
    return np.clip(t.detach().cpu().numpy().transpose(1, 2, 0), 0.0, 1.0)


def _parse_config_from_dirname(exp_name: str) -> tuple[str, int]:
    """Extract (slide_dir_suffix, z_offset_steps) from experiment name."""
    parts = exp_name.split("_")
    offset = 0
    dataset = "ZStack_HE"
    for i, p in enumerate(parts):
        if p.startswith("off"):
            offset = int(p[3:])
        if p == "ZStack" and i + 1 < len(parts):
            dataset = f"ZStack_{parts[i + 1]}"
    return dataset, offset


def _parse_label(exp_name: str) -> str:
    parts = exp_name.split("_")
    offset = ""
    ds = ""
    for i, p in enumerate(parts):
        if p.startswith("off"):
            val = p[3:]
            offset = f"+{val}" if not val.startswith("-") else val
        if p == "ZStack" and i + 1 < len(parts):
            ds = parts[i + 1]
    return f"{ds} {offset}" if ds else exp_name


def predict_for_experiment(
    exp_dir: Path,
    num_samples: int = 3,
    device: str = "cpu",
):
    """Generate prediction panels for one experiment directory."""
    cp = exp_dir / "best_model.pt"
    if not cp.exists():
        print(f"  Skipping {exp_dir.name}: no best_model.pt")
        return

    dataset_name, z_offset = _parse_config_from_dirname(exp_dir.name)
    slide_dir = f"/data/niklas/{dataset_name}"

    config = SyntheticConfig(slide_dir=slide_dir, z_offset_steps=z_offset)
    label = _parse_label(exp_dir.name)

    try:
        _, val_loader = get_synthetic_dataloaders(config, num_workers=0)
    except Exception as e:
        print(f"  Skipping {label}: could not load data ({e})")
        return

    model = _load_model(cp).to(device)
    batch = next(iter(val_loader))
    inputs = batch["input"].to(device)
    targets = batch["target"].to(device)

    with torch.no_grad():
        preds = model(inputs)

    n = min(num_samples, inputs.shape[0])

    sns.set_theme(style="white")
    fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        inp_img = _to_rgb(inputs[i])
        pred_img = _to_rgb(preds[i])
        tgt_img = _to_rgb(targets[i])
        residual = np.abs(pred_img - tgt_img)

        axes[i, 0].imshow(inp_img)
        axes[i, 0].set_title("Input (offset)" if i == 0 else "")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(pred_img)
        axes[i, 1].set_title("Prediction" if i == 0 else "")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(tgt_img)
        axes[i, 2].set_title("Ground Truth" if i == 0 else "")
        axes[i, 2].axis("off")

        im = axes[i, 3].imshow(residual.mean(axis=2), cmap="hot", vmin=0, vmax=0.15)
        axes[i, 3].set_title("Residual" if i == 0 else "")
        axes[i, 3].axis("off")

    fig.suptitle(f"Predictions: {label}", fontsize=15, fontweight="bold")
    plt.tight_layout()
    out = exp_dir / "prediction_samples.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def predict_comparison_grid(
    suite_dir: Path,
    device: str = "cpu",
):
    """One-sample comparison grid: rows = experiments, cols = input/pred/target/residual."""
    suite_dir = Path(suite_dir)
    exp_dirs = sorted([d for d in suite_dir.iterdir() if d.is_dir() and (d / "best_model.pt").exists()])

    if not exp_dirs:
        print("No experiments with best_model.pt found.")
        return

    rows_data = []
    for exp_dir in exp_dirs:
        dataset_name, z_offset = _parse_config_from_dirname(exp_dir.name)
        slide_dir = f"/data/niklas/{dataset_name}"
        config = SyntheticConfig(slide_dir=slide_dir, z_offset_steps=z_offset)
        label = _parse_label(exp_dir.name)

        try:
            _, val_loader = get_synthetic_dataloaders(config, num_workers=0)
        except Exception as e:
            print(f"  Skipping {label}: {e}")
            continue

        model = _load_model(exp_dir / "best_model.pt").to(device)
        batch = next(iter(val_loader))
        inp = batch["input"].to(device)
        tgt = batch["target"].to(device)

        with torch.no_grad():
            pred = model(inp)

        rows_data.append((
            label,
            _to_rgb(inp[0]),
            _to_rgb(pred[0]),
            _to_rgb(tgt[0]),
        ))

    if not rows_data:
        return

    n = len(rows_data)
    sns.set_theme(style="white")
    fig, axes = plt.subplots(n, 4, figsize=(22, 5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    col_titles = ["Input (offset)", "Prediction", "Ground Truth", "Residual"]

    for r, (label, inp_img, pred_img, tgt_img) in enumerate(rows_data):
        residual = np.abs(pred_img - tgt_img)

        axes[r, 0].imshow(inp_img)
        axes[r, 1].imshow(pred_img)
        axes[r, 2].imshow(tgt_img)
        axes[r, 3].imshow(residual.mean(axis=2), cmap="hot", vmin=0, vmax=0.15)

        axes[r, 0].set_ylabel(label, fontsize=13, fontweight="bold", rotation=0, labelpad=80)

        for c in range(4):
            if r == 0:
                axes[r, c].set_title(col_titles[c], fontsize=12)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    fig.suptitle("Comparison: Learned Kernel Predictions", fontsize=16, fontweight="bold")
    plt.tight_layout()
    out = suite_dir / "prediction_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions from best models in a comparison suite."
    )
    parser.add_argument("suite_dir", type=str, help="Path to the comparison suite directory")
    parser.add_argument(
        "--num-samples", type=int, default=3,
        help="Number of sample patches per experiment (default: 3)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (default: auto-detect)",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    suite_dir = Path(args.suite_dir)

    print(f"Generating predictions for: {suite_dir}\n")

    # Per-experiment detailed panels
    for d in sorted(suite_dir.iterdir()):
        if d.is_dir() and (d / "best_model.pt").exists():
            predict_for_experiment(d, num_samples=args.num_samples, device=device)

    # Cross-experiment comparison grid
    print()
    predict_comparison_grid(suite_dir, device=device)

    print(f"\nAll predictions saved to: {suite_dir}")


if __name__ == "__main__":
    main()
