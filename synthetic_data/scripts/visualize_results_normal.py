import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from synthetic_data.config import SyntheticConfig
from synthetic_data.dataset import get_synthetic_dataloaders
from synthetic_data.model import SyntheticConvModel
from synthetic_data.scripts.visualize_results_sim import create_kernel_progression_for_experiment


def _to_rgb(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()
    return np.clip(np.transpose(arr, (1, 2, 0)), 0.0, 1.0)


def _checkpoint_order(path: Path) -> int:
    if path.name == "initial_model.pt":
        return 0
    match = re.match(r"model_epoch_(\d+)\.pt", path.name)
    if match:
        return int(match.group(1))
    if path.name == "best_model.pt":
        return 10**9
    return 10**8


def _select_checkpoints(exp_dir: Path, max_preds: int = 6) -> list[Path]:
    all_paths = []
    for p in [exp_dir / "initial_model.pt", *exp_dir.glob("model_epoch_*.pt"), exp_dir / "best_model.pt"]:
        if p.exists():
            all_paths.append(p)

    all_paths = sorted(set(all_paths), key=_checkpoint_order)
    if not all_paths:
        return []

    if len(all_paths) <= max_preds:
        return all_paths

    # Keep endpoints + evenly-spaced middle checkpoints.
    chosen = [all_paths[0]]
    middle = all_paths[1:-1]
    slots = max_preds - 2
    if slots > 0 and middle:
        idx = np.linspace(0, len(middle) - 1, num=min(slots, len(middle)), dtype=int)
        chosen.extend([middle[i] for i in idx])
    chosen.append(all_paths[-1])
    return chosen


def _checkpoint_label(path: Path) -> str:
    if path.name == "initial_model.pt":
        return "init"
    if path.name == "best_model.pt":
        return "best"
    m = re.match(r"model_epoch_(\d+)\.pt", path.name)
    if m:
        return f"e{int(m.group(1))}"
    return path.stem


def _load_model(checkpoint_path: Path) -> SyntheticConvModel:
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    weight = state_dict["conv.weight"]
    kernel_size = int(weight.shape[-1])
    groups = 3 if int(weight.shape[1]) == 1 else 1
    model = SyntheticConvModel(kernel_size=kernel_size, groups=groups)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def create_prediction_progression_for_experiment(
    exp_dir: Path,
    output_name: str = "prediction_progression.png",
    max_predictions: int = 6,
):
    exp_dir = Path(exp_dir)
    checkpoints = _select_checkpoints(exp_dir, max_preds=max_predictions)
    if not checkpoints:
        print(f"No checkpoints found in: {exp_dir}")
        return None

    config = SyntheticConfig()
    train_loader, val_loader = get_synthetic_dataloaders(config, num_workers=0)
    loader = val_loader if val_loader is not None else train_loader
    batch = next(iter(loader))
    inputs = batch["input"]

    if "target" not in batch:
        print("No target in batch. This normal visualization expects non-simulation data.")
        return None

    target = batch["target"]
    input_img = _to_rgb(inputs[0])
    target_img = _to_rgb(target[0])

    preds: list[tuple[str, np.ndarray]] = []
    with torch.no_grad():
        for cp in checkpoints:
            model = _load_model(cp)
            pred = model(inputs)
            preds.append((_checkpoint_label(cp), _to_rgb(pred[0])))

    sns.set_theme(style="white")
    total = 2 + len(preds)
    fig, axes = plt.subplots(1, total, figsize=(3.8 * total, 4.2))

    axes[0].imshow(input_img)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(target_img)
    axes[1].set_title("Target")
    axes[1].axis("off")

    for i, (label, img) in enumerate(preds):
        axes[2 + i].imshow(img)
        axes[2 + i].set_title(f"Pred {label}")
        axes[2 + i].axis("off")

    fig.suptitle(exp_dir.name, fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_path = exp_dir / output_name
    plt.savefig(save_path, dpi=180)
    plt.close(fig)
    print(f"Prediction progression saved to: {save_path}")
    return save_path


def create_normal_overview(exp_dir: Path):
    exp_dir = Path(exp_dir)
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    create_kernel_progression_for_experiment(exp_dir)
    create_prediction_progression_for_experiment(exp_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Generate normal train/predict visualizations for one experiment directory."
    )
    parser.add_argument("exp_dir", type=str, help="Path to one experiment directory")
    parser.add_argument(
        "--max-predictions",
        type=int,
        default=6,
        help="Maximum number of prediction checkpoints to show",
    )
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    create_kernel_progression_for_experiment(exp_dir)
    create_prediction_progression_for_experiment(
        exp_dir,
        max_predictions=max(1, args.max_predictions),
    )


if __name__ == "__main__":
    main()
