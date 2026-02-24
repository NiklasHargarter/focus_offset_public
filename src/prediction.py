"""
Core prediction / evaluation logic.

Provides ``evaluate()`` — run inference for a single checkpoint on a dataloader
and persist a flat CSV of predictions.

Imported by ``scripts/predict.py`` (CLI) and any other script that needs
programmatic evaluation.
"""

from pathlib import Path

import pandas as pd
import torch
from accelerate import Accelerator
from tqdm.auto import tqdm

from src.models.architectures import MODEL_REGISTRY


def evaluate(
    dataloader: torch.utils.data.DataLoader,
    checkpoint_path: str | Path,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
    dataset_name: str = "unknown",
    model_name_override: str | None = None,
) -> pd.DataFrame:
    """Run inference on the test set and save a predictions CSV.

    Columns
    -------
    model_name, dataset, checkpoint,          ← run context
    pred, target,                             ← core results
    filename, x, y, z_level, optimal_z, …     ← dataset metadata (variable)

    Parameters
    ----------
    dataloader : DataLoader
        Standard PyTorch DataLoader.
    checkpoint_path : path
        Path to the single model checkpoint.
    output_dir : path, optional
        Where to write the CSV.  Defaults to the log folder inferred from
        the checkpoint path.
    dry_run : bool
        Limit to 2 batches for smoke testing.
    dataset_name : str
        Name of the dataset for logging purposes.

    Returns
    -------
    pd.DataFrame
        The full predictions table (same content that was written to disk).
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # 1. Setup Accelerator for inference
    accelerator = Accelerator(
        mixed_precision="bf16",
    )

    # 2. Load Checkpoint
    print(f"Loading checkpoint from {checkpoint_path.name}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats (full dict vs state_dict only)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        model_name = checkpoint.get("model_name", "multimodal")
        print(f"Detected model architecture: {model_name}")
    else:
        state_dict = checkpoint
        # Inference from path if not provided
        if model_name_override:
            model_name = model_name_override
        elif "dwt" in str(checkpoint_path).lower():
            model_name = "dwt"
        elif "fft" in str(checkpoint_path).lower():
            model_name = "fft"
        elif "rgb" in str(checkpoint_path).lower():
            model_name = "rgb"
        else:
            model_name = "multimodal"  # Fallback
        print(
            f"Warning: Raw state dict found. Assuming '{model_name}' based on path/override."
        )

    # 3. Instantiate Model
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model architecture '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )

    import torch.nn as nn

    model: nn.Module = MODEL_REGISTRY[model_name]()
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Prepare
    model, dataloader = accelerator.prepare(model, dataloader)

    # 5. Inference Loop
    results = []

    print(f"Running inference on {accelerator.device}...")
    with torch.no_grad():
        for batch_idx, batch in tqdm(
            enumerate(dataloader),
            desc="Predicting",
            disable=not accelerator.is_local_main_process,
        ):
            if dry_run and batch_idx >= 2:
                break

            images = batch["image"]
            targets = batch["target"]

            preds = model(images)

            # Move to CPU directly since we are not doing multi-gpu gathering logic here for simplicity
            # (Accelerate prepare handles device placement, but we want numpy/list results)
            preds = preds.detach().float().cpu()
            targets = targets.detach().float().cpu()

            batch_size = preds.size(0)

            # Handle metadata
            # Metadata is a dict of lists or tensors. We need to "transpose" it to row-based
            meta = batch.get("metadata", {})

            for i in range(batch_size):
                row = {
                    "pred": preds[i].item(),
                    "target": targets[i].item(),
                }

                # Extract metadata for this sample
                for key, val in meta.items():
                    # Handle multi-element tensors (like tile_coords [R, C])
                    if isinstance(val, torch.Tensor):
                        if val.dim() > 1 and val.size(1) > 1:
                            # It's a batch of vectors, convert this sample's vector to list
                            row[key] = val[i].tolist()
                        elif val.dim() > 0:
                            row[key] = val[i].item()
                        else:
                            row[key] = val.item()
                    elif isinstance(val, list):
                        row[key] = val[i]
                    elif isinstance(val, tuple):
                        row[key] = val[i]

                results.append(row)

    df = pd.DataFrame(results)

    df.insert(0, "model_name", model_name)
    df.insert(1, "dataset", dataset_name)
    df.insert(2, "checkpoint", checkpoint_path.name)

    if output_dir is None:
        output_dir = _infer_log_dir(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_name = f"eval_{model_name}_{dataset_name}.csv"
    csv_path = output_dir / csv_name
    df.to_csv(csv_path, index=False)
    print(f"  → Saved predictions to {csv_path} ({len(df)} rows)")

    return df


def _infer_log_dir(checkpoint_path: Path) -> Path:
    """``logs/rgb/version_0/checkpoints/best.ckpt`` → ``logs/rgb/version_0/``"""
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent
