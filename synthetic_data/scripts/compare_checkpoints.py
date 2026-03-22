import argparse
import hashlib
from pathlib import Path

import torch
from tqdm import tqdm


def get_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compare_checkpoints(exp_dir: str):
    exp_dir = Path(exp_dir)
    if not exp_dir.exists():
        print(f"Error: Directory {exp_dir} does not exist.")
        return

    ckpts = sorted(list(exp_dir.glob("*.pt")))
    if not ckpts:
        print(f"No .pt files found in {exp_dir}")
        return

    print(f"Comparing {len(ckpts)} checkpoints in {exp_dir.name}...\n")
    
    results = []
    
    # 1. Compare file hashes first (fastest)
    # 2. Compare actual tensor weights (more reliable for state dicts)
    
    first_ckpt = None
    first_weights = None
    prev_weights = None
    
    for ckpt_path in tqdm(ckpts, desc="Loading checkpoints"):
        file_hash = get_file_hash(ckpt_path)
        
        try:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            actual_dict = state_dict.get("model_state_dict", state_dict)
            weights = actual_dict.get("conv.weight", None)
            
            if weights is None:
                continue

            if first_weights is None:
                first_weights = weights
                first_ckpt = ckpt_path.name
                prev_weights = weights

            l1_from_first = torch.norm(weights - first_weights, p=1).item()
            l1_from_prev = torch.norm(weights - prev_weights, p=1).item()
            
            results.append({
                "name": ckpt_path.name,
                "hash": file_hash[:10],
                "l1_first": l1_from_first,
                "l1_prev": l1_from_prev,
                "weight_sum": weights.sum().item()
            })
            prev_weights = weights
        except Exception as e:
            print(f"Error loading {ckpt_path.name}: {e}")

    # Print results table
    print(f"{'Filename':<30} | {'L1 vs First':<12} | {'L1 vs Prev':<12} | {'Weight Sum':<12}")
    print("-" * 75)
    for res in results:
        print(f"{res['name']:<30} | {res['l1_first']:<12.6f} | {res['l1_prev']:<12.6f} | {res['weight_sum']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare checkpoints in an experiment directory.")
    parser.add_argument("exp_dir", type=str, help="Path to the experiment directory.")
    args = parser.parse_args()
    
    compare_checkpoints(args.exp_dir)
