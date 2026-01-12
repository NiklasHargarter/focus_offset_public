import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def plot_stability(csv_path: Path):
    """Plot results from bench_stability.py"""
    df = pd.read_csv(csv_path)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot MAE
    color = '#2A9D8F'
    ax1.set_xlabel('Context Size (Raw Pixels)', fontsize=12)
    ax1.set_ylabel('MAE (Z-slices)', color=color, fontsize=12)
    ax1.plot(df['Context_Size'], df['MAE'], color=color, marker='o', linewidth=3, label='MAE')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Plot Accuracy
    ax2 = ax1.twinx()
    color = '#264653'
    ax2.set_ylabel('Accuracy (Exact Match %)', color=color, fontsize=12)
    ax2.plot(df['Context_Size'], df['Accuracy'] * 100, color=color, marker='s', linestyle='--', linewidth=2, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 105)
    
    plt.title(f"Focus Stability vs Context Size\n{csv_path.name}", fontsize=14, fontweight='bold')
    
    out_path = csv_path.with_suffix('.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {out_path}")

def plot_surface(csv_path: Path):
    """Plot results from analyze_focus_surface.py"""
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['Patch_Size'], df['RMSE'], marker='o', color='#E76F51', linewidth=3)
    plt.xlabel('Patch Size (Pixels)', fontsize=12)
    plt.ylabel('Residual RMSE (Local Jitter)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.title(f"Residual Jitter vs Patch Size\n{csv_path.name}", fontsize=14, fontweight='bold')
    
    out_path = csv_path.with_suffix('.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str)
    parser.add_argument("--type", choices=['stability', 'surface'], required=True)
    args = parser.parse_args()
    
    path = Path(args.csv_path)
    if args.type == 'stability':
        plot_stability(path)
    else:
        plot_surface(path)
