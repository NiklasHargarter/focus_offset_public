"""
Clean, publication-quality visualization for a simulation experiment.

Generates a single summary figure with:
  - Teacher kernel vs best learned kernel for all 3 channels (R, G, B)
  - Per-channel kernel progression strip (init → selected epochs → best)

Usage:
  python -m synthetic_data.scripts.visualize_sim_clean <exp_dir>
  python -m synthetic_data.scripts.visualize_sim_clean logs/suite_20260308_195442/conv_k63_off-10_g3_random
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

from synthetic_data.config import SyntheticConfig
from synthetic_data.scripts.train_synthetic import get_teacher_kernel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CHANNELS = ["R", "G", "B"]
CHANNEL_COLORS = ["#E24A33", "#228B22", "#348ABD"]


def _load_kernel_weight(checkpoint: Path) -> torch.Tensor:
    sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return sd["conv.weight"]


def _all_channels(weight: torch.Tensor) -> list[np.ndarray]:
    """Return [R, G, B] kernels from a weight tensor."""
    w = weight.detach().cpu().numpy()
    if w.shape[1] == 1:  # depthwise (groups=3)
        return [w[c, 0] for c in range(3)]
    return [w[c, c] for c in range(3)]  # diagonal of full conv


def _adaptive_imshow(ax, kernel: np.ndarray):
    """Show kernel with per-kernel full-range scaling, consistent colormap."""
    lim = max(float(np.abs(kernel).max()), 1e-12)
    ax.imshow(kernel, cmap="RdBu_r", vmin=-lim, vmax=lim)


def _select_progression_checkpoints(exp_dir: Path, n: int = 6) -> list[Path]:
    """Pick init + evenly-spaced epochs + best."""
    epoch_cps = sorted(
        exp_dir.glob("model_epoch_*.pt"),
        key=lambda p: int(re.match(r"model_epoch_(\d+)\.pt", p.name).group(1)),
    )
    chosen = []
    init = exp_dir / "initial_model.pt"
    if init.exists():
        chosen.append(init)

    if epoch_cps:
        slots = max(1, n - 2)
        idx = np.linspace(0, len(epoch_cps) - 1, slots, dtype=int)
        chosen.extend([epoch_cps[i] for i in idx])

    best = exp_dir / "best_model.pt"
    if best.exists():
        chosen.append(best)

    return chosen


def _checkpoint_label(p: Path) -> str:
    if p.name == "initial_model.pt":
        return "Init"
    if p.name == "best_model.pt":
        return "Best"
    m = re.match(r"model_epoch_(\d+)\.pt", p.name)
    return f"E{m.group(1)}" if m else p.stem


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def create_clean_sim_figure(
    exp_dir: Path,
    config: SyntheticConfig | None = None,
    output_name: str = "sim_summary.png",
):
    exp_dir = Path(exp_dir)
    if config is None:
        config = SyntheticConfig()

    # Load teacher and best kernels (all 3 channels)
    teacher_w = get_teacher_kernel(
        k_size=config.kernel_size,
        radius=config.simulation_radius,
        device="cpu",
    )
    teacher_chs = _all_channels(teacher_w)

    best_path = exp_dir / "best_model.pt"
    if not best_path.exists():
        print(f"No best_model.pt in {exp_dir}")
        return
    best_chs = _all_channels(_load_kernel_weight(best_path))

    # Progression checkpoints (all channels per checkpoint)
    prog_cps = _select_progression_checkpoints(exp_dir, n=6)
    prog_data = [
        (_checkpoint_label(cp), _all_channels(_load_kernel_weight(cp)))
        for cp in prog_cps
    ]
    n_prog = len(prog_data)

    # ---- Layout ----
    # Row 0:  Teacher R | Teacher G | Teacher B | (gap) | Best R | Best G | Best B
    # Rows 1-3: Progression for R, G, B (n_prog columns each)
    n_cols = max(n_prog, 7)  # at least 7 for top row

    fig = plt.figure(figsize=(2.4 * n_cols, 10), facecolor="white")
    gs = gridspec.GridSpec(
        4, n_cols, figure=fig,
        height_ratios=[1.2, 1, 1, 1],
        hspace=0.30, wspace=0.15,
        left=0.03, right=0.97, top=0.92, bottom=0.03,
    )

    # --- Row 0: Teacher (3 ch) + gap + Best (3 ch) ---
    for c in range(3):
        ax = fig.add_subplot(gs[0, c])
        _adaptive_imshow(ax, teacher_chs[c])
        ax.set_title(f"Teacher {CHANNELS[c]}", fontsize=11, fontweight="bold",
                      color=CHANNEL_COLORS[c])
        ax.axis("off")

    # gap column(s) in the middle
    for c in range(3):
        col = n_cols - 3 + c
        ax = fig.add_subplot(gs[0, col])
        _adaptive_imshow(ax, best_chs[c])
        ax.set_title(f"Learned {CHANNELS[c]}", fontsize=11, fontweight="bold",
                      color=CHANNEL_COLORS[c])
        ax.axis("off")

    # --- Rows 1-3: Per-channel progression ---
    for ch_idx in range(3):
        first_ax = None
        for p_idx, (label, chs) in enumerate(prog_data):
            ax = fig.add_subplot(gs[1 + ch_idx, p_idx])
            _adaptive_imshow(ax, chs[ch_idx])
            ax.axis("off")
            if ch_idx == 0:
                ax.set_title(label, fontsize=10, fontweight="bold")
            if p_idx == 0:
                first_ax = ax
        # Channel label to the left of the first column
        if first_ax is not None:
            bbox = first_ax.get_position()
            fig.text(
                bbox.x0 - 0.015, bbox.y0 + bbox.height / 2,
                CHANNELS[ch_idx], fontsize=14, fontweight="bold",
                color=CHANNEL_COLORS[ch_idx], ha="right", va="center",
            )

    fig.suptitle(exp_dir.name, fontsize=15, fontweight="bold")
    out = exp_dir / output_name
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Clean simulation summary figure."
    )
    parser.add_argument("exp_dir", type=str, help="Path to experiment directory")
    parser.add_argument(
        "--output", type=str, default="sim_summary.png",
        help="Output filename (default: sim_summary.png)",
    )
    args = parser.parse_args()
    create_clean_sim_figure(Path(args.exp_dir), output_name=args.output)


if __name__ == "__main__":
    main()
