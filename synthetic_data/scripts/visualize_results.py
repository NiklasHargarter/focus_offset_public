"""Compatibility wrapper.

Use specific scripts instead:
- `synthetic_data.scripts.visualize_results_sim` for simulation-focused suite visuals
- `synthetic_data.scripts.visualize_results_normal` for normal train/predict visuals
"""

import argparse
from pathlib import Path

from synthetic_data.scripts.visualize_results_normal import create_normal_overview
from synthetic_data.scripts.visualize_results_sim import create_suite_overview


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic_data visualizations")
    parser.add_argument("path", type=str, help="Path to experiment or suite directory")
    parser.add_argument(
        "--mode",
        choices=["sim", "normal"],
        default="sim",
        help="Visualization mode: simulation suite (`sim`) or normal train/predict (`normal`).",
    )
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        raise FileNotFoundError(f"Path not found: {target}")

    if args.mode == "sim":
        create_suite_overview(target)
    else:
        create_normal_overview(target)


if __name__ == "__main__":
    main()
