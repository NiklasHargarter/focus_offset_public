"""Plot train vs val loss for a single model. Shows overfitting, divergence, or instability at a glance."""

import argparse

import matplotlib.pyplot as plt
import seaborn as sns

from src.plotting import load_training_curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model name (subfolder under logs/)")
    parser.add_argument(
        "-v",
        "--version",
        nargs="*",
        type=int,
        default=None,
        help="Version number(s) to include (e.g. 0 2)",
    )
    parser.add_argument(
        "-o", "--output", default=None, help="Save path (default: show)"
    )
    args = parser.parse_args()

    df = load_training_curves()
    df = df[df["model"] == args.model]
    if args.version is not None:
        df = df[df["version"].isin(args.version)]
    if df.empty:
        raise SystemExit(
            f"No data for model '{args.model}' with version(s) {args.version}"
        )

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(data=df, x="epoch", y="train_loss_epoch", label="train", ax=ax)
    sns.lineplot(data=df, x="epoch", y="val_loss", label="val", ax=ax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    title = f"{args.model} — Train vs Val Loss"
    if args.version is not None:
        title += f" (version(s): {', '.join(map(str, args.version))})"
    ax.set_title(title)

    if args.output:
        fig.savefig(args.output, dpi=200, bbox_inches="tight")
        print(f"Saved → {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
