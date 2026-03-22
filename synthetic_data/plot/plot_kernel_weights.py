import torch
import matplotlib.pyplot as plt
import seaborn as sns
from synthetic_data.config import SyntheticConfig


def render_kernel_heatmap(weights, ax=None):
    """
    Renders a 3-channel kernel heatmap onto a given axis or returns a new figure.
    Expects weights as a numpy array of shape [3, 1, K, K] or [3, 3, K, K].
    """
    if ax is None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    else:
        # Caller provides axes (e.g. for tiling)
        axes = ax

    channels = ["Red", "Green", "Blue"]
    for i in range(3):
        if weights.shape[1] == 1:
            # Depthwise: take the only input channel
            kernel = weights[i, 0, :, :]
        else:
            # Full: take the diagonal (R->R, G->G, B->B)
            kernel = weights[i, i, :, :]

        cur_ax = axes[i] if hasattr(axes, "__getitem__") else axes
        sns.heatmap(kernel, ax=cur_ax, cmap="viridis", center=0)
        cur_ax.set_title(f"{channels[i]} Channel Kernel", fontsize=9)
        cur_ax.axis("off")

    if ax is None:
        plt.tight_layout()
        return fig


def plot_kernel_weights(weights=None, config=None, epoch=None, log_dir=None):
    if config is None:
        config = SyntheticConfig()

    if log_dir is None:
        log_dir = config.log_dir

    if weights is None:
        model_path = log_dir / "best_model.pt"
        if not model_path.exists():
            print(f"Error: No model found at {model_path}")
            return
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        weights = state_dict["conv.weight"]

    suffix = f"_epoch_{epoch}" if epoch is not None else ""

    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()

    sns.set_theme(style="white")
    fig = render_kernel_heatmap(weights)

    save_path = log_dir / f"kernel_heatmap{suffix}.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Heatmap saved to: {save_path}")


if __name__ == "__main__":
    plot_kernel_weights()
