import matplotlib.pyplot as plt
import seaborn as sns
from synthetic_data.config import SyntheticConfig
from synthetic_data.scripts.train_synthetic import get_teacher_kernel


def verify_simulation():
    config = SyntheticConfig()
    device = "cpu"

    # 1. Build the Teacher
    teacher_kernel = get_teacher_kernel(
        k_size=config.kernel_size, radius=config.simulation_radius, device=device
    )

    # 2. Visualize Teacher weights [3, 1, k, k]
    weights = teacher_kernel.detach().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    channels = ["Red", "Green", "Blue"]

    for i in range(3):
        # We take the only input channel for each output channel group
        kernel = weights[i, 0, :, :]
        sns.heatmap(kernel, ax=axes[i], cmap="viridis", center=0)
        axes[i].set_title(f"Teacher {channels[i]} Kernel")
        axes[i].axis("off")

    path = config.log_dir / "teacher_verification.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    print(f"Teacher verification saved to: {path}")


if __name__ == "__main__":
    verify_simulation()
