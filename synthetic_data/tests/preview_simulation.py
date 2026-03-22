import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from synthetic_data.config import SyntheticConfig
from synthetic_data.dataset import get_synthetic_dataloaders
from synthetic_data.scripts.train_synthetic import get_teacher_kernel


def preview_simulation():
    config = SyntheticConfig()
    config.simulation_mode = True
    config.batch_size = 1
    config.workers = 0
    config.dry_run = True

    print(
        f"Loading simulated dataset and generating Gaussian Teacher (sigma={config.simulation_radius})..."
    )
    train_loader, _ = get_synthetic_dataloaders(config, num_workers=0)

    # Get one batch of sharp input
    batch = next(iter(train_loader))
    img_in_tensor = batch["input"]  # [1, 3, N, N]

    # Pre-build teacher
    teacher_kernel = get_teacher_kernel(
        k_size=config.kernel_size, radius=config.simulation_radius, device="cpu"
    )

    # Dynamically generate simulated target
    with torch.no_grad():
        target_tensor = F.conv2d(img_in_tensor, teacher_kernel, groups=3, padding=0)

    # Convert to numpy for plotting
    img_in = img_in_tensor[0].permute(1, 2, 0).numpy()
    img_target = target_tensor[0].permute(1, 2, 0).numpy()

    # Kernel is [3, 1, K, K], we take the first group [0, 0] for heatmap
    kernel_np = teacher_kernel[0, 0].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img_in)
    axes[0].set_title("Sharp Input (VSI)")
    axes[1].imshow(img_target)
    axes[1].set_title("Simulated Target (Gaussian Conv)")

    sns.heatmap(kernel_np, ax=axes[2], cmap="viridis", center=0)
    axes[2].set_title(f"Teacher Kernel Phase (sigma={config.simulation_radius})")
    axes[2].set_aspect("equal")

    for ax in axes[:2]:
        ax.axis("off")
    axes[2].axis("off")

    save_path = config.log_dir / "simulation_preview.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    print(f"Preview saved to: {save_path}")


if __name__ == "__main__":
    preview_simulation()
