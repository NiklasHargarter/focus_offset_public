import argparse
import matplotlib.pyplot as plt
import slideio
import cv2
from pathlib import Path

from src.utils.io_utils import suppress_stderr


def create_context_visualization(vsi_path, output_dir):
    print(f" Creating Multi-Scale Context Visualization: {vsi_path.name}")

    with suppress_stderr():
        slide = slideio.open_slide(str(vsi_path), "VSI")
    scene = slide.get_scene(0)

    # 1. Find a good tissue spot
    w, h = scene.size
    cx, cy = 12000, 10000

    # Largest area is 1792px
    max_size = 1792

    print(f"  -> Reading {max_size}px native block at Z=13...")
    with suppress_stderr():
        full_block = scene.read_block(
            rect=(cx - max_size // 2, cy - max_size // 2, max_size, max_size),
            size=(max_size, max_size),
            slices=(13, 14),
        )

    # BGR to RGB
    full_block = full_block[..., ::-1]

    # 2. Extract context patches
    target_dim = 224

    # 1x (Native)
    p1x_raw = full_block[
        max_size // 2 - 112 : max_size // 2 + 112,
        max_size // 2 - 112 : max_size // 2 + 112,
    ]

    # 2x (448px area)
    p2x_raw = full_block[
        max_size // 2 - 224 : max_size // 2 + 224,
        max_size // 2 - 224 : max_size // 2 + 224,
    ]
    p2x_res = cv2.resize(
        p2x_raw, (target_dim, target_dim), interpolation=cv2.INTER_AREA
    )

    # 4x (896px area)
    p4x_raw = full_block[
        max_size // 2 - 448 : max_size // 2 + 448,
        max_size // 2 - 448 : max_size // 2 + 448,
    ]
    p4x_res = cv2.resize(
        p4x_raw, (target_dim, target_dim), interpolation=cv2.INTER_AREA
    )

    # 8x (1792px area)
    p8x_res = cv2.resize(
        full_block, (target_dim, target_dim), interpolation=cv2.INTER_AREA
    )

    # 3. LEGACY BRENNER RESOLUTION
    # Legacy: 224px native -> 28px tiny patch for Brenner calculation
    p_legacy_8x_tiny = cv2.resize(p1x_raw, (28, 28), interpolation=cv2.INTER_AREA)

    # --- PLOT 1: THE NESTED BOXES ---
    plt.figure(figsize=(10, 10))
    plt.imshow(full_block)

    icx, icy = max_size // 2, max_size // 2
    sizes = [224, 448, 896, 1792]
    labels = ["1x (224px)", "2x (448px)", "4x (896px)", "8x (1792px)"]
    colors = ["yellow", "cyan", "lime", "magenta"]

    for s, label, c in zip(sizes, labels, colors):
        rect = plt.Rectangle(
            (icx - s // 2, icy - s // 2),
            s,
            s,
            edgecolor=c,
            facecolor="none",
            linewidth=3,
            label=label,
        )
        plt.gca().add_patch(rect)

    plt.title(
        f"Patch Extraction Contexts (Native Resolution Units)\nSlide: {vsi_path.name}",
        fontsize=15,
        fontweight="bold",
    )
    plt.legend(loc="upper right")
    plt.axis("off")

    box_path = output_dir / f"vis_context_boxes_{vsi_path.stem}.png"
    plt.savefig(box_path, dpi=150, bbox_inches="tight")
    plt.close()

    # --- PLOT 2: THE MODEL INPUTS ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        "Model Input Comparison (Different Contexts, All 224px)",
        fontsize=18,
        fontweight="bold",
    )

    titles = ["Native (1x)", "Zoom Out (2x)", "Neighborhood (4x)", "Full Context (8x)"]
    images = [p1x_raw, p2x_res, p4x_res, p8x_res]

    for ax, img, t in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(t, fontsize=14, fontweight="bold")
        ax.axis("off")

    input_path = output_dir / f"vis_context_inputs_{vsi_path.stem}.png"
    plt.savefig(input_path, dpi=150, bbox_inches="tight")
    plt.close()

    # --- PLOT 3: LEGACY BRENNER RESOLUTION ---
    # We use a narrower subplot for the 28x patch to highlight its "smallness"
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1, 0.4]}
    )
    fig.suptitle(
        "Information Density: Native Detail vs. Legacy Brenner",
        fontsize=18,
        fontweight="bold",
    )

    axes[0].imshow(p1x_raw)
    axes[0].set_title(
        "Training Sample (224x224)\nNative Detail used by ResNet", fontsize=12
    )

    axes[1].imshow(p_legacy_8x_tiny)
    axes[1].set_title(
        "Legacy Brenner Input (28x28)\nActual Pixels used for Labels", fontsize=12
    )

    for ax in axes:
        ax.axis("off")

    blur_path = output_dir / f"vis_legacy_blur_{vsi_path.stem}.png"
    plt.savefig(blur_path, dpi=150, bbox_inches="tight")
    plt.close()

    # --- MARKDOWN REPORT ---
    report_path = output_dir / f"vis_data_extraction_manual_{vsi_path.stem}.md"
    with open(report_path, "w") as f:
        f.write("# Visual Study: Multi-Scale Context and Legacy Resolution Gap\n\n")
        f.write("## 1. Context Boundaries (Relative to Slide)\n")
        f.write(
            "This image shows how much physical area on the slide is contained within the "
            "various 'Scales' while maintaining the center coordinate.\n\n"
        )
        f.write(f"![Context Boxes](vis_context_boxes_{vsi_path.stem}.png)\n\n")

        f.write("## 2. Input Scaling (Fixed Model View)\n")
        f.write(
            "Compare the visual information provided to the model at each scale. "
            "Higher downscales provide more structural context at the cost of high-frequency detail.\n\n"
        )
        f.write(f"![Model Inputs](vis_context_inputs_{vsi_path.stem}.png)\n\n")

        f.write("## 3. The Legacy Resolution Gap\n")
        f.write(
            "The legacy approach prioritized speed by reading low-resolution data for Brenner focus scores. "
            "This visual demonstrates the literal information density difference between the 1x Training patch "
            "and the 8x Legacy Brenner patch used for ground-truth labeling:\n\n"
        )
        f.write(f"![Legacy Blur](vis_legacy_blur_{vsi_path.stem}.png)\n\n")
        f.write(
            "**Conclusion**: The 28x28 pixel patch (right) lacks the structural detail to precisely "
            "identify the focus peak of a high-resolution 60x sample. This explains the ~0.8 slice "
            "jitter found in our MAE benchmarks."
        )

    print(f" Visualization sequence complete. Results in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vsi_path", type=str, default="001_1_HE_stack.vsi")
    args = parser.parse_args()

    out_dir = Path("src/experiments/focus_mechanics_v2/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    create_context_visualization(Path(args.vsi_path), out_dir)
