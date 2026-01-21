import argparse
import time
from pathlib import Path
import slideio

from src.utils.io_utils import suppress_stderr


def run_global_z_benchmark(vsi_path: Path):
    print(f" Benchmarking Global Z-Slice Reads: {vsi_path.name}")

    with suppress_stderr():
        slide = slideio.open_slide(str(vsi_path), "VSI")
    scene = slide.get_scene(0)
    width, height = scene.size

    downscale_factors = [2, 4, 8, 16, 32]

    results = []

    print(f"Slide Dimensions: {width} x {height}")
    print("-" * 50)

    for ds in downscale_factors:
        # Calculate target size
        target_w = width // ds
        target_h = height // ds
        estimated_mem = (target_w * target_h * 3) / (1024**2)  # MB for BGR

        print(f"Testing {ds}x Downscale...")
        print(f"  -> Size: {target_w}x{target_h}")
        print(f"  -> Est. Memory: {estimated_mem:.1f} MB per Z-slice")

        try:
            t0 = time.time()
            # Read only ONE Z-slice globally
            with suppress_stderr():
                img = scene.read_block(
                    rect=(0, 0, width, height), size=(target_w, target_h), slices=(0, 1)
                )
            dt = time.time() - t0

            # Actual RAM increase
            actual_mem = img.nbytes / (1024**2)

            print(f"  -> Result: Success in {dt:.2f}s | Actual: {actual_mem:.1f} MB")
            results.append((ds, dt, actual_mem))

            # Clean up immediately
            del img

        except Exception as e:
            print(f"  -> Result: FAILED ({str(e)})")
            results.append((ds, None, None))

    # Final Report
    output_dir = Path("src/experiments/focus_mechanics_v2/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"bench_global_z_reads_{vsi_path.stem}.md"

    with open(report_path, "w") as f:
        f.write("# Benchmark: Global Z-Slice Reads\n\n")
        f.write(f"- **Slide**: {vsi_path.name}\n")
        f.write(f"- **Native Dim**: {width} x {height}\n\n")
        f.write("## Throughput per Downscale (Single Z-Slice)\n\n")
        f.write("| Downscale | Width | Height | Time (s) | Memory (MB) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for ds, dt, mem in results:
            if dt is not None:
                tw, th = width // ds, height // ds
                f.write(f"| {ds}x | {tw} | {th} | {dt:.3f}s | {mem:.1f} |\n")
            else:
                f.write(f"| {ds}x | - | - | FAILED | - |\n")

    print(f"\n Benchmark complete. Results saved to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vsi_path", type=str, default="001_1_HE_stack.vsi")
    args = parser.parse_args()
    run_global_z_benchmark(Path(args.vsi_path))
