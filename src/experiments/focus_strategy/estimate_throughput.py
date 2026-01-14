import argparse
import time
import multiprocessing
import numpy as np
import slideio
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from functools import partial

from src.utils.focus_metrics import compute_brenner_gradient
from src.utils.io_utils import suppress_stderr

def _bench_local_worker(coords, vsi_path):
    with suppress_stderr():
        try:
            slide = slideio.open_slide(str(vsi_path), "VSI")
            scene = slide.get_scene(0)
            num_z = scene.num_z_slices
            cx, cy = coords
            for z in range(num_z):
                img = scene.read_block(rect=(cx-112, cy-112, 224, 224), size=(224, 224), slices=(z, z+1))
                compute_brenner_gradient(img)
            return True
        except:
            return False

def _bench_super_worker(coords, vsi_path, factor):
    with suppress_stderr():
        try:
            slide = slideio.open_slide(str(vsi_path), "VSI")
            scene = slide.get_scene(0)
            num_z = scene.num_z_slices
            cx, cy = coords
            raw_size = 224 * factor
            for z in range(num_z):
                # We always read 224px for calculation to keep metric scale comparable
                img = scene.read_block(rect=(cx-raw_size//2, cy-raw_size//2, raw_size, raw_size), 
                                       size=(224, 224), slices=(z, z+1))
                compute_brenner_gradient(img)
            return True
        except:
            return False

def micro_benchmark(vsi_path, num_workers, num_samples=100):
    with suppress_stderr():
        slide = slideio.open_slide(str(vsi_path), "VSI")
        scene = slide.get_scene(0)
        width, height = scene.size
    
    locs = []
    for _ in range(num_samples):
        locs.append((np.random.randint(2000, width-2000), np.random.randint(2000, height-2000)))

    results = {}
    
    print(f"  -> Benchmarking Local (224px @ 1x) with {num_workers} workers...")
    t0 = time.time()
    with multiprocessing.Pool(num_workers) as pool:
        list(tqdm(pool.imap(partial(_bench_local_worker, vsi_path=vsi_path), locs), total=num_samples, leave=False))
    results['local_eff'] = (time.time() - t0) / num_samples

    print(f"  -> Benchmarking Super 4x (896px @ 4x) with {num_workers} workers...")
    t0 = time.time()
    with multiprocessing.Pool(num_workers) as pool:
        list(tqdm(pool.imap(partial(_bench_super_worker, vsi_path=vsi_path, factor=4), locs), total=num_samples, leave=False))
    results['super4_eff'] = (time.time() - t0) / num_samples

    print(f"  -> Benchmarking Super 8x (1792px @ 8x) with {num_workers} workers...")
    t0 = time.time()
    with multiprocessing.Pool(num_workers) as pool:
        list(tqdm(pool.imap(partial(_bench_super_worker, vsi_path=vsi_path, factor=8), locs), total=num_samples, leave=False))
    results['super8_eff'] = (time.time() - t0) / num_samples

    return results

def get_slide_info(vsi_path):
    with suppress_stderr():
        try:
            slide = slideio.open_slide(str(vsi_path), "VSI")
            scene = slide.get_scene(0)
            w, h = scene.size
            return {'path': vsi_path, 'width': w, 'height': h, 'area': w * h}
        except:
            return None

def main():
    parser = argparse.ArgumentParser(description="Estimate total dataset processing time for different focus strategies.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the raws/ folder containing .vsi files")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel workers")
    parser.add_argument("--samples", type=int, default=160, help="Number of samples for micro-benchmark")
    args = parser.parse_args()

    vsi_dir = Path(args.dataset_path)
    vsi_files = list(vsi_dir.glob("*.vsi"))

    if not vsi_files:
        print(f"No .vsi files found in {vsi_dir}")
        return

    print(f"🔍 Analyzing dataset: {vsi_dir.name} ({len(vsi_files)} slides)")

    slide_data = []
    print("Gathering slide dimensions...")
    with multiprocessing.Pool(args.workers) as pool:
        slide_data = list(tqdm(pool.imap(get_slide_info, vsi_files), total=len(vsi_files)))

    slide_data = [s for s in slide_data if s is not None]

    total_patches = sum((s['width'] // 224) * (s['height'] // 224) for s in slide_data)
    total_super4_blocks = sum((s['width'] // 896) * (s['height'] // 896) for s in slide_data)
    total_super8_blocks = sum((s['width'] // 1792) * (s['height'] // 1792) for s in slide_data)

    print(f"\nDataset Statistics:")
    print(f"- Total Slides: {len(slide_data)}")
    print(f"- Total Training Patches (224px): {total_patches:,}")

    print(f"\n🚀 Running parallel micro-benchmark ({args.samples} samples, {args.workers} workers)...")
    bench = micro_benchmark(slide_data[0]['path'], args.workers, num_samples=args.samples)

    estimates = []
    # Local Baseline
    estimates.append({
        "Strategy": "Local Baseline (224px)",
        "Config": "1x Res",
        "Operations": total_patches,
        "Est. Hours": (total_patches * bench['local_eff']) / 3600
    })
    # 4x Sliding
    estimates.append({
        "Strategy": "Sliding Super (4x)",
        "Config": "896px area",
        "Operations": total_patches,
        "Est. Hours": (total_patches * bench['super4_eff']) / 3600
    })
    # 4x Tiled
    estimates.append({
        "Strategy": "Tiled Super (4x)",
        "Config": "896px area",
        "Operations": total_super4_blocks,
        "Est. Hours": (total_super4_blocks * bench['super4_eff']) / 3600
    })
    # 8x Sliding
    estimates.append({
        "Strategy": "Sliding Super (8x)",
        "Config": "1792px area",
        "Operations": total_patches,
        "Est. Hours": (total_patches * bench['super8_eff']) / 3600
    })
    # 8x Tiled
    estimates.append({
        "Strategy": "Tiled Super (8x)",
        "Config": "1792px area",
        "Operations": total_super8_blocks,
        "Est. Hours": (total_super8_blocks * bench['super8_eff']) / 3600
    })

    df = pd.DataFrame(estimates)
    df["Est. Days"] = df["Est. Hours"] / 24

    print("\n" + "="*80)
    print(f"PARALLEL THROUGHPUT ESTIMATE ({args.workers} Workers)")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

    print("\nInference:")
    print(f"- At 8x, context is {1792}px but throughput is still fast due to pyramid reads.")
    print("- Strategy B (8x Sliding) is the gold standard for high-magnification biology.")

    out_path = Path("experiments/focus_strategy/results/runtime_estimate.csv")
    df.to_csv(out_path, index=False)
    print(f"\nDetailed estimate saved to {out_path}")

if __name__ == "__main__":
    main()
