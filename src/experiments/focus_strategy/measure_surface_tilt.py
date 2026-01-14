import argparse
import numpy as np
import pandas as pd
import slideio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.dataset.vsi_prep.preprocess import detect_tissue, create_focal_map
from src.utils.io_utils import suppress_stderr

def analyze_surface(vsi_path: Path, output_dir: Path, downscales: list[int]):
    print(f"Analyzing Focus Surface: {vsi_path.name}")

    with suppress_stderr():
        slide = slideio.open_slide(str(vsi_path), "VSI")
    scene = slide.get_scene(0)
    width, height = scene.size

    patch_sizes = [224, 448, 896, 1792]

    _, mask_4x = detect_tissue(scene, downscale=4)

    all_results = []

    for ds in downscales:
        print(f"\n{'='*20}\nTesting Downscale: {ds}x\n{'='*20}")

        for fps in patch_sizes:
            print(f"Processing Patch Size: {fps}px")

            z_grid = create_focal_map(scene, focus_patch_size=fps, downscale=ds)

            points = []
            grid_h, grid_w = z_grid.shape

            for gy in range(grid_h):
                for gx in range(grid_w):

                    px, py = gx * fps, gy * fps

                    mx, my = px // 4, py // 4
                    mw, mh = fps // 4, fps // 4

                    if my < mask_4x.shape[0] and mx < mask_4x.shape[1]:
                        sub_mask = mask_4x[my:min(my+mh, mask_4x.shape[0]), mx:min(mx+mw, mask_4x.shape[1])]
                        if sub_mask.size > 0 and np.any(sub_mask > 0):
                            points.append([px, py, z_grid[gy, gx]])

            if len(points) < 5:
                continue

            pts = np.array(points)
            X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]

            A_mat = np.c_[X, Y, np.ones(X.shape[0])]
            coeffs, _, _, _ = np.linalg.lstsq(A_mat, Z, rcond=None)
            A, B, C = coeffs

            Z_pred = A_mat @ coeffs
            residuals = Z - Z_pred
            rmse = np.sqrt(np.mean(residuals**2))

            all_results.append({
                "Downscale": ds,
                "Patch_Size": fps,
                "Tilt_X": A * 10000,
                "Tilt_Y": B * 10000,
                "RMSE": rmse,
                "Point_Count": len(points)
            })

    df = pd.DataFrame(all_results)
    print("\n--- Surface Stability Comparison ---")
    if not df.empty:
        pivot_rmse = df.pivot(index="Patch_Size", columns="Downscale", values="RMSE")
        print("\nRMSE Matrix (Focus Stability):")
        print(pivot_rmse)

        print("\nFull Data (Tilt per 10k pixels):")
        print(df.drop(columns=["Point_Count"]).to_string(index=False))

    out_file = output_dir / f"surface_matrix_{vsi_path.stem}.csv"
    df.to_csv(out_file, index=False)
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("vsi_path", type=str)
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    analyze_surface(Path(args.vsi_path), out_dir, downscales=[4])
