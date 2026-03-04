import argparse
import os
from pathlib import Path
from multiprocessing import Pool
from functools import partial


from shared_datasets.vsi.prep.indexing import create_vsi_split, _run_split
from synthetic_data.prep.preprocess_synthetic import process_vsi_slide_synthetic


def index_vsi_synthetic_dataset(
    slide_dir: Path,
    index_dir: Path,
    split_path: Path,
    params: dict,
    workers: int | None = None,
    dry_run: bool = False,
):
    import json

    # We reuse existing split format
    splits = (
        create_vsi_split(slide_dir, split_path, params.get("exclude_pattern", "_all_"))
        if not split_path.exists()
        else json.loads(split_path.read_text())
    )

    train_names, test_names = (
        set(splits.get("train_pool", [])),
        set(splits.get("test", [])),
    )

    all_files = sorted(slide_dir.glob("*.vsi"))
    if "exclude_pattern" in params:
        all_files = [
            f
            for f in all_files
            if params["exclude_pattern"].lower() not in f.name.lower()
        ]

    if dry_run:
        all_files = all_files[:2]

    if train_names:
        train_files = [f for f in all_files if f.name in train_names]
    else:
        train_files = all_files

    test_files = [f for f in all_files if f.name in test_names]

    process_func = partial(
        process_vsi_slide_synthetic,
        patch_size_n=params["patch_size_n"],
        downsample=params["downsample"],
        stride=params["stride"],
        min_coverage=params["cov"],
        dry_run=dry_run,
    )

    with Pool(workers or os.cpu_count() or 1) as pool:
        _run_split(
            pool,
            train_files,
            process_func,
            index_dir / "train_synthetic.parquet",
            "train",
        )
        _run_split(
            pool, test_files, process_func, index_dir / "test_synthetic.parquet", "test"
        )


def main():
    parser = argparse.ArgumentParser("Prepare Synthetic VSI Index")
    parser.add_argument("--slide_dir", type=str, required=True)
    parser.add_argument("--index_dir", type=str, required=True)
    parser.add_argument("--split_path", type=str, required=True)
    parser.add_argument("--patch_size_n", type=int, default=1024)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--cov", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    params = {
        "patch_size_n": args.patch_size_n,
        "downsample": args.downsample,
        "stride": args.stride,
        "cov": args.cov,
    }

    index_vsi_synthetic_dataset(
        Path(args.slide_dir),
        Path(args.index_dir),
        Path(args.split_path),
        params,
        args.workers,
        args.dry_run,
    )


if __name__ == "__main__":
    main()
