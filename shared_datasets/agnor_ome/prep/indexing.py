import json
import os
import random
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from shared_datasets.agnor_ome.prep.preprocess import process_ome_slide


def _run_split(pool: Pool, files: list[Path], process_func, out_path: Path, label: str):
    if not files:
        return
    print(f">> Processing {label} split ({len(files)} slides)...")
    rows = [row for res in pool.map(process_func, files) for row in res]
    if rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(out_path)
        print(f"  Saved {len(rows)} rows -> {out_path}")


def create_ome_split(
    slide_dir: Path,
    split_path: Path,
    exclude_pattern: str = "_all_",
    split_ratio: float = 0.3,
    seed: int = 42,
):
    all_files = sorted(
        [
            f.name
            for f in slide_dir.glob("*.ome.tif*")
            if exclude_pattern.lower() not in f.name.lower()
        ]
    )
    if not all_files:
        raise RuntimeError(f"No OME-TIFF files found in {slide_dir}")
    random.seed(seed)
    random.shuffle(all_files)
    num_test = max(1, int(len(all_files) * split_ratio))
    data = {
        "test": sorted(all_files[:num_test]),
        "train_pool": sorted(all_files[num_test:]),
        "seed": seed,
        "total_slides": len(all_files),
    }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Created new split file: {split_path}")
    return data


def index_ome_dataset(
    slide_dir: Path,
    index_dir: Path,
    split_path: Path,
    params: dict,
    workers: int | None = None,
    dry_run: bool = False,
):
    splits = (
        create_ome_split(slide_dir, split_path, params.get("exclude_pattern", "_all_"))
        if not split_path.exists()
        else json.loads(split_path.read_text())
    )
    train_names, test_names = (
        set(splits.get("train_pool", [])),
        set(splits.get("test", [])),
    )
    all_files = sorted(
        list(slide_dir.glob("*.ome.tiff")) + list(slide_dir.glob("*.ome.tif"))
    )

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
    process_func = partial(process_ome_slide, params=params, dry_run=dry_run)
    with Pool(workers or os.cpu_count() or 1) as pool:
        _run_split(
            pool, train_files, process_func, index_dir / "train.parquet", "train"
        )
        _run_split(pool, test_files, process_func, index_dir / "test.parquet", "test")
