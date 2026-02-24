import json
import os

from src.datasets.zstack_ihc.config import ZStackIHCConfig
from src.datasets.zstack_he.prep.preprocess import (
    _process_slide_worker,
    _run_split,
)
from functools import partial
from multiprocessing import Pool


def preprocess_dataset(
    dataset_name: str = "ZStack_IHC",
    workers: int | None = None,
    exclude: str = "_all_",
    dry_run: bool = False,
) -> None:
    cfg = ZStackIHCConfig()
    workers = workers or os.cpu_count() or 1

    if not cfg.split_path.exists():
        print(f"Error: split file {cfg.split_path} not found. Run create_split first.")
        return

    splits = json.loads(cfg.split_path.read_text())
    train_names = set(splits.get("train_pool", []))
    test_names = set(splits.get("test", []))

    all_files = sorted(cfg.raw_dir.glob("*.vsi"))
    if exclude:
        all_files = [f for f in all_files if exclude.lower() not in f.name.lower()]
    if dry_run:
        all_files = all_files[:10]

    train_files = [f for f in all_files if f.name in train_names]
    test_files = [f for f in all_files if f.name in test_names]

    print(
        f"Preprocessing {dataset_name} | stride={cfg.prep.stride} patch={cfg.prep.patch_size}"
    )
    print(f"Train: {len(train_files)} slides  Test: {len(test_files)} slides")

    process_func = partial(_process_slide_worker, cfg=cfg.prep, dry_run=dry_run)

    with Pool(workers) as pool:
        _run_split(pool, train_files, process_func, cfg.get_train_index_path(), "train")
        _run_split(pool, test_files, process_func, cfg.get_test_index_path(), "test")
