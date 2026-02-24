from src.datasets.zstack_he import get_dataloaders as get_he_loaders
from src.datasets.zstack_ihc import get_dataloaders as get_ihc_loaders
from src.config import TrainConfig


def test_loader(name, loader_func, train_cfg):
    print(f"\n--- Testing {name} ---", flush=True)
    try:
        print(f"Initializing loaders for {name}...", flush=True)
        train_loader, val_loader = loader_func(train_cfg)
        print(
            f"Loaders initialized. Train: {len(train_loader)} batches, Val: {len(val_loader)} batches",
            flush=True,
        )

        print(f"Retrieving first batch from {name} train_loader...", flush=True)
        batch = next(iter(train_loader))
        print("Batch retrieved successfully!", flush=True)
        print(f"Batch keys: {batch.keys()}", flush=True)
        print(f"Image shape: {batch['image'].shape}", flush=True)
        print(f"Target shape: {batch['target'].shape}", flush=True)
        print(f"Metadata sample: {batch['metadata']['filename'][0]}", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed during {name} testing: {e}", flush=True)
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("Starting Verbose Smoke Test", flush=True)
    train_cfg = TrainConfig(batch_size=4, num_workers=0)

    test_loader("ZStack_HE", get_he_loaders, train_cfg)
    test_loader("ZStack_IHC", get_ihc_loaders, train_cfg)

    print("\nSmoke Test Complete", flush=True)
