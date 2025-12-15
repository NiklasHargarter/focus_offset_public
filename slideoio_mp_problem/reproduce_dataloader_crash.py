import os
import slideio
from torch.utils.data import Dataset, DataLoader

import sys

# Minimal reproduction of DataLoader crash when opening slide in __init__
# This pattern creates unsafe sharing of C++ slideio objects across processes.
# Usage: python reproduce_dataloader_crash.py [path_to_vsi]

DEFAULT_VSI = "/home/niklas/ZStack_HE/raws/001_1_HE_stack.vsi"


class UnsafeDataset(Dataset):
    def __init__(self, vsi_path):
        self.vsi_path = vsi_path

        print(f"[Main PID {os.getpid()}] Opening slide in __init__ (UNSAFE PATTERN)")
        # Open slide in parent process
        # This object contains internal C++ state (file handles, caches) that is NOT fork-safe
        self.slide = slideio.open_slide(vsi_path, "VSI")
        self.scene = self.slide.get_scene(0)

        # Generate non-overlapping patch coordinates
        self.patch_size = 224
        self.patches = []
        w, h = self.scene.size
        for y in range(0, h - self.patch_size, self.patch_size):
            for x in range(0, w - self.patch_size, self.patch_size):
                self.patches.append((x, y))

        # Limit to reasonable amount to avoid extremely long loops if file is huge
        if len(self.patches) > 5000:
            self.patches = self.patches[:5000]

        print(f"Generated {len(self.patches)} non-overlapping patches.")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        # Workers inherit self.scene from parent via fork
        # Multiple workers accessing self.scene simultaneously causes race conditions on the underlying file descriptor
        try:
            x, y = self.patches[idx]
            # Read specific patch
            block = self.scene.read_block(
                rect=(x, y, self.patch_size, self.patch_size), slices=(0, 1)
            )
            return block.shape
        except Exception as e:
            print(f"[Worker PID {os.getpid()}] Error reading patch {idx}: {e}")
            raise e


def main():
    vsi_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VSI

    if not os.path.exists(vsi_file):
        print(f"File {vsi_file} not found. Please provide path to a valid VSI file.")
        return

    print(f"Using file: {vsi_file}")

    # Create dataset (opens slide now in Main process)
    ds = UnsafeDataset(vsi_file)

    # Create DataLoader with workers
    # num_workers > 0 enables multiprocessing
    print("Creating DataLoader with num_workers=8...")
    loader = DataLoader(ds, batch_size=8, num_workers=8)

    print("Starting iteration. This is expected to crash or hang...")
    try:
        for i, batch in enumerate(loader):
            if i % 10 == 0:
                print(f"Batch {i} processed")
    except Exception as e:
        print(f"\nCaught Exception: {e}")


if __name__ == "__main__":
    main()
