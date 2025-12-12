import slideio
import os
import random

import sys

# Minimal reproduction of slideio crash with fork()
# Usage: python reproduce_child_crash.py [path_to_vsi]

DEFAULT_VSI = "data_train/001_1_HE_stack.vsi"


def main():
    vsi_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VSI

    if not os.path.exists(vsi_file):
        print(f"File {vsi_file} not found.")
        return

    print(f"[Parent {os.getpid()}] Opening slide {vsi_file}...")
    slide = slideio.open_slide(vsi_file, "VSI")
    scene = slide.get_scene(0)

    num_children = 10
    print(
        f"[Parent {os.getpid()}] Forking {num_children} children. They will read CONCURRENTLY from inherited FD..."
    )

    pids = []
    for i in range(num_children):
        pid = os.fork()
        if pid == 0:
            # Child
            try:
                # Loop reads to maximize race condition window
                # Each read seeks the SHARED file pointer, causing chaos
                for j in range(200):
                    x = random.randint(0, 5000)
                    # Read is ignored, but the act of reading creates the race condition
                    _ = scene.read_block(rect=(x, 0, 100, 100), slices=(0, 1))
                os._exit(0)
            except Exception as e:
                print(f"[Child {os.getpid()}] CRASHED/ERROR: {e}")
                os._exit(1)
        else:
            pids.append(pid)

    failures = 0
    for p in pids:
        _, status = os.waitpid(p, 0)
        ec = os.waitstatus_to_exitcode(status)
        if ec != 0:
            failures += 1
            print(f"Child {p} failed with exit code {ec}")

    if failures > 0:
        print(f"\nReproduced: {failures}/{num_children} children crashed or errored.")
    else:
        print("\nAll children survived (Race condition might need more load).")


if __name__ == "__main__":
    main()
