import slideio
import os

import sys

# Demonstrates File Descriptor (FD) inheritance across fork
# Usage: python reproduce_fd_inheritance.py [path_to_vsi]

DEFAULT_VSI = "data_train/001_1_HE_stack.vsi"


def main():
    vsi_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VSI

    if not os.path.exists(vsi_file):
        print(f"File {vsi_file} not found.")
        return

    basename = os.path.basename(vsi_file).replace(".vsi", "")
    print(f"[Parent {os.getpid()}] Opening slide {vsi_file}...")

    _slide = slideio.open_slide(vsi_file, "VSI")

    # Check open FDs in Parent (Filter for relevant files)
    print(f"\n[Parent {os.getpid()}] Relevant Open FDs (grep '{basename}'):")
    os.system(f"ls -l /proc/{os.getpid()}/fd | grep '{basename}'")

    print(
        f"\n[Parent {os.getpid()}] Forking child to check inherited FDs (expecting EXACT match)..."
    )

    pid = os.fork()
    if pid == 0:
        # Child
        print(f"\n[Child {os.getpid()}] Inherited FDs (Should match Parent):")
        os.system(f"ls -l /proc/{os.getpid()}/fd | grep '{basename}'")

        print("\n[Child] Exiting.")
        os._exit(0)
    else:
        os.waitpid(pid, 0)
        print("\n[Parent] Child finished.")


if __name__ == "__main__":
    main()
