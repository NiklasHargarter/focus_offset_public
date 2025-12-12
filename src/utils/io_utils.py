import os
import sys
from contextlib import contextmanager


@contextmanager
def suppress_stderr():
    """
    Context manager to suppress stderr (including C-level output).
    Useful for silencing libtiff warnings from slideio.
    """
    try:
        # Open /dev/null
        with open(os.devnull, "w") as devnull:
            # Save original stderr fd
            old_stderr_fd = os.dup(sys.stderr.fileno())
            try:
                # Redirect stderr to /dev/null
                os.dup2(devnull.fileno(), sys.stderr.fileno())
                yield
            finally:
                # Restore original stderr
                os.dup2(old_stderr_fd, sys.stderr.fileno())
                os.close(old_stderr_fd)
    except Exception:
        # Fallback if specific fd operations fail
        yield
