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
        with open(os.devnull, "w") as devnull:
            old_stderr_fd = os.dup(sys.stderr.fileno())
            try:
                os.dup2(devnull.fileno(), sys.stderr.fileno())
                yield
            finally:
                os.dup2(old_stderr_fd, sys.stderr.fileno())
                os.close(old_stderr_fd)
    except Exception:
        yield
