# Multiprocessing Fork Safety in Python DataLoaders

This document details the findings regarding `multiprocessing` crash issues when using C-extension libraries (like `slideio`, `opencv`, `openslide`) with PyTorch `DataLoader`.

## The Problem: Shared File Descriptors
When using `multiprocessing` with the default `fork` start method (standard on Linux), child processes inherit the memory and file descriptors (FDs) of the parent process.

If a file is opened in the parent process (e.g., inside `Dataset.__init__`), the File Descriptor is duplicated in the child processes but points to the **same Open File Description** in the OS kernel. This means they share the **File Offset (Cursor)**.

### Symptoms
- Random crashes with exit code `1` or `-11` (SIGSEGV).
- Library errors like `Not a JPEG file`, `Premature end of JPEG file`.
- Data corruption where workers read data from wrong locations.

### Proof of Inheritance
We created a reproduction script `experiments/reproduce_fd_inheritance.py` that confirms the FD is identical:
```
[Parent] Open FDs: ... 3 -> .../slide.vsi
[Child]  Inherited: ... 3 -> .../slide.vsi
```
This confirms that if *any* process reads (and thus seeks), the file pointer moves for *all* processes.

## The Solution: Lazy Initialization
To ensure thread/process safety, each worker must have its own independent File Descriptor.

### Incorrect Pattern (Crash Prone)
```python
class UnsafeDataset(Dataset):
    def __init__(self, path):
        # Opened in Parent -> Shared FD -> Race Condition
        self.slide = slideio.open_slide(path, "VSI")
```

### Correct Pattern (Fork-Safe)
```python
class SafeDataset(Dataset):
    def __init__(self, path):
        self.path = path
        # Do NOT open here
        
    def __getitem__(self, idx):
        # Open lazily inside the worker process
        if not hasattr(self, 'slide'):
            self.slide = slideio.open_slide(self.path, "VSI")
            
        return self.slide.read_block(...)
```

By opening the file inside `__getitem__` (or a `worker_init_fn`), the `open()` syscall is executed by the worker, granting it a unique, independent file descriptor.

## Related Experiments
- `experiments/reproduce_dataloader_crash.py`: Demonstrates the crash when using the Unsafe Pattern.
- `experiments/benchmark_performance_v2.py`: Demonstrates high throughput using the Correct Pattern.
