# Multiprocessing Fork Safety in Python DataLoaders

This document details the findings regarding `multiprocessing` crash issues when using `slideio` with PyTorch `DataLoader`.

## The Problem: Shared File Descriptors
When using `multiprocessing` with the default `fork` start method (standard on Linux), child processes inherit the memory and file descriptors (FDs) of the parent process.

If a file is opened in the parent process (e.g., inside `Dataset.__init__`), the File Descriptor is duplicated in the child processes but points to the **same Open File Description** in the OS kernel. This means they share the **File Offset (Cursor)**.

### Explaining the File Offset (Cursor)
The **File Offset** is a pointer that indicates the current position in the file where the next read or write operation will occur.

When a File Descriptor is shared across processes (inherited via `fork`), this offset is **also shared** because it is stored in the open file description in the kernel, not in the process itself.
- If **Worker 1** reads 100 bytes, the offset moves forward by 100 bytes (`0 -> 100`).
- If **Worker 2** tries to read the file header, it unknowingly starts reading at `100` instead of `0`.
- This causes "Seek Races" where independent workers fight over the position of the file cursor, leading to them reading header bytes as data or vice versa.

### Symptoms
- Random crashes with exit code `1` or `-11` (SIGSEGV).
- Library errors like `Not a JPEG file`, `Premature end of JPEG file`.
- Data corruption where workers read data from wrong locations.

### Proof of Inheritance
We created a reproduction script `reproduce_fd_inheritance.py` that confirms the FD is identical:
```
[Parent] Open FDs: ... 3 -> .../_001_1_HE_stack_/stack1/frame_t_0.ets
[Child]  Inherited: ... 3 -> .../_001_1_HE_stack_/stack1/frame_t_0.ets
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

## Verdict
This crash is **not a bug in the library**, but a fundamental consequence of **POSIX fork semantics** and how OS file descriptors work.

