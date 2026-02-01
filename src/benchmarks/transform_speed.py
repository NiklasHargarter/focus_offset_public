import torch
import time
import pywt


def haar_dwt_torch(x):
    """
    Simple 2D Haar DWT implementation in PyTorch.
    x: (B, C, H, W)
    Returns: (B, C*4, H/2, W/2) corresponding to LL, LH, HL, HH
    """
    b, c, h, w = x.shape

    # Reshape to (b, c, h/2, 2, w/2, 2)
    x = x.view(b, c, h // 2, 2, w // 2, 2)

    # Average and difference along rows (last dim)
    x0 = x[..., 0]
    x1 = x[..., 1]
    L = x0 + x1  # Scale later
    H = x0 - x1

    # Now columns (h//2, 2) -> (h//2)
    # L is (b, c, h/2, 2, w/2)

    LL = L[:, :, :, 0, :] + L[:, :, :, 1, :]
    LH = L[:, :, :, 0, :] - L[:, :, :, 1, :]
    HL = H[:, :, :, 0, :] + H[:, :, :, 1, :]
    HH = H[:, :, :, 0, :] - H[:, :, :, 1, :]

    # Stack channels: (b, c, 4, h/2, w/2) -> (b, c*4, h/2, w/2)
    # Ensure dimensions are integers
    h_out = int(h // 2)
    w_out = int(w // 2)
    out = torch.stack([LL, LH, HL, HH], dim=2)
    out = out.reshape(b, c * 4, h_out, w_out)

    return out * 0.5  # Normalization


def benchmark():
    B, C, H, W = 64, 3, 224, 224
    iterations = 100
    warmup = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on {device}")

    x = torch.randn(B, C, H, W, device=device)

    # --- PyTorch FFT ---
    # Warmup
    for _ in range(warmup):
        _ = torch.fft.fft2(x).abs()

    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.time()
    for _ in range(iterations):
        _ = torch.fft.fft2(x).abs()

    torch.cuda.synchronize() if device.type == "cuda" else None
    end = time.time()
    fft_time = (end - start) / iterations
    print(f"PyTorch FFT ({device}): {fft_time * 1000:.3f} ms/batch")

    # --- PyTorch DWT ---
    # Warmup
    for _ in range(warmup):
        _ = haar_dwt_torch(x)

    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.time()
    for _ in range(iterations):
        _ = haar_dwt_torch(x)

    torch.cuda.synchronize() if device.type == "cuda" else None
    end = time.time()
    dwt_time = (end - start) / iterations
    print(f"PyTorch DWT ({device}): {dwt_time * 1000:.3f} ms/batch")

    # --- PyWavelets (CPU only) ---
    x_np = x.cpu().numpy()

    start = time.time()
    # Approx batch processing loop since pywt is usually per-image or needs specific handling
    # We'll just do one image and multiply to be generous to CPU, or loop if fast enough
    # Pywavelets generally runs on CPU.
    # Benchmarking single image * Batch size to simulate DataLoader worker load

    x_single = x_np[0]  # (C, H, W)
    t0 = time.time()
    pywt.dwt2(x_single, "haar", axes=(-2, -1))
    t1 = time.time()

    pywt_time_per_image = t1 - t0
    pywt_batch_time = pywt_time_per_image * B
    print(
        f"PyWavelets DWT (CPU, estimated batch): {pywt_batch_time * 1000:.3f} ms/batch"
    )

    # --- Combined Overhead ---
    total_gpu_time = fft_time + dwt_time
    print(f"\nTotal Transform Overhead (GPU): {total_gpu_time * 1000:.3f} ms/batch")

    # Check if this overhead fits within typical batch times
    # A ResNet50 batch takes ~200-500ms depending on GPU.
    # If overhead is < 10-20ms, it's negligible.


if __name__ == "__main__":
    benchmark()
