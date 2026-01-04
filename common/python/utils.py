"""
Common utilities for CUDA kernel tutorial Python examples.
"""

import time
from typing import Callable, Optional, Tuple

import numpy as np
import torch


def get_device_info() -> dict:
    """Get CUDA device information."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    return {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_gb": props.total_memory / (1024**3),
        "multiprocessor_count": props.multi_processor_count,
        "max_threads_per_block": props.max_threads_per_block,
        "warp_size": props.warp_size,
    }


def print_device_info():
    """Print CUDA device information."""
    info = get_device_info()
    if "error" in info:
        print(info["error"])
        return

    print(f"Device: {info['name']}")
    print(f"  Compute capability: {info['compute_capability']}")
    print(f"  Total memory: {info['total_memory_gb']:.2f} GB")
    print(f"  Number of SMs: {info['multiprocessor_count']}")
    print(f"  Max threads per block: {info['max_threads_per_block']}")
    print(f"  Warp size: {info['warp_size']}")


class GpuTimer:
    """GPU timer using CUDA events."""

    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        self.start_event.record()

    def stop(self):
        self.end_event.record()

    def elapsed_ms(self) -> float:
        self.end_event.synchronize()
        return self.start_event.elapsed_time(self.end_event)


def benchmark_kernel(
    kernel_fn: Callable,
    warmup_runs: int = 5,
    timed_runs: int = 20,
    name: Optional[str] = None,
) -> Tuple[float, float, float]:
    """
    Benchmark a kernel function.

    Args:
        kernel_fn: Function to benchmark (should execute the kernel)
        warmup_runs: Number of warmup iterations
        timed_runs: Number of timed iterations
        name: Optional name for printing

    Returns:
        Tuple of (avg_ms, min_ms, max_ms)
    """
    # Warmup
    for _ in range(warmup_runs):
        kernel_fn()
    torch.cuda.synchronize()

    # Timed runs
    timer = GpuTimer()
    times = []

    for _ in range(timed_runs):
        timer.start()
        kernel_fn()
        timer.stop()
        times.append(timer.elapsed_ms())

    avg_ms = np.mean(times)
    min_ms = np.min(times)
    max_ms = np.max(times)

    if name:
        print(f"{name}: avg={avg_ms:.3f} ms, min={min_ms:.3f} ms, max={max_ms:.3f} ms")

    return avg_ms, min_ms, max_ms


def compute_gemm_flops(M: int, N: int, K: int, time_ms: float) -> float:
    """Compute GFLOPS for GEMM operation."""
    flops = 2.0 * M * N * K  # multiply-add
    gflops = flops / (time_ms * 1e6)  # Convert to GFLOPS
    return gflops


def compute_bandwidth(bytes_transferred: int, time_ms: float) -> float:
    """Compute bandwidth in GB/s."""
    gb = bytes_transferred / (1024**3)
    seconds = time_ms / 1000.0
    return gb / seconds


def verify_result(
    expected: np.ndarray,
    actual: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    name: str = "Result",
) -> bool:
    """
    Verify two arrays are equal within tolerance.

    Args:
        expected: Expected values
        actual: Actual values
        rtol: Relative tolerance
        atol: Absolute tolerance
        name: Name for printing

    Returns:
        True if verification passes
    """
    try:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
        print(f"{name}: PASSED (shape={expected.shape})")
        return True
    except AssertionError as e:
        print(f"{name}: FAILED")
        print(str(e)[:500])  # Print first 500 chars of error
        return False


def verify_result_torch(
    expected: torch.Tensor,
    actual: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    name: str = "Result",
) -> bool:
    """Verify two torch tensors are equal within tolerance."""
    passed = torch.allclose(actual, expected, rtol=rtol, atol=atol)
    if passed:
        print(f"{name}: PASSED (shape={tuple(expected.shape)})")
    else:
        diff = (actual - expected).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"{name}: FAILED (max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e})")
    return passed


def random_tensor(
    *shape,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
    low: float = -1.0,
    high: float = 1.0,
) -> torch.Tensor:
    """Create a random tensor."""
    return torch.rand(*shape, dtype=dtype, device=device) * (high - low) + low


def zeros_tensor(
    *shape,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
) -> torch.Tensor:
    """Create a zeros tensor."""
    return torch.zeros(*shape, dtype=dtype, device=device)


def print_tensor_stats(tensor: torch.Tensor, name: str = "Tensor"):
    """Print statistics about a tensor."""
    print(f"{name}:")
    print(f"  Shape: {tuple(tensor.shape)}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Min: {tensor.min().item():.6f}")
    print(f"  Max: {tensor.max().item():.6f}")
    print(f"  Mean: {tensor.mean().item():.6f}")
    print(f"  Std: {tensor.std().item():.6f}")


if __name__ == "__main__":
    print_device_info()
