"""
Exercise 01: Fused ReLU Matrix Multiplication - Solution

This solution demonstrates how to fuse matrix multiplication with ReLU activation.
"""

import torch
import triton
import triton.language as tl
import time


@triton.jit
def matmul_relu_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused matmul + ReLU kernel.

    Computes: C = ReLU(A @ B) = max(0, A @ B)
    """
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create offsets for M and N dimensions
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        # Get K offsets
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)

        # Create pointers for A and B blocks
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # Create masks
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load blocks
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        # Accumulate matmul result
        accumulator += tl.dot(a, b)

    # Apply ReLU activation (key fusion step!)
    # ReLU(x) = max(0, x)
    accumulator_relu = tl.maximum(accumulator, 0.0)

    # Convert to output dtype
    c = accumulator_relu.to(tl.float16)

    # Store result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=mask_c)


def triton_matmul_relu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for fused matmul + ReLU kernel."""
    assert a.shape[1] == b.shape[0]
    assert a.is_cuda and b.is_cuda

    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32

    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )

    matmul_relu_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    )

    return c


def pytorch_matmul_relu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Unfused baseline."""
    return torch.relu(torch.matmul(a, b))


def test_correctness():
    """Test correctness."""
    print("Testing Correctness")
    print("=" * 70)

    test_cases = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 384, 256),
        (1024, 1024, 1024),
    ]

    for M, N, K in test_cases:
        a = torch.randn(M, K, device='cuda', dtype=torch.float16)
        b = torch.randn(K, N, device='cuda', dtype=torch.float16)

        result_triton = triton_matmul_relu(a, b)
        result_torch = pytorch_matmul_relu(a, b)

        if torch.allclose(result_triton, result_torch, rtol=1e-2, atol=1e-2):
            print(f"  ({M:4d}x{K:4d}) @ ({K:4d}x{N:4d}): PASS")
        else:
            print(f"  ({M:4d}x{K:4d}) @ ({K:4d}x{N:4d}): FAIL")
            max_diff = (result_triton - result_torch).abs().max().item()
            print(f"    Max diff: {max_diff:.6f}")
            return False

    print("All tests passed!")
    return True


def benchmark():
    """Benchmark fused vs unfused."""
    print("\nBenchmarking")
    print("=" * 70)

    sizes = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]

    for M, N, K in sizes:
        print(f"\nSize: ({M}x{K}) @ ({K}x{N})")

        a = torch.randn(M, K, device='cuda', dtype=torch.float16)
        b = torch.randn(K, N, device='cuda', dtype=torch.float16)

        # Warmup
        for _ in range(10):
            _ = triton_matmul_relu(a, b)
            _ = pytorch_matmul_relu(a, b)
        torch.cuda.synchronize()

        # Benchmark
        num_iterations = 100

        start = time.time()
        for _ in range(num_iterations):
            result_fused = triton_matmul_relu(a, b)
        torch.cuda.synchronize()
        fused_time = (time.time() - start) / num_iterations * 1000

        start = time.time()
        for _ in range(num_iterations):
            result_unfused = pytorch_matmul_relu(a, b)
        torch.cuda.synchronize()
        unfused_time = (time.time() - start) / num_iterations * 1000

        print(f"  Fused:    {fused_time:6.3f} ms")
        print(f"  Unfused:  {unfused_time:6.3f} ms")
        print(f"  Speedup:  {unfused_time / fused_time:.2f}x")


def explain_solution():
    """Explain the solution."""
    print("\nSolution Explanation")
    print("=" * 70)

    print("Key Changes from Standard Matmul:")
    print()
    print("1. After accumulation loop completes:")
    print("   accumulator = tl.zeros(...)")
    print("   for k in range(...):")
    print("       accumulator += tl.dot(a, b)")
    print()
    print("2. Apply ReLU before storing:")
    print("   accumulator_relu = tl.maximum(accumulator, 0.0)")
    print("   c = accumulator_relu.to(tl.float16)")
    print()
    print("3. Store as normal:")
    print("   tl.store(c_ptrs, c, mask=mask_c)")
    print()
    print("Benefits:")
    print("- No intermediate matmul result written to memory")
    print("- ReLU applied in registers (very fast)")
    print("- One kernel instead of two (less overhead)")
    print("- Expected speedup: 1.3-2x vs unfused")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. This solution requires a GPU.")
        exit(1)

    print("Exercise 01 Solution: Fused ReLU Matrix Multiplication")
    print("=" * 70)

    explain_solution()

    if test_correctness():
        benchmark()

    print("\n" + "=" * 70)
    print("Exercise completed successfully!")
    print("=" * 70)
