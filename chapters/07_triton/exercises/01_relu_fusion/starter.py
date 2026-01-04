"""
Exercise 01: Fused ReLU Matrix Multiplication - Starter Code

TODO: Implement a kernel that fuses matmul with ReLU activation.
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
    TODO: Implement fused matmul + ReLU

    Compute: C = ReLU(A @ B) = max(0, A @ B)

    Hints:
    1. Start with blocked matmul (accumulate over K blocks)
    2. After accumulation, apply ReLU before storing
    3. Use tl.maximum(accumulator, 0) for ReLU
    """
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # TODO: Create offsets for M and N dimensions
    offs_m = # YOUR CODE HERE
    offs_n = # YOUR CODE HERE

    # TODO: Initialize accumulator
    accumulator = # YOUR CODE HERE

    # TODO: Loop over K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        # TODO: Get K offsets
        offs_k = # YOUR CODE HERE

        # TODO: Create pointers for A and B blocks
        a_ptrs = # YOUR CODE HERE
        b_ptrs = # YOUR CODE HERE

        # TODO: Create masks
        mask_a = # YOUR CODE HERE
        mask_b = # YOUR CODE HERE

        # TODO: Load blocks
        a = # YOUR CODE HERE
        b = # YOUR CODE HERE

        # TODO: Accumulate matmul result
        accumulator += # YOUR CODE HERE

    # TODO: Apply ReLU activation
    c = # YOUR CODE HERE (apply ReLU to accumulator)

    # TODO: Store result
    offs_cm = # YOUR CODE HERE
    offs_cn = # YOUR CODE HERE
    c_ptrs = # YOUR CODE HERE
    mask_c = # YOUR CODE HERE
    # YOUR CODE HERE (store c)


def triton_matmul_relu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for fused matmul + ReLU kernel.
    """
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
    """Test your implementation."""
    print("Testing Correctness")
    print("=" * 70)

    test_cases = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 384, 256),
    ]

    for M, N, K in test_cases:
        a = torch.randn(M, K, device='cuda', dtype=torch.float16)
        b = torch.randn(K, N, device='cuda', dtype=torch.float16)

        result_triton = triton_matmul_relu(a, b)
        result_torch = pytorch_matmul_relu(a, b)

        if torch.allclose(result_triton, result_torch, rtol=1e-2, atol=1e-2):
            print(f"  ({M:3d}x{K:3d}) @ ({K:3d}x{N:3d}): PASS")
        else:
            print(f"  ({M:3d}x{K:3d}) @ ({K:3d}x{N:3d}): FAIL")
            print(f"    Max diff: {(result_triton - result_torch).abs().max().item():.6f}")
            return False

    print("All tests passed!")
    return True


def benchmark():
    """Benchmark fused vs unfused."""
    print("\nBenchmarking")
    print("=" * 70)

    M, N, K = 1024, 1024, 1024
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

    print(f"Fused:    {fused_time:6.3f} ms")
    print(f"Unfused:  {unfused_time:6.3f} ms")
    print(f"Speedup:  {unfused_time / fused_time:.2f}x")

    if unfused_time / fused_time >= 1.3:
        print("\n✓ Performance goal achieved (1.3x+ speedup)!")
    else:
        print("\n✗ Performance goal not met (need 1.3x+ speedup)")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. This exercise requires a GPU.")
        exit(1)

    print("Exercise 01: Fused ReLU Matrix Multiplication")
    print("=" * 70)
    print("TODO: Implement the matmul_relu_kernel function")
    print()

    # Uncomment when ready to test
    # if test_correctness():
    #     benchmark()
