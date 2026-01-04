"""
Naive Matrix Multiplication in Triton

This example demonstrates:
- 2D block-based computation
- Basic tiling strategy
- Baseline performance

This is the starting point before optimizations.
Matrix multiplication: C = A @ B where A is MxK, B is KxN, C is MxN
"""

import torch
import triton
import triton.language as tl
import time


@triton.jit
def matmul_kernel_naive(
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
    Naive matrix multiplication kernel.

    Each program computes a BLOCK_SIZE_M x BLOCK_SIZE_N tile of C.
    This version loads all K elements at once (requires K to be small).
    """
    # Get program IDs for this output tile
    pid_m = tl.program_id(0)  # Which row block
    pid_n = tl.program_id(1)  # Which column block

    # Create offsets for the output tile
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Create 2D offset grids using broadcasting
    # offs_am has shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
    # offs_bn has shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
    offs_am = offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    offs_bn = offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Pointers to blocks of A and B
    a_ptrs = a_ptr + offs_am
    b_ptrs = b_ptr + offs_bn

    # Create masks for boundary conditions
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_k = offs_k < K

    # 2D masks using broadcasting
    mask_a = mask_m[:, None] & mask_k[None, :]
    mask_b = mask_k[:, None] & mask_n[None, :]

    # Load blocks of A and B
    a = tl.load(a_ptrs, mask=mask_a, other=0.0)
    b = tl.load(b_ptrs, mask=mask_b, other=0.0)

    # Compute matrix multiplication for this tile
    c = tl.dot(a, b)  # This is the key operation!

    # Create output pointers and mask
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_c = offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_ptrs = c_ptr + offs_c

    mask_c = (offs_cm < M)[:, None] & (offs_cn < N)[None, :]

    # Store result
    tl.store(c_ptrs, c, mask=mask_c)


def triton_matmul_naive(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication using naive Triton kernel.

    Args:
        a: Matrix of shape (M, K)
        b: Matrix of shape (K, N)

    Returns:
        c: Matrix of shape (M, N)
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_cuda and b.is_cuda

    M, K = a.shape
    K, N = b.shape

    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Define block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = K  # Naive: load entire K dimension

    # Calculate grid dimensions
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )

    # Launch kernel
    matmul_kernel_naive[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    )

    return c


def visualize_tiling():
    """
    Visualize how matrix multiplication is tiled.
    """
    print("\nMatrix Multiplication Tiling")
    print("=" * 70)

    M, N, K = 256, 192, 128
    BLOCK_M, BLOCK_N = 64, 64

    print(f"Matrix dimensions: C({M}x{N}) = A({M}x{K}) @ B({K}x{N})")
    print(f"Block size: {BLOCK_M}x{BLOCK_N}")
    print()

    num_blocks_m = triton.cdiv(M, BLOCK_M)
    num_blocks_n = triton.cdiv(N, BLOCK_N)
    total_blocks = num_blocks_m * num_blocks_n

    print(f"Grid dimensions: {num_blocks_m} x {num_blocks_n} = {total_blocks} blocks")
    print()

    print("Block assignment:")
    for pid_m in range(num_blocks_m):
        for pid_n in range(num_blocks_n):
            m_start = pid_m * BLOCK_M
            m_end = min(m_start + BLOCK_M, M)
            n_start = pid_n * BLOCK_N
            n_end = min(n_start + BLOCK_N, N)

            print(f"  Block ({pid_m},{pid_n}): "
                  f"C[{m_start}:{m_end}, {n_start}:{n_end}] = "
                  f"A[{m_start}:{m_end}, 0:{K}] @ B[0:{K}, {n_start}:{n_end}]")


def test_correctness():
    """
    Test matrix multiplication for correctness.
    """
    print("\nTesting Correctness")
    print("=" * 70)

    test_cases = [
        (64, 64, 64),    # Small square
        (128, 256, 64),  # Rectangle
        (100, 100, 100), # Non-power-of-2
        (1, 128, 1),     # Edge case
        (128, 1, 128),   # Edge case
    ]

    for M, N, K in test_cases:
        a = torch.randn(M, K, device='cuda')
        b = torch.randn(K, N, device='cuda')

        # Triton result
        c_triton = triton_matmul_naive(a, b)

        # PyTorch result
        c_torch = torch.matmul(a, b)

        # Verify
        assert torch.allclose(c_triton, c_torch, rtol=1e-4, atol=1e-4), \
            f"Mismatch at shape M={M}, N={N}, K={K}"

        print(f"  Shape ({M:3d}x{K:3d}) @ ({K:3d}x{N:3d}): PASS")

    print("All tests passed!")


def benchmark_matmul(M: int, N: int, K: int, num_iterations: int = 100):
    """
    Benchmark naive Triton matmul vs PyTorch.
    """
    print(f"\nBenchmarking matmul ({M}x{K}) @ ({K}x{N})")
    print("=" * 70)

    a = torch.randn(M, K, device='cuda')
    b = torch.randn(K, N, device='cuda')

    # Warmup
    for _ in range(10):
        _ = triton_matmul_naive(a, b)
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()

    # Benchmark Triton
    start = time.time()
    for _ in range(num_iterations):
        c_triton = triton_matmul_naive(a, b)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / num_iterations * 1000

    # Benchmark PyTorch
    start = time.time()
    for _ in range(num_iterations):
        c_torch = torch.matmul(a, b)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / num_iterations * 1000

    # Calculate FLOPS
    # Matrix multiply: 2*M*N*K operations (multiply-add for each element)
    flops = 2 * M * N * K
    triton_tflops = flops / (triton_time * 1e-3) / 1e12
    torch_tflops = flops / (torch_time * 1e-3) / 1e12

    print(f"Triton:  {triton_time:7.3f} ms  ({triton_tflops:6.2f} TFLOPS)")
    print(f"PyTorch: {torch_time:7.3f} ms  ({torch_tflops:6.2f} TFLOPS)")
    print(f"Speedup: {torch_time / triton_time:.2f}x")
    print(f"Efficiency: {triton_tflops / torch_tflops * 100:.1f}% of PyTorch")


def explain_dot_operation():
    """
    Explain the tl.dot operation.
    """
    print("\nUnderstanding tl.dot()")
    print("=" * 70)

    print("tl.dot(a, b) computes matrix multiplication of two blocks:")
    print()
    print("  a: (BLOCK_SIZE_M, BLOCK_SIZE_K)")
    print("  b: (BLOCK_SIZE_K, BLOCK_SIZE_N)")
    print("  result: (BLOCK_SIZE_M, BLOCK_SIZE_N)")
    print()
    print("Equivalent to:")
    print("  for i in range(BLOCK_SIZE_M):")
    print("    for j in range(BLOCK_SIZE_N):")
    print("      result[i, j] = sum(a[i, k] * b[k, j] for k in range(BLOCK_SIZE_K))")
    print()
    print("tl.dot() is a primitive that compiles to tensor cores (if available)")
    print("or optimized CUDA cores operations.")
    print()
    print("Limitations of naive implementation:")
    print("  1. BLOCK_SIZE_K = K means we load entire rows/columns at once")
    print("  2. For large K, this exceeds SRAM capacity")
    print("  3. No reuse of loaded data across different output tiles")
    print()
    print("Solution: Blocked matmul (see matmul_blocked.py)")


if __name__ == "__main__":
    print("Naive Matrix Multiplication in Triton")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available. This example requires a GPU.")
        exit(1)

    # 1. Visualize tiling
    visualize_tiling()

    # 2. Explain tl.dot
    explain_dot_operation()

    # 3. Test correctness
    test_correctness()

    # 4. Benchmark (small sizes only - naive doesn't scale)
    benchmark_matmul(256, 256, 256)
    benchmark_matmul(512, 512, 512)
    # Note: Larger sizes will be slow or OOM

    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("1. Each program computes a BLOCK_M x BLOCK_N tile of output")
    print("2. tl.dot(a, b) performs block matrix multiplication")
    print("3. Naive version loads entire K dimension (doesn't scale)")
    print("4. Need blocking in K dimension for efficiency")
    print("5. See matmul_blocked.py for improved version")
    print("=" * 70)
