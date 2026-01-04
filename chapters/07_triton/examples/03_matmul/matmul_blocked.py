"""
Blocked Matrix Multiplication in Triton

This example demonstrates:
- Tiling in all three dimensions (M, N, K)
- Accumulation pattern for blocked computation
- Memory reuse through blocking
- Significant performance improvement

C = A @ B where A is MxK, B is KxN, C is MxN
"""

import torch
import triton
import triton.language as tl
import time


@triton.jit
def matmul_kernel_blocked(
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
    Blocked matrix multiplication kernel.

    Key improvement: Loop over K dimension in blocks, accumulating partial results.
    This enables:
    - Handling arbitrary K sizes
    - Better memory reuse
    - More efficient SRAM usage
    """
    # Program IDs for output tile
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for output tile
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Initialize accumulator for this output tile
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        # Current K block offsets
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)

        # Pointers to blocks of A and B
        # A block: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        # B block: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # Create masks
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load blocks
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        # Accumulate partial result
        accumulator += tl.dot(a, b)

    # Convert accumulator to output type
    c = accumulator.to(tl.float16)  # Or keep as float32

    # Write result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)

    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)


def triton_matmul_blocked(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication using blocked Triton kernel.

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

    # Define block sizes (tunable parameters)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32  # Key difference: K is also blocked!

    # Calculate grid dimensions
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )

    # Launch kernel
    matmul_kernel_blocked[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    )

    return c


def visualize_blocking():
    """
    Visualize how blocked matmul works.
    """
    print("\nBlocked Matrix Multiplication")
    print("=" * 70)

    M, N, K = 256, 256, 1024
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32

    print(f"Matrix dimensions: C({M}x{N}) = A({M}x{K}) @ B({K}x{N})")
    print(f"Block sizes: M={BLOCK_M}, N={BLOCK_N}, K={BLOCK_K}")
    print()

    # Calculate number of blocks
    num_blocks_m = triton.cdiv(M, BLOCK_M)
    num_blocks_n = triton.cdiv(N, BLOCK_N)
    num_blocks_k = triton.cdiv(K, BLOCK_K)

    print(f"Grid dimensions: {num_blocks_m}x{num_blocks_n} = "
          f"{num_blocks_m * num_blocks_n} output tiles")
    print(f"K dimension blocks: {num_blocks_k}")
    print()

    # Show computation for one output tile
    pid_m, pid_n = 0, 0
    print(f"Computation for output tile ({pid_m}, {pid_n}):")
    print()

    m_start = pid_m * BLOCK_M
    m_end = min(m_start + BLOCK_M, M)
    n_start = pid_n * BLOCK_N
    n_end = min(n_start + BLOCK_N, N)

    print(f"Output: C[{m_start}:{m_end}, {n_start}:{n_end}]")
    print()
    print("Computed by accumulating K blocks:")

    for k_block in range(num_blocks_k):
        k_start = k_block * BLOCK_K
        k_end = min(k_start + BLOCK_K, K)

        print(f"  Block {k_block}: "
              f"A[{m_start}:{m_end}, {k_start}:{k_end}] @ "
              f"B[{k_start}:{k_end}, {n_start}:{n_end}]")

    print()
    print("Total operations per output tile:")
    print(f"  {num_blocks_k} loads of A ({BLOCK_M}x{BLOCK_K})")
    print(f"  {num_blocks_k} loads of B ({BLOCK_K}x{BLOCK_N})")
    print(f"  {num_blocks_k} matrix multiplies (accumulate)")
    print(f"  1 store of C ({BLOCK_M}x{BLOCK_N})")


def test_correctness():
    """
    Test blocked matmul for correctness.
    """
    print("\nTesting Correctness")
    print("=" * 70)

    test_cases = [
        (64, 64, 64),
        (128, 256, 512),
        (1024, 1024, 1024),
        (100, 200, 300),  # Non-power-of-2
        (1, 1024, 1),
        (1024, 1, 1024),
    ]

    for M, N, K in test_cases:
        a = torch.randn(M, K, device='cuda', dtype=torch.float16)
        b = torch.randn(K, N, device='cuda', dtype=torch.float16)

        # Triton result
        c_triton = triton_matmul_blocked(a, b)

        # PyTorch result
        c_torch = torch.matmul(a, b)

        # Verify
        assert torch.allclose(c_triton, c_torch, rtol=1e-2, atol=1e-2), \
            f"Mismatch at shape M={M}, N={N}, K={K}"

        print(f"  Shape ({M:4d}x{K:4d}) @ ({K:4d}x{N:4d}): PASS")

    print("All tests passed!")


def benchmark_matmul(M: int, N: int, K: int, num_iterations: int = 100):
    """
    Benchmark blocked Triton matmul vs PyTorch.
    """
    print(f"\nBenchmarking matmul ({M}x{K}) @ ({K}x{N})")
    print("=" * 70)

    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)

    # Warmup
    for _ in range(10):
        _ = triton_matmul_blocked(a, b)
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()

    # Benchmark Triton
    start = time.time()
    for _ in range(num_iterations):
        c_triton = triton_matmul_blocked(a, b)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / num_iterations * 1000

    # Benchmark PyTorch
    start = time.time()
    for _ in range(num_iterations):
        c_torch = torch.matmul(a, b)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / num_iterations * 1000

    # Calculate FLOPS
    flops = 2 * M * N * K
    triton_tflops = flops / (triton_time * 1e-3) / 1e12
    torch_tflops = flops / (torch_time * 1e-3) / 1e12

    print(f"Triton:  {triton_time:7.3f} ms  ({triton_tflops:6.2f} TFLOPS)")
    print(f"PyTorch: {torch_time:7.3f} ms  ({torch_tflops:6.2f} TFLOPS)")
    print(f"Speedup: {torch_time / triton_time:.2f}x")
    print(f"Efficiency: {triton_tflops / torch_tflops * 100:.1f}% of PyTorch")


def analyze_memory_reuse():
    """
    Analyze memory access patterns and reuse.
    """
    print("\nMemory Reuse Analysis")
    print("=" * 70)

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    K = 1024
    num_k_blocks = K // BLOCK_K

    print("For one output tile computation:")
    print()

    # Data loaded from A
    a_loads = num_k_blocks  # Number of times we load A blocks
    a_elements_per_load = BLOCK_M * BLOCK_K
    a_total_elements = a_loads * a_elements_per_load
    a_unique_elements = BLOCK_M * K  # Actual unique elements in this A row

    print(f"Matrix A:")
    print(f"  Loads: {a_loads}")
    print(f"  Elements per load: {a_elements_per_load}")
    print(f"  Total elements loaded: {a_total_elements:,}")
    print(f"  Unique elements: {a_unique_elements:,}")
    print(f"  Reuse factor: {a_total_elements / a_unique_elements:.2f}x")
    print()

    # Data loaded from B
    b_loads = num_k_blocks
    b_elements_per_load = BLOCK_K * BLOCK_N
    b_total_elements = b_loads * b_elements_per_load
    b_unique_elements = K * BLOCK_N

    print(f"Matrix B:")
    print(f"  Loads: {b_loads}")
    print(f"  Elements per load: {b_elements_per_load}")
    print(f"  Total elements loaded: {b_total_elements:,}")
    print(f"  Unique elements: {b_unique_elements:,}")
    print(f"  Reuse factor: {b_total_elements / b_unique_elements:.2f}x")
    print()

    # Compute operations
    compute_ops = 2 * BLOCK_M * BLOCK_N * K  # Multiply-add for each output
    memory_bytes = (a_unique_elements + b_unique_elements) * 2  # fp16

    print(f"Compute intensity:")
    print(f"  Operations: {compute_ops:,}")
    print(f"  Memory bytes: {memory_bytes:,}")
    print(f"  Arithmetic intensity: {compute_ops / memory_bytes:.2f} ops/byte")
    print()
    print("Note: Arithmetic intensity > 1.0 means compute-bound")
    print("      (good for utilizing GPU compute resources)")


def compare_naive_vs_blocked():
    """
    Compare naive and blocked implementations.
    """
    print("\nNaive vs Blocked Comparison")
    print("=" * 70)

    print("Naive Implementation:")
    print("  - Loads entire K dimension at once")
    print("  - BLOCK_SIZE_K = K")
    print("  - Single tl.dot() call per output tile")
    print("  - Limitation: K must fit in SRAM")
    print("  - Max K ≈ 1024-2048 (depending on BLOCK_M, BLOCK_N)")
    print()

    print("Blocked Implementation:")
    print("  - Loops over K in chunks")
    print("  - BLOCK_SIZE_K = 32-64 (tunable)")
    print("  - Multiple tl.dot() calls, accumulate results")
    print("  - Handles arbitrary K sizes")
    print("  - Better memory reuse")
    print()

    print("Performance:")
    print("  - Naive: Good for small K, simple to understand")
    print("  - Blocked: Scales to large K, production-ready")
    print("  - Blocked typically 2-10x faster for K > 1024")


if __name__ == "__main__":
    print("Blocked Matrix Multiplication in Triton")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available. This example requires a GPU.")
        exit(1)

    # 1. Visualize blocking strategy
    visualize_blocking()

    # 2. Analyze memory reuse
    analyze_memory_reuse()

    # 3. Compare with naive
    compare_naive_vs_blocked()

    # 4. Test correctness
    test_correctness()

    # 5. Benchmark performance
    benchmark_matmul(512, 512, 512)
    benchmark_matmul(1024, 1024, 1024)
    benchmark_matmul(2048, 2048, 2048)
    benchmark_matmul(4096, 4096, 4096)

    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("1. Block K dimension to handle arbitrary sizes")
    print("2. Accumulate partial results in loop")
    print("3. Better memory reuse improves performance")
    print("4. Arithmetic intensity > 1.0 (compute-bound)")
    print("5. See matmul_autotuned.py for automatic optimization")
    print("=" * 70)
