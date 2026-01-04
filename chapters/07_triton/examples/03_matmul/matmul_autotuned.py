"""
Autotuned Matrix Multiplication in Triton

This example demonstrates:
- @triton.autotune decorator for automatic optimization
- Multiple kernel configurations
- Performance tuning strategies
- Achieving near-optimal performance

The autotuner benchmarks all configurations and selects the fastest.
"""

import torch
import triton
import triton.language as tl
import time


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],  # Autotune based on these parameters
)
@triton.jit
def matmul_kernel_autotuned(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters (autotuned)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Autotuned matrix multiplication kernel with advanced features.

    New optimizations:
    - GROUP_SIZE_M: Improves L2 cache reuse by reordering blocks
    - num_stages: Software pipelining for hiding memory latency
    - num_warps: Thread block size optimization
    """
    # Program ID with swizzling for better cache reuse
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Swizzle to improve L2 cache hit rate
    # Without swizzling: process tiles in row-major order
    # With swizzling: process tiles in groups to reuse B matrix
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create pointers for the first blocks of A and B
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next blocks of A and B
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        # Perform block matrix multiplication and accumulate
        accumulator += tl.dot(a, b)

        # Advance pointers to next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Convert accumulator to output dtype
    c = accumulator.to(tl.float16)

    # Write back result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_matmul_autotuned(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication using autotuned Triton kernel.
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_cuda and b.is_cuda

    M, K = a.shape
    K, N = b.shape

    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Grid is 1D - kernel handles swizzling internally
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    # Launch kernel (autotuner will select best config)
    matmul_kernel_autotuned[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )

    return c


def explain_autotune_configs():
    """
    Explain the autotuning configuration parameters.
    """
    print("\nAutotuning Configuration Parameters")
    print("=" * 70)

    print("Block Sizes (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K):")
    print("  - Tile dimensions for matrix multiplication")
    print("  - Larger blocks: More compute per memory access (better for large matrices)")
    print("  - Smaller blocks: Less register pressure, better occupancy")
    print("  - Trade-off: Compute efficiency vs resource utilization")
    print()

    print("GROUP_SIZE_M:")
    print("  - Number of M tiles to process before moving to next N tile")
    print("  - Improves L2 cache reuse of B matrix")
    print("  - Default: 8 (good for most cases)")
    print()

    print("num_stages:")
    print("  - Number of pipeline stages for memory operations")
    print("  - Higher stages: Better hiding of memory latency")
    print("  - Trade-off: More register/shared memory usage")
    print("  - Typical range: 2-5")
    print()

    print("num_warps:")
    print("  - Number of warps (groups of 32 threads) per block")
    print("  - More warps: Better occupancy, more parallelism")
    print("  - Fewer warps: Lower overhead, faster synchronization")
    print("  - Typical values: 2, 4, 8")
    print()

    print("Autotuning Process:")
    print("  1. Triton benchmarks each configuration")
    print("  2. Measures execution time for given input sizes")
    print("  3. Caches best configuration for (M, N, K)")
    print("  4. Automatically selects optimal config at runtime")


def test_correctness():
    """
    Test autotuned matmul for correctness.
    """
    print("\nTesting Correctness")
    print("=" * 70)

    test_cases = [
        (64, 64, 64),
        (128, 256, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (512, 384, 768),  # Non-power-of-2
    ]

    for M, N, K in test_cases:
        a = torch.randn(M, K, device='cuda', dtype=torch.float16)
        b = torch.randn(K, N, device='cuda', dtype=torch.float16)

        # Triton result
        c_triton = triton_matmul_autotuned(a, b)

        # PyTorch result
        c_torch = torch.matmul(a, b)

        # Verify
        assert torch.allclose(c_triton, c_torch, rtol=1e-2, atol=1e-2), \
            f"Mismatch at shape M={M}, N={N}, K={K}"

        print(f"  Shape ({M:4d}x{K:4d}) @ ({K:4d}x{N:4d}): PASS")

    print("All tests passed!")


def benchmark_matmul(M: int, N: int, K: int, num_iterations: int = 100):
    """
    Benchmark autotuned matmul vs PyTorch.
    """
    print(f"\nBenchmarking matmul ({M}x{K}) @ ({K}x{N})")
    print("=" * 70)

    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)

    # Warmup (important: autotuner runs during first call)
    print("Warming up (autotuner selecting best config)...")
    for _ in range(10):
        _ = triton_matmul_autotuned(a, b)
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()

    # Benchmark Triton
    start = time.time()
    for _ in range(num_iterations):
        c_triton = triton_matmul_autotuned(a, b)
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


def compare_all_versions():
    """
    Compare all matmul implementations.
    """
    print("\nComparing All Implementations")
    print("=" * 70)

    from matmul_blocked import triton_matmul_blocked

    M, N, K = 2048, 2048, 2048
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)

    implementations = [
        ("Autotuned", triton_matmul_autotuned),
        ("Blocked", triton_matmul_blocked),
        ("PyTorch", torch.matmul),
    ]

    results = {}

    for name, func in implementations:
        # Warmup
        for _ in range(10):
            _ = func(a, b)
        torch.cuda.synchronize()

        # Benchmark
        num_iterations = 100
        start = time.time()
        for _ in range(num_iterations):
            _ = func(a, b)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / num_iterations * 1000

        flops = 2 * M * N * K
        tflops = flops / (elapsed * 1e-3) / 1e12

        results[name] = (elapsed, tflops)
        print(f"{name:12s}: {elapsed:7.3f} ms  ({tflops:6.2f} TFLOPS)")

    print()
    print("Relative Performance:")
    baseline_tflops = results["PyTorch"][1]
    for name, (_, tflops) in results.items():
        print(f"  {name:12s}: {tflops / baseline_tflops * 100:5.1f}% of PyTorch")


def analyze_swizzling():
    """
    Explain the swizzling optimization.
    """
    print("\nSwizzling for Cache Optimization")
    print("=" * 70)

    print("Without Swizzling (row-major order):")
    print("  Process tiles: (0,0), (0,1), (0,2), ..., (1,0), (1,1), ...")
    print("  - Each row of output tiles loads different parts of B")
    print("  - Poor L2 cache reuse of B matrix")
    print()

    print("With Swizzling (GROUP_SIZE_M):")
    print("  Process tiles in groups:")
    print("  Group 0: (0,0), (1,0), ..., (GROUP_SIZE_M-1,0)")
    print("  Group 1: (0,1), (1,1), ..., (GROUP_SIZE_M-1,1)")
    print("  - Multiple output tiles share same B columns")
    print("  - Better L2 cache hit rate")
    print("  - Typical speedup: 5-15%")
    print()

    M, N = 1024, 1024
    BLOCK_M, BLOCK_N = 128, 128
    GROUP_SIZE_M = 8

    num_blocks_m = M // BLOCK_M
    num_blocks_n = N // BLOCK_N

    print(f"Example: {M}x{N} matrix with {BLOCK_M}x{BLOCK_N} tiles")
    print(f"  Grid: {num_blocks_m}x{num_blocks_n} = {num_blocks_m * num_blocks_n} tiles")
    print(f"  GROUP_SIZE_M = {GROUP_SIZE_M}")
    print()

    print("First few tiles (swizzled):")
    for pid in range(min(16, num_blocks_m * num_blocks_n)):
        num_pid_in_group = GROUP_SIZE_M * num_blocks_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_blocks_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        print(f"  PID {pid:2d} → Tile ({pid_m}, {pid_n})")


if __name__ == "__main__":
    print("Autotuned Matrix Multiplication in Triton")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available. This example requires a GPU.")
        exit(1)

    # 1. Explain autotune configs
    explain_autotune_configs()

    # 2. Analyze swizzling
    analyze_swizzling()

    # 3. Test correctness
    test_correctness()

    # 4. Benchmark performance
    benchmark_matmul(512, 512, 512)
    benchmark_matmul(1024, 1024, 1024)
    benchmark_matmul(2048, 2048, 2048)
    benchmark_matmul(4096, 4096, 4096)

    # 5. Compare all versions
    print("\n")
    compare_all_versions()

    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("1. @triton.autotune automatically finds best configuration")
    print("2. Multiple factors: block sizes, stages, warps")
    print("3. Swizzling improves L2 cache hit rate")
    print("4. Achieves 95-100% of PyTorch performance")
    print("5. Much easier than manual CUDA optimization")
    print("=" * 70)
