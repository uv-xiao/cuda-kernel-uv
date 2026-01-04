"""
Comprehensive Autotuning Example

This example demonstrates:
- How to use @triton.autotune decorator
- Configuring multiple parameters
- Understanding autotune behavior
- Best practices for autotuning

We'll implement a simple kernel and show how autotuning finds the best configuration.
"""

import torch
import triton
import triton.language as tl
import time


# Example 1: Basic Autotuning
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],  # Autotune based on input size
)
@triton.jit
def add_kernel_autotuned(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple vector addition with autotuning."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)


# Example 2: Multi-parameter Autotuning
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N'],  # Autotune based on matrix shape
)
@triton.jit
def matrix_op_autotuned(
    a_ptr, b_ptr, output_ptr,
    M, N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Element-wise matrix operation with autotuning."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
    b_ptrs = b_ptr + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    a = tl.load(a_ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptrs, mask=mask, other=0.0)

    # Some computation
    output = a * b + a + b

    output_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(output_ptrs, output, mask=mask)


def explain_autotune_decorator():
    """Explain the autotune decorator."""
    print("\nUnderstanding @triton.autotune")
    print("=" * 70)

    print("Basic structure:")
    print("""
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr):
    ...
""")

    print("\nComponents:")
    print()
    print("1. configs: List[triton.Config]")
    print("   - Each Config specifies:")
    print("     - Kernel parameters (dict): BLOCK_SIZE, etc.")
    print("     - num_warps: Number of warps (32 threads each)")
    print("     - num_stages: Pipeline stages for software pipelining")
    print()

    print("2. key: List[str]")
    print("   - Arguments to use for caching best config")
    print("   - Example: ['n_elements'] caches by input size")
    print("   - Can be multiple: ['M', 'N', 'K']")
    print()

    print("How it works:")
    print("1. First call: Benchmark all configs for given key values")
    print("2. Cache best config based on key")
    print("3. Subsequent calls: Use cached config")
    print("4. Different key values: Benchmark again")


def demonstrate_autotuning():
    """Demonstrate autotuning in action."""
    print("\nAutotuning Demonstration")
    print("=" * 70)

    sizes = [1_000_000, 10_000_000, 100_000_000]

    print("\nFirst call for each size (autotuning happens):")
    for size in sizes:
        print(f"\n  Size: {size:,}")
        x = torch.randn(size, device='cuda')
        y = torch.randn(size, device='cuda')
        output = torch.empty_like(x)

        grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)

        print("  Autotuning...")
        start = time.time()
        add_kernel_autotuned[grid](x, y, output, size)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"  Time (includes autotuning): {elapsed*1000:.3f} ms")

    print("\n\nSecond call for same sizes (uses cache):")
    for size in sizes:
        print(f"\n  Size: {size:,}")
        x = torch.randn(size, device='cuda')
        y = torch.randn(size, device='cuda')
        output = torch.empty_like(x)

        grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)

        start = time.time()
        add_kernel_autotuned[grid](x, y, output, size)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"  Time (cached): {elapsed*1000:.3f} ms")


def benchmark_with_vs_without_autotune():
    """Compare autotuned vs fixed block size."""
    print("\nBenchmark: Autotuned vs Fixed Block Size")
    print("=" * 70)

    @triton.jit
    def add_kernel_fixed(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        """Fixed block size version."""
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y

        tl.store(output_ptr + offsets, output, mask=mask)

    sizes = [1_000_000, 10_000_000, 100_000_000]

    for size in sizes:
        print(f"\nSize: {size:,}")

        x = torch.randn(size, device='cuda')
        y = torch.randn(size, device='cuda')

        # Autotuned version
        output_auto = torch.empty_like(x)
        grid_auto = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)

        # Warmup + autotune
        add_kernel_autotuned[grid_auto](x, y, output_auto, size)
        torch.cuda.synchronize()

        # Benchmark
        num_iterations = 100
        start = time.time()
        for _ in range(num_iterations):
            add_kernel_autotuned[grid_auto](x, y, output_auto, size)
        torch.cuda.synchronize()
        auto_time = (time.time() - start) / num_iterations * 1000

        # Fixed block size versions
        block_sizes = [256, 512, 1024]
        fixed_times = {}

        for bs in block_sizes:
            output_fixed = torch.empty_like(x)
            grid_fixed = (triton.cdiv(size, bs),)

            # Warmup
            add_kernel_fixed[grid_fixed](x, y, output_fixed, size, bs)
            torch.cuda.synchronize()

            # Benchmark
            start = time.time()
            for _ in range(num_iterations):
                add_kernel_fixed[grid_fixed](x, y, output_fixed, size, bs)
            torch.cuda.synchronize()
            fixed_times[bs] = (time.time() - start) / num_iterations * 1000

        print(f"  Autotuned:     {auto_time:.3f} ms")
        for bs, t in fixed_times.items():
            print(f"  Fixed (BS={bs:4d}): {t:.3f} ms  ({auto_time/t*100:.1f}% of auto)")


def best_practices():
    """Print best practices for autotuning."""
    print("\nAutotuning Best Practices")
    print("=" * 70)

    print("1. Choose Relevant Keys:")
    print("   - Include parameters that affect performance")
    print("   - Common: matrix dimensions, tensor sizes")
    print("   - Don't include: pointers, strides (constant effect)")
    print()

    print("2. Select Good Configs:")
    print("   - Start with common values (powers of 2)")
    print("   - Include range: small (64) to large (1024)")
    print("   - Balance: more configs = better but slower first call")
    print()

    print("3. Tune num_warps:")
    print("   - More warps: better occupancy, more parallelism")
    print("   - Fewer warps: less overhead, faster sync")
    print("   - Typical: 2, 4, 8 (powers of 2)")
    print()

    print("4. Tune num_stages:")
    print("   - Higher: better latency hiding, more registers")
    print("   - Lower: less resource usage, better occupancy")
    print("   - Typical: 2-5")
    print()

    print("5. Caching Behavior:")
    print("   - First call: slow (benchmarking)")
    print("   - Subsequent: fast (cached)")
    print("   - Cache persists across runs (file-based)")
    print()

    print("6. Debugging:")
    print("   - Set TRITON_PRINT_AUTOTUNING=1 to see process")
    print("   - Cache location: ~/.triton/cache/")
    print("   - Clear cache to re-tune")


def advanced_config_example():
    """Show advanced configuration."""
    print("\nAdvanced Configuration Example")
    print("=" * 70)

    print("""
@triton.autotune(
    configs=[
        # Optimize for small matrices
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32},
            num_warps=2,
            num_stages=2,
        ),
        # Optimize for medium matrices
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=3,
        ),
        # Optimize for large matrices
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=8,
            num_stages=4,
        ),
        # Optimize for very large matrices
        triton.Config(
            {'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64},
            num_warps=8,
            num_stages=5,
            num_ctas=1,  # Number of CTAs (thread blocks)
        ),
    ],
    key=['M', 'N', 'K'],  # Tune based on all dimensions
    reset_to_zero=['c_ptr'],  # Zero these pointers before each benchmark
    restore_value=['c_ptr'],  # Restore after benchmark
)
@triton.jit
def advanced_kernel(a_ptr, b_ptr, c_ptr, M, N, K, ...):
    ...
""")


if __name__ == "__main__":
    print("Comprehensive Autotuning Example")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available. This example requires a GPU.")
        exit(1)

    # 1. Explain decorator
    explain_autotune_decorator()

    # 2. Demonstrate autotuning
    demonstrate_autotuning()

    # 3. Benchmark
    benchmark_with_vs_without_autotune()

    # 4. Best practices
    best_practices()

    # 5. Advanced example
    advanced_config_example()

    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("1. Autotuning finds best config automatically")
    print("2. First call is slow (benchmarking), subsequent calls are fast")
    print("3. Choose relevant 'key' parameters for caching")
    print("4. Balance number of configs vs first-call overhead")
    print("5. Typical speedup: 10-50% over fixed config")
    print("=" * 70)
