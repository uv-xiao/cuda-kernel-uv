"""
Exercise 1: Tiled Reduction - Solution
=======================================

Complete implementation of tiled reduction using TileLang.
"""

import tilelang as T
import torch
import time


@T.prim_func
def tiled_reduction(
    A: T.Buffer((1048576,), "float32"),
    partial_sums: T.Buffer((4096,), "float32")
):
    """
    Tiled reduction with three-level hierarchy:
    1. Thread-level: Each thread accumulates multiple elements
    2. Block-level: Parallel reduction in shared memory
    3. Grid-level: Partial sums written to global memory
    """
    N = 1048576
    BLOCK_SIZE = 256
    ELEMENTS_PER_THREAD = 4

    with T.block("root"):
        # Thread organization
        bx = T.thread_binding(0, N // (BLOCK_SIZE * ELEMENTS_PER_THREAD), "blockIdx.x")
        tx = T.thread_binding(0, BLOCK_SIZE, "threadIdx.x")

        # Shared memory for block reduction
        shared = T.alloc_shared([BLOCK_SIZE], "float32")

        # Register for thread-local sum
        thread_sum = T.alloc_fragment([1], "float32")
        thread_sum[0] = 0.0

        # Step 1: Each thread accumulates ELEMENTS_PER_THREAD elements
        base_idx = (bx * BLOCK_SIZE + tx) * ELEMENTS_PER_THREAD
        for i in T.serial(ELEMENTS_PER_THREAD):
            idx = base_idx + i
            if idx < N:
                thread_sum[0] = thread_sum[0] + A[idx]

        # Step 2: Store thread sum in shared memory
        shared[tx] = thread_sum[0]
        T.sync_threads()

        # Step 3: Parallel tree reduction in shared memory
        stride = BLOCK_SIZE // 2
        while stride > 0:
            if tx < stride:
                shared[tx] = shared[tx] + shared[tx + stride]
            T.sync_threads()
            stride = stride // 2

        # Step 4: Thread 0 writes the block's partial sum
        if tx == 0:
            partial_sums[bx] = shared[0]


@T.prim_func
def tiled_reduction_optimized(
    A: T.Buffer((1048576,), "float32"),
    partial_sums: T.Buffer((4096,), "float32")
):
    """
    Optimized version with vectorized loads and reduced synchronization.
    """
    N = 1048576
    BLOCK_SIZE = 256
    ELEMENTS_PER_THREAD = 4

    with T.block("root"):
        bx = T.thread_binding(0, N // (BLOCK_SIZE * ELEMENTS_PER_THREAD), "blockIdx.x")
        tx = T.thread_binding(0, BLOCK_SIZE, "threadIdx.x")

        shared = T.alloc_shared([BLOCK_SIZE], "float32")

        # Use fragment for multiple elements
        values = T.alloc_fragment([ELEMENTS_PER_THREAD], "float32")
        thread_sum = T.alloc_fragment([1], "float32")
        thread_sum[0] = 0.0

        # Load multiple elements (compiler can vectorize)
        base_idx = (bx * BLOCK_SIZE + tx) * ELEMENTS_PER_THREAD
        for i in T.serial(ELEMENTS_PER_THREAD):
            idx = base_idx + i
            if idx < N:
                values[i] = A[idx]
            else:
                values[i] = 0.0

        # Accumulate in registers
        for i in T.serial(ELEMENTS_PER_THREAD):
            thread_sum[0] = thread_sum[0] + values[i]

        shared[tx] = thread_sum[0]
        T.sync_threads()

        # Unroll first few iterations (avoids some syncs)
        if BLOCK_SIZE >= 512:
            if tx < 256:
                shared[tx] = shared[tx] + shared[tx + 256]
            T.sync_threads()

        if BLOCK_SIZE >= 256:
            if tx < 128:
                shared[tx] = shared[tx] + shared[tx + 128]
            T.sync_threads()

        if BLOCK_SIZE >= 128:
            if tx < 64:
                shared[tx] = shared[tx] + shared[tx + 64]
            T.sync_threads()

        # Final warp reduction (can be done without sync in same warp)
        if tx < 32:
            shared[tx] = shared[tx] + shared[tx + 32]
            shared[tx] = shared[tx] + shared[tx + 16]
            shared[tx] = shared[tx] + shared[tx + 8]
            shared[tx] = shared[tx] + shared[tx + 4]
            shared[tx] = shared[tx] + shared[tx + 2]
            shared[tx] = shared[tx] + shared[tx + 1]

        if tx == 0:
            partial_sums[bx] = shared[0]


def test_correctness():
    """Test correctness of reduction implementations."""
    print("="*60)
    print("Correctness Tests")
    print("="*60)

    N = 1048576
    BLOCK_SIZE = 256
    ELEMENTS_PER_THREAD = 4
    num_blocks = N // (BLOCK_SIZE * ELEMENTS_PER_THREAD)

    # Compile both versions
    mod_basic = T.compile(tiled_reduction, target="cuda")
    mod_opt = T.compile(tiled_reduction_optimized, target="cuda")

    test_cases = [
        ("All ones", torch.ones(N, device="cuda", dtype=torch.float32), float(N)),
        ("Sequential", torch.arange(N, device="cuda", dtype=torch.float32),
         float(N * (N - 1) // 2)),
        ("Random", torch.randn(N, device="cuda", dtype=torch.float32), None),
    ]

    for name, A, expected in test_cases:
        if expected is None:
            expected = A.sum().item()

        print(f"\nTest: {name}")

        # Test basic version
        partial_sums = torch.zeros(num_blocks, device="cuda", dtype=torch.float32)
        mod_basic(A, partial_sums)
        result_basic = partial_sums.sum().item()

        # Test optimized version
        partial_sums = torch.zeros(num_blocks, device="cuda", dtype=torch.float32)
        mod_opt(A, partial_sums)
        result_opt = partial_sums.sum().item()

        # Compare
        error_basic = abs(result_basic - expected)
        error_opt = abs(result_opt - expected)
        rel_error = error_basic / abs(expected) if expected != 0 else error_basic

        print(f"  Expected:        {expected:.6f}")
        print(f"  Basic:           {result_basic:.6f} (error: {error_basic:.6f})")
        print(f"  Optimized:       {result_opt:.6f} (error: {error_opt:.6f})")
        print(f"  Relative error:  {rel_error:.2e}")

        assert rel_error < 1e-3, f"Test {name} failed!"
        print(f"  ✓ Passed")


def benchmark_performance():
    """Benchmark reduction performance."""
    print("\n" + "="*60)
    print("Performance Benchmark")
    print("="*60)

    N = 1048576
    BLOCK_SIZE = 256
    ELEMENTS_PER_THREAD = 4
    num_blocks = N // (BLOCK_SIZE * ELEMENTS_PER_THREAD)

    A = torch.randn(N, device="cuda", dtype=torch.float32)
    partial_sums = torch.zeros(num_blocks, device="cuda", dtype=torch.float32)

    # Compile
    mod_basic = T.compile(tiled_reduction, target="cuda")
    mod_opt = T.compile(tiled_reduction_optimized, target="cuda")

    # Warmup
    for _ in range(100):
        _ = A.sum()
        mod_basic(A, partial_sums)
        mod_opt(A, partial_sums)
    torch.cuda.synchronize()

    # Benchmark PyTorch
    start = time.time()
    for _ in range(1000):
        _ = A.sum()
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / 1000

    # Benchmark basic
    start = time.time()
    for _ in range(1000):
        mod_basic(A, partial_sums)
    torch.cuda.synchronize()
    basic_time = (time.time() - start) / 1000

    # Benchmark optimized
    start = time.time()
    for _ in range(1000):
        mod_opt(A, partial_sums)
    torch.cuda.synchronize()
    opt_time = (time.time() - start) / 1000

    # Calculate bandwidth
    bytes_accessed = N * 4  # Read once
    pytorch_bw = bytes_accessed / pytorch_time / 1e9
    basic_bw = bytes_accessed / basic_time / 1e9
    opt_bw = bytes_accessed / opt_time / 1e9

    print(f"\nArray size: {N} elements ({N*4/1e6:.2f} MB)")
    print(f"\n{'Implementation':<20} {'Time (μs)':<12} {'BW (GB/s)':<12} {'vs PyTorch'}")
    print("-" * 60)
    print(f"{'PyTorch':<20} {pytorch_time*1e6:>10.2f}   {pytorch_bw:>10.2f}   {'100%':>10}")
    print(f"{'TileLang (Basic)':<20} {basic_time*1e6:>10.2f}   {basic_bw:>10.2f}   "
          f"{100*pytorch_time/basic_time:>9.1f}%")
    print(f"{'TileLang (Opt)':<20} {opt_time*1e6:>10.2f}   {opt_bw:>10.2f}   "
          f"{100*pytorch_time/opt_time:>9.1f}%")
    print("="*60)


def main():
    """Run all tests and benchmarks."""
    print("="*60)
    print("Exercise 1: Tiled Reduction - Solution")
    print("="*60)

    test_correctness()
    benchmark_performance()

    print("\n✓ All tests passed!")
    print("\nKey Insights:")
    print("1. Three-level hierarchy: thread -> block -> grid")
    print("2. Shared memory dramatically reduces global memory access")
    print("3. Tree reduction maximizes parallelism")
    print("4. Vectorization and unrolling improve performance")


if __name__ == "__main__":
    main()
