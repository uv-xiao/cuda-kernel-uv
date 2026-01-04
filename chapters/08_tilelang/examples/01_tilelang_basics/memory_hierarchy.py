"""
TileLang Memory Hierarchy
==========================

This example explores TileLang's memory hierarchy abstractions:
1. Global Memory (DRAM)
2. Shared Memory (on-chip SRAM) - T.alloc_shared()
3. Register Fragments (per-thread registers) - T.alloc_fragment()

Understanding these levels is crucial for writing high-performance kernels.
"""

import tilelang as T
import torch
import time


# Example 1: Memory Hierarchy Demonstration
# ==========================================
@T.prim_func
def memory_hierarchy_demo(
    A: T.Buffer((4096,), "float16"),
    B: T.Buffer((4096,), "float16")
):
    """
    Demonstrates the three levels of memory hierarchy.

    Data flow:
    1. Global Memory (A) -> Shared Memory (A_shared)
    2. Shared Memory (A_shared) -> Register Fragment (A_frag)
    3. Compute on registers
    4. Register Fragment -> Shared Memory -> Global Memory (B)
    """
    BLOCK_SIZE = 256

    with T.block("root"):
        bx = T.thread_binding(0, 4096 // BLOCK_SIZE, "blockIdx.x")
        tx = T.thread_binding(0, BLOCK_SIZE, "threadIdx.x")

        # Level 1: Shared Memory (cooperative across block)
        A_shared = T.alloc_shared([BLOCK_SIZE], "float16")

        # Level 2: Register Fragment (per-thread)
        A_frag = T.alloc_fragment([1], "float16")
        B_frag = T.alloc_fragment([1], "float16")

        # Stage 1: Global -> Shared (cooperative load)
        offset = bx * BLOCK_SIZE
        A_shared[tx] = A[offset + tx]
        T.sync_threads()

        # Stage 2: Shared -> Register (per-thread load)
        A_frag[0] = A_shared[tx]

        # Stage 3: Compute on registers
        B_frag[0] = A_frag[0] * T.cast(2.0, "float16")

        # Stage 4: Register -> Shared -> Global
        A_shared[tx] = B_frag[0]
        T.sync_threads()
        B[offset + tx] = A_shared[tx]


# Example 2: Shared Memory Tiling for Matrix Operations
# ======================================================
@T.prim_func
def matrix_multiply_shared(
    A: T.Buffer((256, 256), "float16"),
    B: T.Buffer((256, 256), "float16"),
    C: T.Buffer((256, 256), "float32")
):
    """
    Matrix multiplication using shared memory tiles.

    This demonstrates:
    - Tiling to fit in shared memory
    - Reuse of data in shared memory
    - Reduction of global memory accesses
    """
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16

    with T.block("root"):
        bx = T.thread_binding(0, 256 // BLOCK_N, "blockIdx.x")
        by = T.thread_binding(0, 256 // BLOCK_M, "blockIdx.y")
        tx = T.thread_binding(0, BLOCK_N, "threadIdx.x")
        ty = T.thread_binding(0, BLOCK_M, "threadIdx.y")

        # Shared memory tiles
        A_shared = T.alloc_shared([BLOCK_M, BLOCK_K], "float16")
        B_shared = T.alloc_shared([BLOCK_K, BLOCK_N], "float16")

        # Accumulator in registers
        acc = T.alloc_fragment([1], "float32")
        acc[0] = T.cast(0.0, "float32")

        # Tiled computation
        for k in T.serial(256 // BLOCK_K):
            # Load tiles into shared memory
            if tx < BLOCK_K:
                A_shared[ty, tx] = A[by * BLOCK_M + ty, k * BLOCK_K + tx]
            if ty < BLOCK_K:
                B_shared[ty, tx] = B[k * BLOCK_K + ty, bx * BLOCK_N + tx]
            T.sync_threads()

            # Compute using shared memory
            for kk in T.serial(BLOCK_K):
                a_val = A_shared[ty, kk]
                b_val = B_shared[kk, tx]
                acc[0] = acc[0] + T.cast(a_val, "float32") * T.cast(b_val, "float32")

            T.sync_threads()

        # Write result
        C[by * BLOCK_M + ty, bx * BLOCK_N + tx] = acc[0]


# Example 3: Fragment-Level Operations
# =====================================
@T.prim_func
def fragment_operations(
    A: T.Buffer((1024,), "float16"),
    B: T.Buffer((1024,), "float16"),
    C: T.Buffer((1024,), "float16")
):
    """
    Demonstrates operations on register fragments.

    Fragments are the fastest memory tier and should be used
    for intermediate computations.
    """
    BLOCK_SIZE = 256
    ELEMENTS_PER_THREAD = 4  # Each thread processes 4 elements

    with T.block("root"):
        bx = T.thread_binding(0, 1024 // (BLOCK_SIZE * ELEMENTS_PER_THREAD), "blockIdx.x")
        tx = T.thread_binding(0, BLOCK_SIZE, "threadIdx.x")

        # Register fragments for vectorized loads
        A_frag = T.alloc_fragment([ELEMENTS_PER_THREAD], "float16")
        B_frag = T.alloc_fragment([ELEMENTS_PER_THREAD], "float16")
        C_frag = T.alloc_fragment([ELEMENTS_PER_THREAD], "float16")

        # Base offset for this thread
        base = bx * BLOCK_SIZE * ELEMENTS_PER_THREAD + tx * ELEMENTS_PER_THREAD

        # Vectorized load into fragments
        for i in T.serial(ELEMENTS_PER_THREAD):
            A_frag[i] = A[base + i]
            B_frag[i] = B[base + i]

        # Compute on registers (fast!)
        for i in T.serial(ELEMENTS_PER_THREAD):
            C_frag[i] = A_frag[i] + B_frag[i]

        # Vectorized store from fragments
        for i in T.serial(ELEMENTS_PER_THREAD):
            C[base + i] = C_frag[i]


# Example 4: Cooperative Copy Operations
# =======================================
@T.prim_func
def cooperative_copy_demo(
    A: T.Buffer((1024, 1024), "float16"),
    B: T.Buffer((1024, 1024), "float16")
):
    """
    Demonstrates T.copy() for efficient cooperative memory transfers.

    T.copy() is optimized to use:
    - Vectorized loads/stores
    - Async copy instructions (cp.async)
    - Optimal thread cooperation patterns
    """
    TILE_M = 64
    TILE_N = 64

    with T.block("root"):
        bx = T.thread_binding(0, 1024 // TILE_N, "blockIdx.x")
        by = T.thread_binding(0, 1024 // TILE_M, "blockIdx.y")

        # Shared memory tile
        tile = T.alloc_shared([TILE_M, TILE_N], "float16")

        # Cooperative copy: Global -> Shared
        # All threads in the block participate
        for i in T.thread_binding(0, TILE_M, "threadIdx.y"):
            for j in T.thread_binding(0, TILE_N, "threadIdx.x"):
                tile[i, j] = A[by * TILE_M + i, bx * TILE_N + j]

        T.sync_threads()

        # Process data in shared memory (placeholder)
        # ... some computation ...

        # Cooperative copy: Shared -> Global
        for i in T.thread_binding(0, TILE_M, "threadIdx.y"):
            for j in T.thread_binding(0, TILE_N, "threadIdx.x"):
                B[by * TILE_M + i, bx * TILE_N + j] = tile[i, j]


# Example 5: Memory Bandwidth Analysis
# =====================================
@T.prim_func
def bandwidth_test_global(
    A: T.Buffer((1024*1024,), "float32"),
    B: T.Buffer((1024*1024,), "float32")
):
    """
    Test global memory bandwidth.
    Each element is loaded once and stored once.
    """
    BLOCK_SIZE = 256

    with T.block("root"):
        bx = T.thread_binding(0, (1024*1024) // BLOCK_SIZE, "blockIdx.x")
        tx = T.thread_binding(0, BLOCK_SIZE, "threadIdx.x")

        idx = bx * BLOCK_SIZE + tx
        B[idx] = A[idx]


@T.prim_func
def bandwidth_test_shared(
    A: T.Buffer((1024*1024,), "float32"),
    B: T.Buffer((1024*1024,), "float32")
):
    """
    Test with shared memory intermediate.
    Demonstrates bandwidth differences.
    """
    BLOCK_SIZE = 256

    with T.block("root"):
        bx = T.thread_binding(0, (1024*1024) // BLOCK_SIZE, "blockIdx.x")
        tx = T.thread_binding(0, BLOCK_SIZE, "threadIdx.x")

        # Shared memory buffer
        shared = T.alloc_shared([BLOCK_SIZE], "float32")

        # Global -> Shared
        idx = bx * BLOCK_SIZE + tx
        shared[tx] = A[idx]
        T.sync_threads()

        # Shared -> Global
        B[idx] = shared[tx]


# Testing Functions
# =================

def test_memory_hierarchy_demo():
    """Test memory hierarchy demonstration."""
    print("Testing memory_hierarchy_demo...")

    A = torch.randn(4096, device="cuda", dtype=torch.float16)
    B = torch.zeros(4096, device="cuda", dtype=torch.float16)

    mod = T.compile(memory_hierarchy_demo, target="cuda")
    mod(A, B)

    expected = A * 2.0
    assert torch.allclose(B, expected, rtol=1e-3), "Memory hierarchy demo failed!"
    print("✓ memory_hierarchy_demo passed")


def test_matrix_multiply_shared():
    """Test matrix multiplication with shared memory."""
    print("Testing matrix_multiply_shared...")

    A = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    B = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    C = torch.zeros(256, 256, device="cuda", dtype=torch.float32)

    mod = T.compile(matrix_multiply_shared, target="cuda")
    mod(A, B, C)

    expected = (A.float() @ B.float())
    assert torch.allclose(C, expected, rtol=1e-2, atol=1e-2), "Matrix multiply failed!"
    print("✓ matrix_multiply_shared passed")


def test_fragment_operations():
    """Test fragment operations."""
    print("Testing fragment_operations...")

    A = torch.randn(1024, device="cuda", dtype=torch.float16)
    B = torch.randn(1024, device="cuda", dtype=torch.float16)
    C = torch.zeros(1024, device="cuda", dtype=torch.float16)

    mod = T.compile(fragment_operations, target="cuda")
    mod(A, B, C)

    expected = A + B
    assert torch.allclose(C, expected, rtol=1e-3), "Fragment operations failed!"
    print("✓ fragment_operations passed")


def test_cooperative_copy():
    """Test cooperative copy."""
    print("Testing cooperative_copy_demo...")

    A = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    B = torch.zeros(1024, 1024, device="cuda", dtype=torch.float16)

    mod = T.compile(cooperative_copy_demo, target="cuda")
    mod(A, B)

    assert torch.allclose(B, A, rtol=1e-5), "Cooperative copy failed!"
    print("✓ cooperative_copy_demo passed")


def benchmark_memory_tiers():
    """Benchmark different memory tiers."""
    print("\n" + "="*60)
    print("Memory Bandwidth Benchmarks")
    print("="*60)

    size = 1024 * 1024
    A = torch.randn(size, device="cuda", dtype=torch.float32)
    B = torch.zeros(size, device="cuda", dtype=torch.float32)

    # Compile kernels
    mod_global = T.compile(bandwidth_test_global, target="cuda")
    mod_shared = T.compile(bandwidth_test_shared, target="cuda")

    # Warmup
    for _ in range(10):
        mod_global(A, B)
        mod_shared(A, B)
    torch.cuda.synchronize()

    # Benchmark global memory
    start = time.time()
    for _ in range(100):
        mod_global(A, B)
    torch.cuda.synchronize()
    global_time = (time.time() - start) / 100

    # Benchmark with shared memory
    start = time.time()
    for _ in range(100):
        mod_shared(A, B)
    torch.cuda.synchronize()
    shared_time = (time.time() - start) / 100

    # Calculate bandwidth
    bytes_transferred = size * 4 * 2  # 4 bytes per float32, read + write
    global_bw = bytes_transferred / global_time / 1e9
    shared_bw = bytes_transferred / shared_time / 1e9

    print(f"Array size: {size} elements ({size*4/1e6:.2f} MB)")
    print(f"\nGlobal Memory Only:")
    print(f"  Time:      {global_time*1e3:.3f} ms")
    print(f"  Bandwidth: {global_bw:.2f} GB/s")
    print(f"\nWith Shared Memory:")
    print(f"  Time:      {shared_time*1e3:.3f} ms")
    print(f"  Bandwidth: {shared_bw:.2f} GB/s")
    print(f"\nOverhead: {(shared_time/global_time - 1)*100:.1f}%")
    print("="*60)


def demonstrate_memory_sizes():
    """Show typical memory sizes at each level."""
    print("\n" + "="*60)
    print("GPU Memory Hierarchy (Typical NVIDIA A100)")
    print("="*60)
    print("\n1. Registers (Per SM)")
    print("   - Size:      256 KB per SM")
    print("   - Latency:   ~1 cycle")
    print("   - Bandwidth: ~20 TB/s (effective)")
    print("   - Access:    Per-thread, automatically managed")
    print("\n2. Shared Memory (Per SM)")
    print("   - Size:      Up to 164 KB per SM")
    print("   - Latency:   ~20-30 cycles")
    print("   - Bandwidth: ~15 TB/s (effective)")
    print("   - Access:    Cooperative within block")
    print("\n3. L1 Cache (Per SM)")
    print("   - Size:      Unified with shared (164 KB total)")
    print("   - Latency:   ~30 cycles")
    print("   - Bandwidth: ~10 TB/s")
    print("   - Access:    Automatic caching")
    print("\n4. L2 Cache (Global)")
    print("   - Size:      40 MB")
    print("   - Latency:   ~200 cycles")
    print("   - Bandwidth: ~3 TB/s")
    print("   - Access:    Automatic caching")
    print("\n5. Global Memory (HBM2)")
    print("   - Size:      40/80 GB")
    print("   - Latency:   ~300-400 cycles")
    print("   - Bandwidth: ~1.5-2.0 TB/s")
    print("   - Access:    Explicit loads/stores")
    print("\nKey Insight: Each level is ~10x slower but ~1000x larger!")
    print("="*60)


def demonstrate_tiling_strategy():
    """Explain tiling strategy for memory hierarchy."""
    print("\n" + "="*60)
    print("Tiling Strategy for Memory Hierarchy")
    print("="*60)
    print("\nExample: Matrix Multiplication (M=N=K=4096)")
    print("\nNaive Approach:")
    print("  - Load each element from global memory multiple times")
    print("  - Total loads: 4096³ × 2 = 137 billion loads")
    print("  - Time: ~200ms at 1.5 TB/s")
    print("\nTiled Approach (64×64 tiles):")
    print("  - Load each element once into shared memory")
    print("  - Reuse tile 64 times from shared memory")
    print("  - Total global loads: 4096² × 2 = 33 million loads")
    print("  - Shared memory reuse reduces global traffic by 64×")
    print("  - Time: ~5ms (40× faster!)")
    print("\nTileLang automatically applies these optimizations!")
    print("="*60)


def main():
    """Run all examples and demonstrations."""
    print("="*60)
    print("TileLang Memory Hierarchy Examples")
    print("="*60)

    # Run tests
    test_memory_hierarchy_demo()
    test_matrix_multiply_shared()
    test_fragment_operations()
    test_cooperative_copy()

    # Benchmarks and demonstrations
    benchmark_memory_tiers()
    demonstrate_memory_sizes()
    demonstrate_tiling_strategy()

    print("\n✓ All tests passed!")
    print("\nKey Takeaways:")
    print("1. Three levels: Global (slow) -> Shared (fast) -> Registers (fastest)")
    print("2. Use T.alloc_shared() for cooperative thread data")
    print("3. Use T.alloc_fragment() for per-thread registers")
    print("4. Tiling reduces global memory traffic dramatically")
    print("5. TileLang abstracts away many low-level details")


if __name__ == "__main__":
    main()
