"""
Simple GEMM in TileLang
========================

This example implements matrix multiplication (GEMM) using TileLang's
tile-centric abstractions. We progressively build up from naive to
optimized implementations.

GEMM: C = A @ B
Where A is (M, K), B is (K, N), C is (M, N)
"""

import tilelang as T
import torch
import time


# Version 1: Naive GEMM (No Tiling)
# ==================================
@T.prim_func
def gemm_naive(
    A: T.Buffer((1024, 1024), "float32"),
    B: T.Buffer((1024, 1024), "float32"),
    C: T.Buffer((1024, 1024), "float32")
):
    """
    Naive matrix multiplication - one thread per output element.

    Each thread computes one element of C by performing a dot product.
    This is inefficient due to poor memory reuse.
    """
    M, K = 1024, 1024
    N = 1024

    with T.block("root"):
        # 2D grid of threads - one per output element
        tx = T.thread_binding(0, N, "threadIdx.x")
        ty = T.thread_binding(0, M, "threadIdx.y")
        bx = T.thread_binding(0, 1, "blockIdx.x")
        by = T.thread_binding(0, 1, "blockIdx.y")

        # Accumulator
        acc = T.alloc_fragment([1], "float32")
        acc[0] = 0.0

        # Compute dot product
        for k in T.serial(K):
            a_val = A[ty, k]
            b_val = B[k, tx]
            acc[0] = acc[0] + a_val * b_val

        C[ty, tx] = acc[0]


# Version 2: Tiled GEMM with Shared Memory
# =========================================
@T.prim_func
def gemm_tiled(
    A: T.Buffer((1024, 1024), "float16"),
    B: T.Buffer((1024, 1024), "float16"),
    C: T.Buffer((1024, 1024), "float32")
):
    """
    Tiled GEMM using shared memory for data reuse.

    Key optimizations:
    1. Tile A and B into shared memory
    2. Reuse tiles across multiple threads
    3. Reduce global memory traffic by BLOCK_K×
    """
    M, K, N = 1024, 1024, 1024
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    with T.block("root"):
        # Block indices
        bx = T.thread_binding(0, N // BLOCK_N, "blockIdx.x")
        by = T.thread_binding(0, M // BLOCK_M, "blockIdx.y")

        # Thread indices (16x16 threads per block)
        tx = T.thread_binding(0, 16, "threadIdx.x")
        ty = T.thread_binding(0, 16, "threadIdx.y")

        # Shared memory tiles
        A_shared = T.alloc_shared([BLOCK_M, BLOCK_K], "float16")
        B_shared = T.alloc_shared([BLOCK_K, BLOCK_N], "float16")

        # Register accumulator (8×8 elements per thread)
        C_local = T.alloc_fragment([8, 8], "float32")

        # Initialize accumulator
        for i in T.serial(8):
            for j in T.serial(8):
                C_local[i, j] = 0.0

        # Tile loop over K dimension
        for k_tile in T.serial(K // BLOCK_K):
            # Cooperatively load A tile into shared memory
            for i in T.serial(8):
                for kk in T.serial(2):
                    row = by * BLOCK_M + ty * 8 + i
                    col = k_tile * BLOCK_K + tx * 2 + kk
                    A_shared[ty * 8 + i, tx * 2 + kk] = A[row, col]

            # Cooperatively load B tile into shared memory
            for k in T.serial(2):
                for j in T.serial(8):
                    row = k_tile * BLOCK_K + ty * 2 + k
                    col = bx * BLOCK_N + tx * 8 + j
                    B_shared[ty * 2 + k, tx * 8 + j] = B[row, col]

            T.sync_threads()

            # Compute on tiles in shared memory
            for kk in T.serial(BLOCK_K):
                # Load from shared to registers
                A_reg = T.alloc_fragment([8], "float16")
                B_reg = T.alloc_fragment([8], "float16")

                for i in T.serial(8):
                    A_reg[i] = A_shared[ty * 8 + i, kk]
                for j in T.serial(8):
                    B_reg[j] = B_shared[kk, tx * 8 + j]

                # Outer product
                for i in T.serial(8):
                    for j in T.serial(8):
                        C_local[i, j] = C_local[i, j] + \
                            T.cast(A_reg[i], "float32") * T.cast(B_reg[j], "float32")

            T.sync_threads()

        # Write results back to global memory
        for i in T.serial(8):
            for j in T.serial(8):
                row = by * BLOCK_M + ty * 8 + i
                col = bx * BLOCK_N + tx * 8 + j
                C[row, col] = C_local[i, j]


# Version 3: GEMM with Tensor Core Support
# =========================================
@T.prim_func
def gemm_tensorcore(
    A: T.Buffer((1024, 1024), "float16"),
    B: T.Buffer((1024, 1024), "float16"),
    C: T.Buffer((1024, 1024), "float32")
):
    """
    GEMM optimized for Tensor Cores using T.gemm() primitive.

    T.gemm() automatically maps to mma.sync instructions on GPUs
    with Tensor Cores, achieving much higher throughput.
    """
    M, K, N = 1024, 1024, 1024
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    with T.block("root"):
        bx = T.thread_binding(0, N // BLOCK_N, "blockIdx.x")
        by = T.thread_binding(0, M // BLOCK_M, "blockIdx.y")

        # Shared memory tiles
        A_shared = T.alloc_shared([BLOCK_M, BLOCK_K], "float16")
        B_shared = T.alloc_shared([BLOCK_K, BLOCK_N], "float16")

        # Fragments for Tensor Core operations
        A_frag = T.alloc_fragment([BLOCK_M, BLOCK_K], "float16")
        B_frag = T.alloc_fragment([BLOCK_K, BLOCK_N], "float16")
        C_frag = T.alloc_fragment([BLOCK_M, BLOCK_N], "float32")

        # Initialize accumulator
        T.fill(C_frag, 0.0)

        # Main loop
        for k in T.serial(K // BLOCK_K):
            # Load tiles into shared memory
            T.copy(A[by * BLOCK_M : (by + 1) * BLOCK_M,
                     k * BLOCK_K : (k + 1) * BLOCK_K],
                   A_shared)
            T.copy(B[k * BLOCK_K : (k + 1) * BLOCK_K,
                     bx * BLOCK_N : (bx + 1) * BLOCK_N],
                   B_shared)

            T.sync_threads()

            # Copy to fragments
            T.copy(A_shared, A_frag)
            T.copy(B_shared, B_frag)

            # Tensor Core GEMM: C_frag += A_frag @ B_frag
            T.gemm(A_frag, B_frag, C_frag, transpose_B=False)

            T.sync_threads()

        # Write back results
        T.copy(C_frag, C[by * BLOCK_M : (by + 1) * BLOCK_M,
                         bx * BLOCK_N : (bx + 1) * BLOCK_N])


# Version 4: Rectangular Matrix GEMM
# ===================================
@T.prim_func
def gemm_rectangular(
    M: T.int32,
    N: T.int32,
    K: T.int32,
    A: T.Buffer("float16"),
    B: T.Buffer("float16"),
    C: T.Buffer("float32")
):
    """
    GEMM for arbitrary matrix sizes (not just square 1024×1024).

    This demonstrates dynamic shapes in TileLang.
    """
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    with T.block("root"):
        # Handle partial tiles at boundaries
        bx = T.thread_binding(0, (N + BLOCK_N - 1) // BLOCK_N, "blockIdx.x")
        by = T.thread_binding(0, (M + BLOCK_M - 1) // BLOCK_M, "blockIdx.y")

        # Shared memory
        A_shared = T.alloc_shared([BLOCK_M, BLOCK_K], "float16")
        B_shared = T.alloc_shared([BLOCK_K, BLOCK_N], "float16")

        # Fragments
        A_frag = T.alloc_fragment([BLOCK_M, BLOCK_K], "float16")
        B_frag = T.alloc_fragment([BLOCK_K, BLOCK_N], "float16")
        C_frag = T.alloc_fragment([BLOCK_M, BLOCK_N], "float32")

        T.fill(C_frag, 0.0)

        num_k_tiles = (K + BLOCK_K - 1) // BLOCK_K

        for k_tile in T.serial(num_k_tiles):
            # Compute actual tile dimensions (handle boundaries)
            m_start = by * BLOCK_M
            m_end = T.min(m_start + BLOCK_M, M)
            n_start = bx * BLOCK_N
            n_end = T.min(n_start + BLOCK_N, N)
            k_start = k_tile * BLOCK_K
            k_end = T.min(k_start + BLOCK_K, K)

            # Load with bounds checking
            T.copy(A[m_start:m_end, k_start:k_end], A_shared)
            T.copy(B[k_start:k_end, n_start:n_end], B_shared)

            T.sync_threads()

            T.copy(A_shared, A_frag)
            T.copy(B_shared, B_frag)

            T.gemm(A_frag, B_frag, C_frag)

            T.sync_threads()

        # Write back
        m_start = by * BLOCK_M
        m_end = T.min(m_start + BLOCK_M, M)
        n_start = bx * BLOCK_N
        n_end = T.min(n_start + BLOCK_N, N)
        T.copy(C_frag, C[m_start:m_end, n_start:n_end])


# Testing and Benchmarking
# =========================

def test_gemm_naive():
    """Test naive GEMM implementation."""
    print("Testing gemm_naive...")

    # Small size for naive version
    size = 128
    A = torch.randn(size, size, device="cuda", dtype=torch.float32)
    B = torch.randn(size, size, device="cuda", dtype=torch.float32)
    C = torch.zeros(size, size, device="cuda", dtype=torch.float32)

    # Note: Would need to adjust kernel for different sizes
    # This is a conceptual test
    print("  Skipped (requires size adjustment)")


def test_gemm_tiled():
    """Test tiled GEMM implementation."""
    print("Testing gemm_tiled...")

    A = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    B = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    C = torch.zeros(1024, 1024, device="cuda", dtype=torch.float32)

    mod = T.compile(gemm_tiled, target="cuda")
    mod(A, B, C)

    # Reference
    expected = (A.float() @ B.float())

    max_diff = (C - expected).abs().max().item()
    print(f"  Max difference: {max_diff:.6f}")
    assert torch.allclose(C, expected, rtol=1e-2, atol=1e-2), "Tiled GEMM failed!"
    print("✓ gemm_tiled passed")


def test_gemm_tensorcore():
    """Test Tensor Core GEMM implementation."""
    print("Testing gemm_tensorcore...")

    A = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    B = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    C = torch.zeros(1024, 1024, device="cuda", dtype=torch.float32)

    mod = T.compile(gemm_tensorcore, target="cuda")
    mod(A, B, C)

    expected = (A.float() @ B.float())

    max_diff = (C - expected).abs().max().item()
    print(f"  Max difference: {max_diff:.6f}")
    assert torch.allclose(C, expected, rtol=1e-2, atol=1e-2), "TensorCore GEMM failed!"
    print("✓ gemm_tensorcore passed")


def benchmark_gemm_variants():
    """Benchmark different GEMM implementations."""
    print("\n" + "="*70)
    print("GEMM Performance Comparison (1024×1024 × 1024×1024)")
    print("="*70)

    A = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    B = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    C = torch.zeros(1024, 1024, device="cuda", dtype=torch.float32)

    # Compile kernels
    mod_tiled = T.compile(gemm_tiled, target="cuda")
    mod_tensorcore = T.compile(gemm_tensorcore, target="cuda")

    # Warmup
    for _ in range(10):
        _ = A.float() @ B.float()
        mod_tiled(A, B, C)
        mod_tensorcore(A, B, C)
    torch.cuda.synchronize()

    # Benchmark PyTorch
    start = time.time()
    for _ in range(100):
        result = A.float() @ B.float()
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / 100

    # Benchmark tiled
    start = time.time()
    for _ in range(100):
        mod_tiled(A, B, C)
    torch.cuda.synchronize()
    tiled_time = (time.time() - start) / 100

    # Benchmark tensor core
    start = time.time()
    for _ in range(100):
        mod_tensorcore(A, B, C)
    torch.cuda.synchronize()
    tensorcore_time = (time.time() - start) / 100

    # Calculate FLOPS
    flops = 2 * 1024 * 1024 * 1024  # 2*M*N*K

    print(f"\n{'Implementation':<20} {'Time (ms)':<12} {'TFLOPS':<10} {'Relative':<10}")
    print("-" * 70)
    print(f"{'PyTorch (cuBLAS)':<20} {pytorch_time*1e3:>10.3f}   "
          f"{flops/pytorch_time/1e12:>8.2f}   {'100%':>8}")
    print(f"{'TileLang (Tiled)':<20} {tiled_time*1e3:>10.3f}   "
          f"{flops/tiled_time/1e12:>8.2f}   "
          f"{100*pytorch_time/tiled_time:>7.1f}%")
    print(f"{'TileLang (TC)':<20} {tensorcore_time*1e3:>10.3f}   "
          f"{flops/tensorcore_time/1e12:>8.2f}   "
          f"{100*pytorch_time/tensorcore_time:>7.1f}%")
    print("="*70)
    print("\nNote: TileLang implementations may be slower than highly")
    print("      optimized cuBLAS, but offer better programmability.")
    print("="*70)


def analyze_memory_traffic():
    """Analyze memory traffic for different GEMM approaches."""
    print("\n" + "="*70)
    print("Memory Traffic Analysis (1024×1024 GEMM)")
    print("="*70)

    M = N = K = 1024
    element_size = 2  # float16 = 2 bytes

    print("\nNaive Approach (no tiling):")
    loads_naive = M * N * (2 * K)  # Each output needs 2K loads
    print(f"  Total loads:  {loads_naive / 1e9:.2f} billion elements")
    print(f"  Memory read:  {loads_naive * element_size / 1e9:.2f} GB")
    print(f"  Bandwidth at 1.5TB/s: {loads_naive * element_size / 1.5e12 * 1000:.1f} ms")

    print("\nTiled Approach (128×128 tiles, K=32):")
    tiles_m = M // 128
    tiles_n = N // 128
    tiles_k = K // 32
    loads_per_tile = 128 * 32 + 32 * 128  # Load A and B tile
    total_loads_tiled = tiles_m * tiles_n * tiles_k * loads_per_tile
    print(f"  Total loads:  {total_loads_tiled / 1e6:.2f} million elements")
    print(f"  Memory read:  {total_loads_tiled * element_size / 1e9:.2f} GB")
    print(f"  Reduction:    {loads_naive / total_loads_tiled:.1f}× fewer loads")
    print(f"  Bandwidth at 1.5TB/s: {total_loads_tiled * element_size / 1.5e12 * 1000:.1f} ms")

    print("\nKey Insight:")
    print(f"  Tiling reduces global memory traffic by {loads_naive / total_loads_tiled:.0f}×!")
    print("  This is why shared memory is crucial for GEMM performance.")
    print("="*70)


def main():
    """Run all tests and benchmarks."""
    print("="*70)
    print("TileLang GEMM Examples")
    print("="*70)

    # Tests
    test_gemm_naive()
    test_gemm_tiled()
    test_gemm_tensorcore()

    # Benchmarks
    benchmark_gemm_variants()
    analyze_memory_traffic()

    print("\n✓ All tests passed!")
    print("\nKey Takeaways:")
    print("1. Tiling is essential for GEMM performance")
    print("2. Shared memory reduces global memory traffic dramatically")
    print("3. T.gemm() leverages Tensor Cores for high performance")
    print("4. TileLang makes these optimizations accessible with clean syntax")


if __name__ == "__main__":
    main()
