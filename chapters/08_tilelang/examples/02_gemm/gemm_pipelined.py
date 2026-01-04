"""
Pipelined GEMM in TileLang
===========================

This example demonstrates software pipelining to hide memory latency
by overlapping computation with data transfers.

Software pipelining is one of the most important optimizations for
achieving peak performance on modern GPUs.
"""

import tilelang as T
import torch
import time


# Version 1: GEMM Without Pipelining (Baseline)
# ==============================================
@T.prim_func
def gemm_no_pipeline(
    A: T.Buffer((2048, 2048), "float16"),
    B: T.Buffer((2048, 2048), "float16"),
    C: T.Buffer((2048, 2048), "float32")
):
    """
    GEMM without pipelining - baseline for comparison.

    Execution pattern:
    1. Load tile k
    2. Wait for load to complete
    3. Compute on tile k
    4. Repeat

    Memory and compute are sequential - no overlap!
    """
    M, K, N = 2048, 2048, 2048
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    with T.block("root"):
        bx = T.thread_binding(0, N // BLOCK_N, "blockIdx.x")
        by = T.thread_binding(0, M // BLOCK_M, "blockIdx.y")

        # Shared memory
        A_shared = T.alloc_shared([BLOCK_M, BLOCK_K], "float16")
        B_shared = T.alloc_shared([BLOCK_K, BLOCK_N], "float16")

        # Fragments
        A_frag = T.alloc_fragment([BLOCK_M, BLOCK_K], "float16")
        B_frag = T.alloc_fragment([BLOCK_K, BLOCK_N], "float16")
        C_frag = T.alloc_fragment([BLOCK_M, BLOCK_N], "float32")

        T.fill(C_frag, 0.0)

        # Main loop - NO PIPELINING
        for k in T.serial(K // BLOCK_K):
            # Stage 1: Load data (blocks compute)
            T.copy(A[by * BLOCK_M : (by + 1) * BLOCK_M,
                     k * BLOCK_K : (k + 1) * BLOCK_K],
                   A_shared)
            T.copy(B[k * BLOCK_K : (k + 1) * BLOCK_K,
                     bx * BLOCK_N : (bx + 1) * BLOCK_N],
                   B_shared)
            T.sync_threads()

            # Stage 2: Compute (blocks memory)
            T.copy(A_shared, A_frag)
            T.copy(B_shared, B_frag)
            T.gemm(A_frag, B_frag, C_frag)
            T.sync_threads()

        # Write back
        T.copy(C_frag, C[by * BLOCK_M : (by + 1) * BLOCK_M,
                         bx * BLOCK_N : (bx + 1) * BLOCK_N])


# Version 2: GEMM With Double Buffering
# ======================================
@T.prim_func
def gemm_double_buffer(
    A: T.Buffer((2048, 2048), "float16"),
    B: T.Buffer((2048, 2048), "float16"),
    C: T.Buffer((2048, 2048), "float32")
):
    """
    GEMM with double buffering for software pipelining.

    Execution pattern:
    Iteration k:
    1. Load tile k+1 (async) while computing tile k
    2. Swap buffers
    3. Repeat

    This overlaps memory transfers with computation!
    """
    M, K, N = 2048, 2048, 2048
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    with T.block("root"):
        bx = T.thread_binding(0, N // BLOCK_N, "blockIdx.x")
        by = T.thread_binding(0, M // BLOCK_M, "blockIdx.y")

        # Double buffered shared memory
        A_shared = T.alloc_shared([2, BLOCK_M, BLOCK_K], "float16")
        B_shared = T.alloc_shared([2, BLOCK_K, BLOCK_N], "float16")

        # Fragments
        A_frag = T.alloc_fragment([BLOCK_M, BLOCK_K], "float16")
        B_frag = T.alloc_fragment([BLOCK_K, BLOCK_N], "float16")
        C_frag = T.alloc_fragment([BLOCK_M, BLOCK_N], "float32")

        T.fill(C_frag, 0.0)

        # Prologue: Load first tile
        T.copy(A[by * BLOCK_M : (by + 1) * BLOCK_M, 0 : BLOCK_K],
               A_shared[0])
        T.copy(B[0 : BLOCK_K, bx * BLOCK_N : (bx + 1) * BLOCK_N],
               B_shared[0])
        T.sync_threads()

        # Main loop with pipelining
        for k in T.serial(K // BLOCK_K):
            # Current and next buffer indices
            curr = k % 2
            next_buf = (k + 1) % 2

            # Prefetch next tile (async, if k+1 is valid)
            if k + 1 < K // BLOCK_K:
                T.copy(A[by * BLOCK_M : (by + 1) * BLOCK_M,
                         (k + 1) * BLOCK_K : (k + 2) * BLOCK_K],
                       A_shared[next_buf])
                T.copy(B[(k + 1) * BLOCK_K : (k + 2) * BLOCK_K,
                         bx * BLOCK_N : (bx + 1) * BLOCK_N],
                       B_shared[next_buf])

            # Compute on current tile
            T.copy(A_shared[curr], A_frag)
            T.copy(B_shared[curr], B_frag)
            T.gemm(A_frag, B_frag, C_frag)

            T.sync_threads()

        # Write back
        T.copy(C_frag, C[by * BLOCK_M : (by + 1) * BLOCK_M,
                         bx * BLOCK_N : (bx + 1) * BLOCK_N])


# Version 3: GEMM With Automatic Pipelining
# ==========================================
@T.prim_func
def gemm_auto_pipeline(
    A: T.Buffer((2048, 2048), "float16"),
    B: T.Buffer((2048, 2048), "float16"),
    C: T.Buffer((2048, 2048), "float32")
):
    """
    GEMM with TileLang's automatic pipelining using T.pipeline().

    TileLang automatically:
    1. Allocates double/multi-buffers
    2. Schedules async copies
    3. Manages synchronization
    4. Overlaps stages optimally
    """
    M, K, N = 2048, 2048, 2048
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    with T.block("root"):
        bx = T.thread_binding(0, N // BLOCK_N, "blockIdx.x")
        by = T.thread_binding(0, M // BLOCK_M, "blockIdx.y")

        # Shared memory (TileLang manages buffering)
        A_shared = T.alloc_shared([BLOCK_M, BLOCK_K], "float16")
        B_shared = T.alloc_shared([BLOCK_K, BLOCK_N], "float16")

        # Fragments
        A_frag = T.alloc_fragment([BLOCK_M, BLOCK_K], "float16")
        B_frag = T.alloc_fragment([BLOCK_K, BLOCK_N], "float16")
        C_frag = T.alloc_fragment([BLOCK_M, BLOCK_N], "float32")

        T.fill(C_frag, 0.0)

        # Pipeline context automatically manages staging
        with T.pipeline(num_stages=2):
            for k in T.serial(K // BLOCK_K):
                # These operations are automatically pipelined
                # Load (async)
                T.copy(A[by * BLOCK_M : (by + 1) * BLOCK_M,
                         k * BLOCK_K : (k + 1) * BLOCK_K],
                       A_shared)
                T.copy(B[k * BLOCK_K : (k + 1) * BLOCK_K,
                         bx * BLOCK_N : (bx + 1) * BLOCK_N],
                       B_shared)

                # Compute (overlapped with next iteration's load)
                T.copy(A_shared, A_frag)
                T.copy(B_shared, B_frag)
                T.gemm(A_frag, B_frag, C_frag)

        # Write back
        T.copy(C_frag, C[by * BLOCK_M : (by + 1) * BLOCK_M,
                         bx * BLOCK_N : (bx + 1) * BLOCK_N])


# Version 4: Multi-Stage Pipeline
# ================================
@T.prim_func
def gemm_multistage_pipeline(
    A: T.Buffer((2048, 2048), "float16"),
    B: T.Buffer((2048, 2048), "float16"),
    C: T.Buffer((2048, 2048), "float32")
):
    """
    GEMM with multi-stage pipelining (3+ stages).

    More stages can hide more latency but require more shared memory.

    Stages:
    - Stage 0: Load tile k
    - Stage 1: Load tile k+1, compute tile k
    - Stage 2: Load tile k+2, compute tile k+1
    """
    M, K, N = 2048, 2048, 2048
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    with T.block("root"):
        bx = T.thread_binding(0, N // BLOCK_N, "blockIdx.x")
        by = T.thread_binding(0, M // BLOCK_M, "blockIdx.y")

        # Shared memory
        A_shared = T.alloc_shared([BLOCK_M, BLOCK_K], "float16")
        B_shared = T.alloc_shared([BLOCK_K, BLOCK_N], "float16")

        # Fragments
        A_frag = T.alloc_fragment([BLOCK_M, BLOCK_K], "float16")
        B_frag = T.alloc_fragment([BLOCK_K, BLOCK_N], "float16")
        C_frag = T.alloc_fragment([BLOCK_M, BLOCK_N], "float32")

        T.fill(C_frag, 0.0)

        # Multi-stage pipeline (3 stages)
        with T.pipeline(num_stages=3):
            for k in T.serial(K // BLOCK_K):
                T.copy(A[by * BLOCK_M : (by + 1) * BLOCK_M,
                         k * BLOCK_K : (k + 1) * BLOCK_K],
                       A_shared)
                T.copy(B[k * BLOCK_K : (k + 1) * BLOCK_K,
                         bx * BLOCK_N : (bx + 1) * BLOCK_N],
                       B_shared)

                T.copy(A_shared, A_frag)
                T.copy(B_shared, B_frag)
                T.gemm(A_frag, B_frag, C_frag)

        # Write back
        T.copy(C_frag, C[by * BLOCK_M : (by + 1) * BLOCK_M,
                         bx * BLOCK_N : (bx + 1) * BLOCK_N])


# Testing and Benchmarking
# =========================

def test_all_variants():
    """Test correctness of all GEMM variants."""
    print("Testing all GEMM variants...")

    size = 2048
    A = torch.randn(size, size, device="cuda", dtype=torch.float16)
    B = torch.randn(size, size, device="cuda", dtype=torch.float16)

    # Reference result
    expected = (A.float() @ B.float())

    # Test each variant
    variants = [
        ("No Pipeline", gemm_no_pipeline),
        ("Double Buffer", gemm_double_buffer),
        ("Auto Pipeline", gemm_auto_pipeline),
        ("Multi-Stage", gemm_multistage_pipeline),
    ]

    for name, kernel in variants:
        print(f"  Testing {name}...")
        C = torch.zeros(size, size, device="cuda", dtype=torch.float32)

        mod = T.compile(kernel, target="cuda")
        mod(A, B, C)

        max_diff = (C - expected).abs().max().item()
        rel_error = max_diff / expected.abs().max().item()

        assert rel_error < 1e-2, f"{name} failed! Relative error: {rel_error}"
        print(f"    ✓ {name} passed (max error: {max_diff:.6f})")


def benchmark_pipeline_variants():
    """Benchmark different pipelining strategies."""
    print("\n" + "="*70)
    print("Software Pipelining Performance Comparison (2048×2048)")
    print("="*70)

    size = 2048
    A = torch.randn(size, size, device="cuda", dtype=torch.float16)
    B = torch.randn(size, size, device="cuda", dtype=torch.float16)
    C = torch.zeros(size, size, device="cuda", dtype=torch.float32)

    # Compile all variants
    mod_no_pipe = T.compile(gemm_no_pipeline, target="cuda")
    mod_double = T.compile(gemm_double_buffer, target="cuda")
    mod_auto = T.compile(gemm_auto_pipeline, target="cuda")
    mod_multi = T.compile(gemm_multistage_pipeline, target="cuda")

    # Warmup
    for mod in [mod_no_pipe, mod_double, mod_auto, mod_multi]:
        for _ in range(10):
            mod(A, B, C)
    torch.cuda.synchronize()

    # Benchmark function
    def bench(mod, name):
        start = time.time()
        for _ in range(50):
            mod(A, B, C)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / 50
        return elapsed

    # Run benchmarks
    times = {}
    times["No Pipeline"] = bench(mod_no_pipe, "No Pipeline")
    times["Double Buffer"] = bench(mod_double, "Double Buffer")
    times["Auto Pipeline"] = bench(mod_auto, "Auto Pipeline")
    times["Multi-Stage"] = bench(mod_multi, "Multi-Stage")

    # Also benchmark PyTorch
    start = time.time()
    for _ in range(50):
        _ = A.float() @ B.float()
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / 50

    # Calculate FLOPS
    flops = 2 * size * size * size

    # Print results
    print(f"\n{'Variant':<20} {'Time (ms)':<12} {'TFLOPS':<10} {'Speedup':<10}")
    print("-" * 70)

    baseline = times["No Pipeline"]
    print(f"{'No Pipeline':<20} {times['No Pipeline']*1e3:>10.3f}   "
          f"{flops/times['No Pipeline']/1e12:>8.2f}   {'1.00×':>8}")

    for name in ["Double Buffer", "Auto Pipeline", "Multi-Stage"]:
        speedup = baseline / times[name]
        print(f"{name:<20} {times[name]*1e3:>10.3f}   "
              f"{flops/times[name]/1e12:>8.2f}   {speedup:>7.2f}×")

    print("-" * 70)
    print(f"{'PyTorch (cuBLAS)':<20} {pytorch_time*1e3:>10.3f}   "
          f"{flops/pytorch_time/1e12:>8.2f}   "
          f"{baseline/pytorch_time:>7.2f}×")

    print("\n" + "="*70)
    print("Key Observations:")
    print(f"  • Double buffering gives {baseline/times['Double Buffer']:.2f}× speedup")
    print(f"  • Auto pipeline is {baseline/times['Auto Pipeline']:.2f}× faster")
    print(f"  • More stages may not always help (diminishing returns)")
    print("="*70)


def visualize_pipeline_execution():
    """Visualize how pipelining overlaps operations."""
    print("\n" + "="*70)
    print("Pipeline Execution Timeline")
    print("="*70)

    print("\nWithout Pipelining (Sequential):")
    print("  Time →")
    print("  Iter 0: [Load 0]       [Compute 0]")
    print("  Iter 1:                            [Load 1]       [Compute 1]")
    print("  Iter 2:                                                       [Load 2]       [Compute 2]")
    print("  Total: 6 time units")

    print("\nWith 2-Stage Pipelining (Overlapped):")
    print("  Time →")
    print("  Iter 0: [Load 0]       [Compute 0]")
    print("  Iter 1:          [Load 1]       [Compute 1]")
    print("  Iter 2:                   [Load 2]       [Compute 2]")
    print("  Total: 4 time units (1.5× faster!)")

    print("\nWith 3-Stage Pipelining (More Overlap):")
    print("  Time →")
    print("  Iter 0: [Load 0] [Compute 0]")
    print("  Iter 1:       [Load 1] [Compute 1]")
    print("  Iter 2:             [Load 2] [Compute 2]")
    print("  Total: 3 time units (2× faster!)")

    print("\nNote: Actual speedup depends on load/compute time ratio")
    print("="*70)


def analyze_memory_requirements():
    """Analyze shared memory requirements for pipelining."""
    print("\n" + "="*70)
    print("Shared Memory Requirements for Pipelining")
    print("="*70)

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    element_size = 2  # float16

    # No pipelining
    smem_single = (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * element_size
    print(f"\nNo Pipelining (Single Buffer):")
    print(f"  A tile: {BLOCK_M}×{BLOCK_K} = {BLOCK_M*BLOCK_K} elements")
    print(f"  B tile: {BLOCK_K}×{BLOCK_N} = {BLOCK_K*BLOCK_N} elements")
    print(f"  Total:  {smem_single} bytes = {smem_single/1024:.1f} KB")

    # Double buffering
    smem_double = 2 * smem_single
    print(f"\nDouble Buffering (2 Stages):")
    print(f"  Total:  {smem_double} bytes = {smem_double/1024:.1f} KB")
    print(f"  Overhead: {(smem_double/smem_single - 1)*100:.0f}%")

    # Triple buffering
    smem_triple = 3 * smem_single
    print(f"\nTriple Buffering (3 Stages):")
    print(f"  Total:  {smem_triple} bytes = {smem_triple/1024:.1f} KB")
    print(f"  Overhead: {(smem_triple/smem_single - 1)*100:.0f}%")

    # Compare to GPU limits
    print(f"\nGPU Limits (e.g., A100):")
    print(f"  Max shared memory per block: 164 KB")
    print(f"  Single buffer uses:  {smem_single/1024/164*100:.1f}%")
    print(f"  Double buffer uses:  {smem_double/1024/164*100:.1f}%")
    print(f"  Triple buffer uses:  {smem_triple/1024/164*100:.1f}%")

    print("\nConclusion: More stages = better overlap but uses more smem")
    print("           Must balance performance vs resource usage")
    print("="*70)


def main():
    """Run all tests, benchmarks, and demonstrations."""
    print("="*70)
    print("TileLang Software Pipelining Examples")
    print("="*70)

    # Tests
    test_all_variants()

    # Benchmarks
    benchmark_pipeline_variants()

    # Visualizations and analysis
    visualize_pipeline_execution()
    analyze_memory_requirements()

    print("\n✓ All tests passed!")
    print("\nKey Takeaways:")
    print("1. Software pipelining overlaps memory and compute")
    print("2. Can achieve 1.5-2× speedup with proper pipelining")
    print("3. TileLang's T.pipeline() automates complex scheduling")
    print("4. More stages improve performance but use more shared memory")
    print("5. Pipelining is critical for reaching peak GPU performance")


if __name__ == "__main__":
    main()
