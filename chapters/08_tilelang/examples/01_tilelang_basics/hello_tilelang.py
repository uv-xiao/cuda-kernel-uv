"""
TileLang Basics - Hello TileLang
================================

This example demonstrates the fundamental concepts of TileLang:
1. Defining kernels with @T.prim_func
2. Buffer declarations
3. Basic tile operations
4. Memory copies between global and shared memory

We'll implement simple operations to understand TileLang's syntax.
"""

import tilelang as T
import torch


# Example 1: Vector Addition
# ==========================
@T.prim_func
def vector_add(
    A: T.Buffer((1024,), "float32"),
    B: T.Buffer((1024,), "float32"),
    C: T.Buffer((1024,), "float32")
):
    """
    Simple vector addition: C = A + B

    This demonstrates:
    - Basic kernel structure
    - Thread indexing
    - Element-wise operations
    """
    with T.block("root"):
        # Get thread index
        tx = T.thread_binding(0, 1024, "threadIdx.x")

        # Read values
        a_val = A[tx]
        b_val = B[tx]

        # Compute
        c_val = a_val + b_val

        # Write result
        C[tx] = c_val


# Example 2: Tiled Vector Addition
# =================================
@T.prim_func
def vector_add_tiled(
    A: T.Buffer((4096,), "float32"),
    B: T.Buffer((4096,), "float32"),
    C: T.Buffer((4096,), "float32")
):
    """
    Vector addition with tiling and shared memory.

    This demonstrates:
    - Thread block organization
    - Shared memory allocation
    - Cooperative loading with T.copy()
    - Tiling for larger arrays
    """
    BLOCK_SIZE = 256

    with T.block("root"):
        # Block and thread indices
        bx = T.thread_binding(0, 4096 // BLOCK_SIZE, "blockIdx.x")
        tx = T.thread_binding(0, BLOCK_SIZE, "threadIdx.x")

        # Allocate shared memory tiles
        A_shared = T.alloc_shared([BLOCK_SIZE], "float32")
        B_shared = T.alloc_shared([BLOCK_SIZE], "float32")

        # Global memory offset for this block
        offset = bx * BLOCK_SIZE

        # Cooperative load into shared memory
        A_shared[tx] = A[offset + tx]
        B_shared[tx] = B[offset + tx]

        # Synchronize to ensure all threads have loaded data
        T.sync_threads()

        # Compute from shared memory
        C[offset + tx] = A_shared[tx] + B_shared[tx]


# Example 3: Matrix Transpose
# ============================
@T.prim_func
def matrix_transpose(
    A: T.Buffer((1024, 1024), "float32"),
    B: T.Buffer((1024, 1024), "float32")
):
    """
    Matrix transpose using shared memory to coalesce memory accesses.

    This demonstrates:
    - 2D thread blocks
    - Shared memory for memory coalescing
    - Avoiding bank conflicts with padding
    """
    TILE_SIZE = 32

    with T.block("root"):
        # 2D block and thread indices
        bx = T.thread_binding(0, 1024 // TILE_SIZE, "blockIdx.x")
        by = T.thread_binding(0, 1024 // TILE_SIZE, "blockIdx.y")
        tx = T.thread_binding(0, TILE_SIZE, "threadIdx.x")
        ty = T.thread_binding(0, TILE_SIZE, "threadIdx.y")

        # Shared memory with padding to avoid bank conflicts
        # +1 in second dimension prevents conflicts
        tile = T.alloc_shared([TILE_SIZE, TILE_SIZE + 1], "float32")

        # Read from global memory (coalesced)
        row = by * TILE_SIZE + ty
        col = bx * TILE_SIZE + tx
        tile[ty, tx] = A[row, col]

        # Synchronize
        T.sync_threads()

        # Write to global memory (transposed, coalesced)
        row_t = bx * TILE_SIZE + ty
        col_t = by * TILE_SIZE + tx
        B[row_t, col_t] = tile[tx, ty]


# Example 4: Reduction
# ====================
@T.prim_func
def block_reduce_sum(
    A: T.Buffer((1024,), "float32"),
    B: T.Buffer((4,), "float32")  # Output per block
):
    """
    Parallel reduction within a thread block.

    This demonstrates:
    - Shared memory reduction
    - Parallel reduction pattern
    - Warp-level synchronization
    """
    BLOCK_SIZE = 256

    with T.block("root"):
        bx = T.thread_binding(0, 4, "blockIdx.x")
        tx = T.thread_binding(0, BLOCK_SIZE, "threadIdx.x")

        # Shared memory for reduction
        shared = T.alloc_shared([BLOCK_SIZE], "float32")

        # Load data into shared memory
        offset = bx * BLOCK_SIZE
        shared[tx] = A[offset + tx]
        T.sync_threads()

        # Parallel reduction (tree-based)
        stride = BLOCK_SIZE // 2
        while stride > 0:
            if tx < stride:
                shared[tx] = shared[tx] + shared[tx + stride]
            T.sync_threads()
            stride = stride // 2

        # Thread 0 writes the result
        if tx == 0:
            B[bx] = shared[0]


# Example 5: Element-wise Operations with Broadcasting
# ====================================================
@T.prim_func
def vector_scalar_op(
    A: T.Buffer((4096,), "float16"),
    scale: T.Buffer((), "float16"),  # Scalar
    bias: T.Buffer((), "float16"),   # Scalar
    C: T.Buffer((4096,), "float16")
):
    """
    Element-wise operation: C = A * scale + bias

    This demonstrates:
    - Scalar broadcasting
    - Mixed operations
    - FP16 computation
    """
    BLOCK_SIZE = 256

    with T.block("root"):
        bx = T.thread_binding(0, 4096 // BLOCK_SIZE, "blockIdx.x")
        tx = T.thread_binding(0, BLOCK_SIZE, "threadIdx.x")

        # Load scalar values (broadcast)
        s = scale[()]
        b = bias[()]

        # Compute
        idx = bx * BLOCK_SIZE + tx
        C[idx] = A[idx] * s + b


# Testing Functions
# =================

def test_vector_add():
    """Test basic vector addition."""
    print("Testing vector_add...")

    # Create input tensors
    A = torch.randn(1024, device="cuda", dtype=torch.float32)
    B = torch.randn(1024, device="cuda", dtype=torch.float32)
    C = torch.zeros(1024, device="cuda", dtype=torch.float32)

    # Compile and run
    mod = T.compile(vector_add, target="cuda")
    mod(A, B, C)

    # Verify
    expected = A + B
    assert torch.allclose(C, expected, rtol=1e-5), "Vector add failed!"
    print("✓ vector_add passed")


def test_vector_add_tiled():
    """Test tiled vector addition."""
    print("Testing vector_add_tiled...")

    A = torch.randn(4096, device="cuda", dtype=torch.float32)
    B = torch.randn(4096, device="cuda", dtype=torch.float32)
    C = torch.zeros(4096, device="cuda", dtype=torch.float32)

    mod = T.compile(vector_add_tiled, target="cuda")
    mod(A, B, C)

    expected = A + B
    assert torch.allclose(C, expected, rtol=1e-5), "Tiled vector add failed!"
    print("✓ vector_add_tiled passed")


def test_matrix_transpose():
    """Test matrix transpose."""
    print("Testing matrix_transpose...")

    A = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
    B = torch.zeros(1024, 1024, device="cuda", dtype=torch.float32)

    mod = T.compile(matrix_transpose, target="cuda")
    mod(A, B)

    expected = A.T
    assert torch.allclose(B, expected, rtol=1e-5), "Transpose failed!"
    print("✓ matrix_transpose passed")


def test_block_reduce_sum():
    """Test block reduction."""
    print("Testing block_reduce_sum...")

    A = torch.randn(1024, device="cuda", dtype=torch.float32)
    B = torch.zeros(4, device="cuda", dtype=torch.float32)

    mod = T.compile(block_reduce_sum, target="cuda")
    mod(A, B)

    # Verify each block's sum
    for i in range(4):
        expected = A[i*256:(i+1)*256].sum()
        assert torch.allclose(B[i], expected, rtol=1e-4), f"Reduction failed for block {i}!"
    print("✓ block_reduce_sum passed")


def test_vector_scalar_op():
    """Test vector-scalar operations."""
    print("Testing vector_scalar_op...")

    A = torch.randn(4096, device="cuda", dtype=torch.float16)
    scale = torch.tensor(2.5, device="cuda", dtype=torch.float16)
    bias = torch.tensor(1.0, device="cuda", dtype=torch.float16)
    C = torch.zeros(4096, device="cuda", dtype=torch.float16)

    mod = T.compile(vector_scalar_op, target="cuda")
    mod(A, scale, bias, C)

    expected = A * scale + bias
    assert torch.allclose(C, expected, rtol=1e-3), "Vector-scalar op failed!"
    print("✓ vector_scalar_op passed")


def benchmark_comparison():
    """Compare TileLang with PyTorch native operations."""
    print("\n" + "="*60)
    print("Performance Comparison: TileLang vs PyTorch")
    print("="*60)

    import time

    # Setup
    size = 4096
    A = torch.randn(size, device="cuda", dtype=torch.float32)
    B = torch.randn(size, device="cuda", dtype=torch.float32)
    C = torch.zeros(size, device="cuda", dtype=torch.float32)

    # Compile TileLang kernel
    mod = T.compile(vector_add_tiled, target="cuda")

    # Warmup
    for _ in range(10):
        mod(A, B, C)
        _ = A + B
    torch.cuda.synchronize()

    # Benchmark TileLang
    start = time.time()
    for _ in range(1000):
        mod(A, B, C)
    torch.cuda.synchronize()
    tilelang_time = (time.time() - start) / 1000

    # Benchmark PyTorch
    start = time.time()
    for _ in range(1000):
        C = A + B
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / 1000

    print(f"Vector size: {size}")
    print(f"TileLang:    {tilelang_time*1e6:.2f} μs")
    print(f"PyTorch:     {pytorch_time*1e6:.2f} μs")
    print(f"Overhead:    {(tilelang_time/pytorch_time - 1)*100:.1f}%")
    print("="*60)


def main():
    """Run all examples and tests."""
    print("="*60)
    print("TileLang Hello World Examples")
    print("="*60)

    # Run all tests
    test_vector_add()
    test_vector_add_tiled()
    test_matrix_transpose()
    test_block_reduce_sum()
    test_vector_scalar_op()

    # Benchmark
    benchmark_comparison()

    print("\n✓ All tests passed!")
    print("\nKey Takeaways:")
    print("1. TileLang kernels are defined with @T.prim_func")
    print("2. Thread binding uses T.thread_binding()")
    print("3. Shared memory uses T.alloc_shared()")
    print("4. Synchronization uses T.sync_threads()")
    print("5. TileLang compiles to efficient CUDA code")


if __name__ == "__main__":
    main()
