"""
Triton Vector Addition - First Triton Kernel

This example demonstrates the basics of Triton programming:
- Program IDs and block-based computation
- Loading and storing with masks
- Launching kernels from Python

Compare with CUDA vector addition from Chapter 01.
"""

import torch
import triton
import triton.language as tl
import time


@triton.jit
def add_kernel(
    x_ptr,  # Pointer to first input vector
    y_ptr,  # Pointer to second input vector
    output_ptr,  # Pointer to output vector
    n_elements,  # Size of vectors
    BLOCK_SIZE: tl.constexpr,  # Number of elements per block (compile-time constant)
):
    """
    Triton kernel for vector addition: output = x + y

    Each program instance handles BLOCK_SIZE elements.
    """
    # Get the program ID - which block of elements are we processing?
    pid = tl.program_id(axis=0)

    # Calculate the starting index for this block
    block_start = pid * BLOCK_SIZE

    # Create offsets for elements in this block
    # offsets = [block_start, block_start+1, ..., block_start+BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask to guard against out-of-bounds accesses
    # This is crucial when n_elements is not a multiple of BLOCK_SIZE
    mask = offsets < n_elements

    # Load data from DRAM into SRAM (masked load)
    # Elements outside bounds are not loaded (mask=False)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute the result
    output = x + y

    # Write back to DRAM (masked store)
    tl.store(output_ptr + offsets, output, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the Triton kernel.

    Args:
        x: First input tensor
        y: Second input tensor

    Returns:
        output: x + y
    """
    # Ensure inputs are on GPU and contiguous
    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape

    # Allocate output
    output = torch.empty_like(x)
    n_elements = x.numel()

    # Choose block size (tunable parameter)
    BLOCK_SIZE = 1024

    # Calculate grid size - how many blocks do we need?
    # We need ceil(n_elements / BLOCK_SIZE) blocks
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch kernel
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE)

    return output


def benchmark_add(size: int, num_iterations: int = 100):
    """
    Benchmark Triton vs PyTorch vector addition.
    """
    print(f"\nBenchmarking vector addition (size={size:,})")
    print("=" * 70)

    # Create test data
    x = torch.randn(size, device='cuda')
    y = torch.randn(size, device='cuda')

    # Warmup
    for _ in range(10):
        _ = triton_add(x, y)
        _ = x + y
    torch.cuda.synchronize()

    # Benchmark Triton
    start = time.time()
    for _ in range(num_iterations):
        result_triton = triton_add(x, y)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / num_iterations * 1000  # ms

    # Benchmark PyTorch
    start = time.time()
    for _ in range(num_iterations):
        result_pytorch = x + y
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / num_iterations * 1000  # ms

    # Verify correctness
    assert torch.allclose(result_triton, result_pytorch, rtol=1e-5)

    # Calculate bandwidth (GB/s)
    # We read 2 vectors and write 1 vector (3 * size * 4 bytes for fp32)
    bytes_accessed = 3 * size * 4
    triton_bandwidth = bytes_accessed / (triton_time * 1e-3) / 1e9
    pytorch_bandwidth = bytes_accessed / (pytorch_time * 1e-3) / 1e9

    print(f"Triton:  {triton_time:6.3f} ms  ({triton_bandwidth:6.2f} GB/s)")
    print(f"PyTorch: {pytorch_time:6.3f} ms  ({pytorch_bandwidth:6.2f} GB/s)")
    print(f"Speedup: {pytorch_time / triton_time:.2f}x")


def test_correctness():
    """
    Test vector addition for correctness.
    """
    print("Testing correctness...")

    # Test various sizes
    sizes = [1, 127, 128, 1023, 1024, 10000, 1000000]

    for size in sizes:
        x = torch.randn(size, device='cuda')
        y = torch.randn(size, device='cuda')

        result_triton = triton_add(x, y)
        result_pytorch = x + y

        assert torch.allclose(result_triton, result_pytorch, rtol=1e-5), \
            f"Mismatch at size {size}"
        print(f"  Size {size:7d}: PASS")

    print("All tests passed!")


def visualize_block_pattern():
    """
    Visualize how Triton divides work into blocks.
    """
    print("\nBlock-based Computation Pattern")
    print("=" * 70)

    n_elements = 5000
    BLOCK_SIZE = 1024

    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)

    print(f"Total elements: {n_elements}")
    print(f"Block size:     {BLOCK_SIZE}")
    print(f"Num blocks:     {num_blocks}")
    print()

    for pid in range(num_blocks):
        block_start = pid * BLOCK_SIZE
        block_end = min(block_start + BLOCK_SIZE, n_elements)
        num_valid = block_end - block_start

        print(f"Block {pid}: elements [{block_start:4d}:{block_end:4d}]  "
              f"({num_valid:4d} valid elements)")


def compare_with_cuda():
    """
    Compare Triton code structure with equivalent CUDA code.
    """
    print("\nTriton vs CUDA Comparison")
    print("=" * 70)

    cuda_code = """
// CUDA Version (from Chapter 01)
__global__ void add_kernel(float* x, float* y, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] + y[idx];
    }
}

// Launch: add_kernel<<<grid, block>>>(x, y, out, n);
"""

    triton_code = """
# Triton Version
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)

# Launch: add_kernel[grid](x, y, out, n, BLOCK_SIZE)
"""

    print("CUDA:")
    print(cuda_code)
    print("\nTriton:")
    print(triton_code)

    print("\nKey Differences:")
    print("1. CUDA: Thread-level programming (each thread = 1 element)")
    print("   Triton: Block-level programming (each program = BLOCK_SIZE elements)")
    print()
    print("2. CUDA: Manual indexing (blockIdx.x * blockDim.x + threadIdx.x)")
    print("   Triton: Automatic indexing (tl.arange)")
    print()
    print("3. CUDA: Scalar operations (out[idx] = x[idx] + y[idx])")
    print("   Triton: Vector operations (out = x + y)")
    print()
    print("4. CUDA: Explicit boundary check (if idx < n)")
    print("   Triton: Masked loads/stores (mask=offsets < n)")
    print()
    print("5. CUDA: C++ syntax")
    print("   Triton: Python syntax")


if __name__ == "__main__":
    print("Triton Vector Addition Tutorial")
    print("=" * 70)

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available. This example requires a GPU.")
        exit(1)

    # 1. Test correctness
    test_correctness()

    # 2. Visualize block pattern
    visualize_block_pattern()

    # 3. Benchmark performance
    for size in [1_000_000, 10_000_000, 100_000_000]:
        benchmark_add(size)

    # 4. Compare with CUDA
    compare_with_cuda()

    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("1. Triton uses block-based programming (vs CUDA's thread-based)")
    print("2. Masks handle boundary conditions automatically")
    print("3. Python-like syntax makes development faster")
    print("4. Performance matches PyTorch (both use optimized libraries)")
    print("=" * 70)
