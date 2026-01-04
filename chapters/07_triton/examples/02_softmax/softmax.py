"""
Triton Softmax - Reductions and Numerical Stability

This example demonstrates:
- Reduction operations (max, sum)
- Multi-dimensional indexing
- Numerically stable softmax implementation
- Row-wise parallel processing

Softmax formula:
    softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

The subtraction of max(x) prevents overflow in exp().
"""

import torch
import triton
import triton.language as tl
import time


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Numerically stable softmax kernel.

    Each program processes one row of the input matrix.

    Args:
        output_ptr: Pointer to output matrix
        input_ptr: Pointer to input matrix
        input_row_stride: Stride to move to next row in input
        output_row_stride: Stride to move to next row in output
        n_cols: Number of columns
        BLOCK_SIZE: Number of elements per block (must be >= n_cols)
    """
    # Each program handles one row
    row_idx = tl.program_id(0)

    # Calculate starting pointer for this row
    row_start_ptr = input_ptr + row_idx * input_row_stride

    # Create column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load the row
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    # Using other=-inf ensures max won't be affected by out-of-bounds values

    # Step 1: Find maximum value in row (for numerical stability)
    row_max = tl.max(row, axis=0)

    # Step 2: Subtract max and compute exponential
    # x - max(x) prevents overflow in exp()
    row_shifted = row - row_max
    numerator = tl.exp(row_shifted)

    # Step 3: Compute sum of exponentials
    denominator = tl.sum(numerator, axis=0)

    # Step 4: Normalize
    softmax_output = numerator / denominator

    # Step 5: Write result
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Compute softmax along the last dimension using Triton.

    Args:
        x: Input tensor of shape (..., n_cols)

    Returns:
        Softmax output of same shape as input
    """
    # Flatten to 2D for simplicity: (n_rows, n_cols)
    x_2d = x.view(-1, x.shape[-1])
    n_rows, n_cols = x_2d.shape

    # Allocate output
    output = torch.empty_like(x_2d)

    # Choose BLOCK_SIZE that can fit all columns
    # Round up to next power of 2 for efficiency
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Launch kernel with one program per row
    num_programs = n_rows
    grid = (num_programs,)

    softmax_kernel[grid](
        output,
        x_2d,
        x_2d.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.view_as(x)


@triton.jit
def softmax_kernel_online(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Online softmax kernel - computes max and sum in single pass.

    More memory efficient for very large rows that don't fit in SRAM.
    Uses the "online" algorithm that processes chunks sequentially.
    """
    row_idx = tl.program_id(0)

    # Number of blocks to process
    num_blocks = tl.cdiv(n_cols, BLOCK_SIZE)

    # Initialize running max and sum
    running_max = -float('inf')
    running_sum = 0.0

    # First pass: compute max and sum
    for block_idx in range(num_blocks):
        col_start = block_idx * BLOCK_SIZE
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        # Load block
        input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
        block_vals = tl.load(input_ptrs, mask=mask, other=-float('inf'))

        # Update running max
        block_max = tl.max(block_vals, axis=0)
        old_max = running_max
        running_max = tl.maximum(running_max, block_max)

        # Update running sum with correction factor
        # When max changes, we need to rescale previous sum
        running_sum = running_sum * tl.exp(old_max - running_max)
        running_sum += tl.sum(tl.exp(block_vals - running_max), axis=0)

    # Second pass: compute and store softmax
    for block_idx in range(num_blocks):
        col_start = block_idx * BLOCK_SIZE
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        # Load block
        input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
        block_vals = tl.load(input_ptrs, mask=mask, other=-float('inf'))

        # Compute softmax
        softmax_vals = tl.exp(block_vals - running_max) / running_sum

        # Store result
        output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
        tl.store(output_ptrs, softmax_vals, mask=mask)


def triton_softmax_online(x: torch.Tensor) -> torch.Tensor:
    """
    Online softmax - better for very large row sizes.
    """
    x_2d = x.view(-1, x.shape[-1])
    n_rows, n_cols = x_2d.shape

    output = torch.empty_like(x_2d)
    BLOCK_SIZE = 1024  # Fixed block size

    grid = (n_rows,)

    softmax_kernel_online[grid](
        output,
        x_2d,
        x_2d.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.view_as(x)


def test_numerical_stability():
    """
    Demonstrate why numerical stability matters.
    """
    print("\nNumerical Stability Test")
    print("=" * 70)

    # Create input with large values that would overflow naive softmax
    x = torch.tensor([[1000.0, 1001.0, 1002.0]], device='cuda')

    print(f"Input: {x}")
    print(f"Max value: {x.max().item()}")
    print()

    # Naive softmax (would overflow)
    print("Naive approach: exp(x) / sum(exp(x))")
    try:
        exp_x = torch.exp(x)
        print(f"  exp(x) = {exp_x}")
        print(f"  Result: OVERFLOW (inf values)")
    except:
        pass

    # Stable softmax
    print("\nStable approach: exp(x - max(x)) / sum(exp(x - max(x)))")
    x_shifted = x - x.max()
    print(f"  x - max(x) = {x_shifted}")
    exp_shifted = torch.exp(x_shifted)
    print(f"  exp(x - max(x)) = {exp_shifted}")
    stable_result = exp_shifted / exp_shifted.sum()
    print(f"  Result: {stable_result}")
    print(f"  Sum: {stable_result.sum().item():.10f} (should be 1.0)")

    # Triton result
    triton_result = triton_softmax(x)
    print(f"\nTriton result: {triton_result}")
    print(f"Match: {torch.allclose(triton_result, stable_result)}")


def test_correctness():
    """
    Test softmax for correctness across various shapes.
    """
    print("\nTesting Correctness")
    print("=" * 70)

    test_cases = [
        (1, 4),      # Single row
        (4, 1),      # Single column
        (128, 256),  # Medium
        (512, 512),  # Square
        (1024, 64),  # Many rows
    ]

    for n_rows, n_cols in test_cases:
        x = torch.randn(n_rows, n_cols, device='cuda')

        # Compute with Triton
        output_triton = triton_softmax(x)

        # Compute with PyTorch
        output_torch = torch.softmax(x, dim=-1)

        # Verify
        assert torch.allclose(output_triton, output_torch, rtol=1e-5, atol=1e-6), \
            f"Mismatch at shape ({n_rows}, {n_cols})"

        # Verify properties
        assert torch.allclose(output_triton.sum(dim=-1), torch.ones(n_rows, device='cuda')), \
            "Softmax should sum to 1"
        assert (output_triton >= 0).all(), "Softmax should be non-negative"

        print(f"  Shape ({n_rows:4d}, {n_cols:4d}): PASS")

    print("All tests passed!")


def benchmark_softmax(n_rows: int, n_cols: int, num_iterations: int = 100):
    """
    Benchmark Triton vs PyTorch softmax.
    """
    print(f"\nBenchmarking softmax (shape={n_rows}x{n_cols})")
    print("=" * 70)

    x = torch.randn(n_rows, n_cols, device='cuda')

    # Warmup
    for _ in range(10):
        _ = triton_softmax(x)
        _ = torch.softmax(x, dim=-1)
    torch.cuda.synchronize()

    # Benchmark Triton (standard)
    start = time.time()
    for _ in range(num_iterations):
        result_triton = triton_softmax(x)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / num_iterations * 1000

    # Benchmark Triton (online)
    start = time.time()
    for _ in range(num_iterations):
        result_online = triton_softmax_online(x)
    torch.cuda.synchronize()
    online_time = (time.time() - start) / num_iterations * 1000

    # Benchmark PyTorch
    start = time.time()
    for _ in range(num_iterations):
        result_torch = torch.softmax(x, dim=-1)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / num_iterations * 1000

    # Verify correctness
    assert torch.allclose(result_triton, result_torch, rtol=1e-5, atol=1e-6)
    assert torch.allclose(result_online, result_torch, rtol=1e-5, atol=1e-6)

    print(f"Triton (standard): {triton_time:6.3f} ms")
    print(f"Triton (online):   {online_time:6.3f} ms")
    print(f"PyTorch:           {torch_time:6.3f} ms")
    print(f"Speedup (std):     {torch_time / triton_time:.2f}x")
    print(f"Speedup (online):  {torch_time / online_time:.2f}x")


def analyze_algorithm():
    """
    Explain the softmax algorithm step by step.
    """
    print("\nSoftmax Algorithm Analysis")
    print("=" * 70)

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device='cuda')

    print(f"Input: {x}\n")

    print("Step 1: Find maximum (for stability)")
    max_val = x.max()
    print(f"  max(x) = {max_val.item()}")

    print("\nStep 2: Subtract max")
    x_shifted = x - max_val
    print(f"  x - max(x) = {x_shifted}")

    print("\nStep 3: Compute exponential")
    exp_shifted = torch.exp(x_shifted)
    print(f"  exp(x - max(x)) = {exp_shifted}")

    print("\nStep 4: Compute sum")
    sum_exp = exp_shifted.sum()
    print(f"  sum(exp(x - max(x))) = {sum_exp.item():.4f}")

    print("\nStep 5: Normalize")
    softmax = exp_shifted / sum_exp
    print(f"  softmax(x) = {softmax}")
    print(f"  sum(softmax(x)) = {softmax.sum().item():.10f}")

    print("\nProperties:")
    print(f"  - All values in [0, 1]: {(softmax >= 0).all() and (softmax <= 1).all()}")
    print(f"  - Sum equals 1: {torch.allclose(softmax.sum(), torch.tensor(1.0))}")
    print(f"  - Larger inputs → larger outputs: {(softmax[0, 1:] > softmax[0, :-1]).all()}")


if __name__ == "__main__":
    print("Triton Softmax Tutorial")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available. This example requires a GPU.")
        exit(1)

    # 1. Analyze algorithm
    analyze_algorithm()

    # 2. Test numerical stability
    test_numerical_stability()

    # 3. Test correctness
    test_correctness()

    # 4. Benchmark performance
    benchmark_softmax(1024, 512)
    benchmark_softmax(4096, 1024)
    benchmark_softmax(8192, 2048)

    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("1. Subtract max(x) before exp() for numerical stability")
    print("2. Use tl.max() and tl.sum() for reductions")
    print("3. Each program processes one row (row-wise parallelism)")
    print("4. Online algorithm better for very large rows")
    print("5. Triton matches PyTorch performance for softmax")
    print("=" * 70)
