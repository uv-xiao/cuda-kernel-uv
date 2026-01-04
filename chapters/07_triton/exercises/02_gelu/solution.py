"""
Exercise 02: GELU Activation - Solution

Implements GELU using the tanh approximation.
"""

import torch
import triton
import triton.language as tl
import time


@triton.jit
def gelu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    GELU activation kernel.

    Formula: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    """
    # Program ID
    pid = tl.program_id(0)

    # Calculate offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # GELU computation
    # Constants
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    coeff = 0.044715

    # Compute x^3
    x_cubed = x * x * x

    # Inner term: sqrt(2/pi) * (x + 0.044715 * x^3)
    inner = sqrt_2_over_pi * (x + coeff * x_cubed)

    # Apply tanh
    tanh_inner = tl.libdevice.tanh(inner)

    # Final GELU
    gelu = 0.5 * x * (1.0 + tanh_inner)

    # Store output
    tl.store(output_ptr + offsets, gelu, mask=mask)


def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Triton GELU implementation.
    """
    output = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    gelu_kernel[grid](x, output, n_elements, BLOCK_SIZE)

    return output


def test_correctness():
    """Test against PyTorch GELU."""
    print("Testing Correctness")
    print("=" * 70)

    # Test various input ranges
    test_cases = [
        ("Small values", torch.randn(1000, device='cuda') * 0.1),
        ("Normal values", torch.randn(1000, device='cuda')),
        ("Large values", torch.randn(1000, device='cuda') * 10),
        ("Mixed values", torch.randn(1000, device='cuda') * 5),
        ("Large tensor", torch.randn(1000000, device='cuda')),
    ]

    for name, x in test_cases:
        result_triton = triton_gelu(x)
        result_torch = torch.nn.functional.gelu(x, approximate='tanh')

        # Calculate relative error
        rel_error = ((result_triton - result_torch).abs() / (result_torch.abs() + 1e-8)).max()

        passed = rel_error < 1e-4

        print(f"  {name:20s}: {'PASS' if passed else 'FAIL'} (max rel error: {rel_error:.2e})")

        if not passed:
            return False

    print("All tests passed!")
    return True


def benchmark():
    """Benchmark Triton vs PyTorch GELU."""
    print("\nBenchmarking")
    print("=" * 70)

    sizes = [1_000_000, 10_000_000, 100_000_000]

    for size in sizes:
        print(f"\nSize: {size:,} elements")

        x = torch.randn(size, device='cuda')

        # Warmup
        for _ in range(10):
            _ = triton_gelu(x)
            _ = torch.nn.functional.gelu(x, approximate='tanh')
        torch.cuda.synchronize()

        # Benchmark Triton
        num_iterations = 100
        start = time.time()
        for _ in range(num_iterations):
            result_triton = triton_gelu(x)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / num_iterations * 1000

        # Benchmark PyTorch
        start = time.time()
        for _ in range(num_iterations):
            result_torch = torch.nn.functional.gelu(x, approximate='tanh')
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / num_iterations * 1000

        # Calculate bandwidth
        bytes_accessed = 2 * size * 4  # 1 read + 1 write, fp32
        triton_bw = bytes_accessed / (triton_time * 1e-3) / 1e9
        torch_bw = bytes_accessed / (torch_time * 1e-3) / 1e9

        print(f"  Triton:  {triton_time:6.3f} ms  ({triton_bw:6.2f} GB/s)")
        print(f"  PyTorch: {torch_time:6.3f} ms  ({torch_bw:6.2f} GB/s)")
        print(f"  Speedup: {torch_time / triton_time:.2f}x")


def visualize_gelu():
    """Visualize GELU function."""
    print("\nGELU Function Visualization")
    print("=" * 70)

    # Test points
    test_x = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]

    print("x      GELU(x)")
    print("-" * 25)

    for x_val in test_x:
        x = torch.tensor([x_val], device='cuda')
        gelu = triton_gelu(x).item()
        print(f"{x_val:5.1f}  {gelu:8.4f}")

    print()
    print("Properties:")
    print("- GELU(0) ≈ 0")
    print("- GELU(x) → x for large positive x")
    print("- GELU(x) → 0 for large negative x")
    print("- Smooth everywhere (differentiable)")


def explain_formula():
    """Explain the GELU formula."""
    print("\nGELU Formula Explanation")
    print("=" * 70)

    print("Exact formula:")
    print("  GELU(x) = x * Φ(x)")
    print("  where Φ(x) is the CDF of standard normal distribution")
    print()

    print("Tanh approximation (used here):")
    print("  GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))")
    print()

    print("Why tanh approximation?")
    print("  - Exact formula requires erf() which is expensive")
    print("  - Tanh approximation is accurate (error < 0.1%)")
    print("  - Faster to compute")
    print("  - Widely used in practice (GPT, BERT, etc.)")
    print()

    print("Step-by-step computation:")
    x = 1.0
    print(f"  x = {x}")
    x_cubed = x ** 3
    print(f"  x³ = {x_cubed}")
    inner = 0.7978845608028654 * (x + 0.044715 * x_cubed)
    print(f"  inner = √(2/π) * (x + 0.044715 * x³) = {inner:.6f}")
    tanh_inner = torch.tanh(torch.tensor(inner)).item()
    print(f"  tanh(inner) = {tanh_inner:.6f}")
    gelu = 0.5 * x * (1 + tanh_inner)
    print(f"  GELU(x) = 0.5 * x * (1 + tanh(inner)) = {gelu:.6f}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. This exercise requires a GPU.")
        exit(1)

    print("Exercise 02 Solution: GELU Activation")
    print("=" * 70)

    # Explain formula
    explain_formula()

    # Visualize GELU
    visualize_gelu()

    # Test correctness
    if test_correctness():
        # Benchmark performance
        benchmark()

    print("\n" + "=" * 70)
    print("Exercise completed successfully!")
    print("=" * 70)
