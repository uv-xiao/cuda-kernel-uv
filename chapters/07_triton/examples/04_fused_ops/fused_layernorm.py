"""
Fused LayerNorm in Triton

LayerNorm: y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta

This kernel fuses:
- Mean computation
- Variance computation
- Normalization
- Affine transformation

Compared to PyTorch's unfused version (4-5 separate kernels).
"""

import torch
import triton
import triton.language as tl
import time


@triton.jit
def layernorm_kernel(
    output_ptr, input_ptr, gamma_ptr, beta_ptr,
    input_row_stride, output_row_stride,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused LayerNorm kernel.

    Each program normalizes one row.
    """
    row_idx = tl.program_id(0)

    # Pointers to this row
    input_row_ptr = input_ptr + row_idx * input_row_stride
    output_row_ptr = output_ptr + row_idx * output_row_stride

    # Column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load row
    x = tl.load(input_row_ptr + col_offsets, mask=mask, other=0.0)

    # Compute mean
    mean = tl.sum(x, axis=0) / n_cols

    # Compute variance
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / n_cols

    # Normalize
    rstd = 1 / tl.sqrt(var + eps)
    x_normed = x_centered * rstd

    # Load gamma and beta
    gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + col_offsets, mask=mask, other=0.0)

    # Affine transform
    y = x_normed * gamma + beta

    # Store output
    tl.store(output_row_ptr + col_offsets, y, mask=mask)


def triton_layernorm(x, gamma, beta, eps=1e-5):
    """
    Fused LayerNorm using Triton.

    Args:
        x: Input tensor (n_rows, n_cols)
        gamma: Scale parameter (n_cols,)
        beta: Shift parameter (n_cols,)
        eps: Epsilon for numerical stability
    """
    n_rows, n_cols = x.shape
    assert gamma.shape[0] == n_cols
    assert beta.shape[0] == n_cols

    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    grid = (n_rows,)

    layernorm_kernel[grid](
        output, x, gamma, beta,
        x.stride(0), output.stride(0),
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def pytorch_layernorm(x, gamma, beta, eps=1e-5):
    """Reference PyTorch implementation."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_normed = (x - mean) / torch.sqrt(var + eps)
    return x_normed * gamma + beta


def test_correctness():
    """Test correctness."""
    print("\nTesting Correctness")
    print("=" * 70)

    test_cases = [
        (1, 128),
        (128, 256),
        (1024, 512),
        (4096, 1024),
    ]

    for n_rows, n_cols in test_cases:
        x = torch.randn(n_rows, n_cols, device='cuda')
        gamma = torch.randn(n_cols, device='cuda')
        beta = torch.randn(n_cols, device='cuda')
        eps = 1e-5

        result_triton = triton_layernorm(x, gamma, beta, eps)
        result_torch = pytorch_layernorm(x, gamma, beta, eps)

        assert torch.allclose(result_triton, result_torch, rtol=1e-4, atol=1e-4), \
            f"Mismatch at shape ({n_rows}, {n_cols})"

        print(f"  Shape ({n_rows:4d}, {n_cols:4d}): PASS")

    print("All tests passed!")


def benchmark(n_rows, n_cols, num_iterations=100):
    """Benchmark Triton vs PyTorch LayerNorm."""
    print(f"\nBenchmarking LayerNorm ({n_rows}x{n_cols})")
    print("=" * 70)

    x = torch.randn(n_rows, n_cols, device='cuda')
    gamma = torch.randn(n_cols, device='cuda')
    beta = torch.randn(n_cols, device='cuda')
    eps = 1e-5

    # Warmup
    for _ in range(10):
        _ = triton_layernorm(x, gamma, beta, eps)
        _ = pytorch_layernorm(x, gamma, beta, eps)
    torch.cuda.synchronize()

    # Benchmark Triton
    start = time.time()
    for _ in range(num_iterations):
        result_triton = triton_layernorm(x, gamma, beta, eps)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / num_iterations * 1000

    # Benchmark PyTorch
    start = time.time()
    for _ in range(num_iterations):
        result_torch = pytorch_layernorm(x, gamma, beta, eps)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / num_iterations * 1000

    # Benchmark PyTorch native
    ln = torch.nn.LayerNorm(n_cols).cuda()
    ln.weight.data = gamma
    ln.bias.data = beta

    start = time.time()
    for _ in range(num_iterations):
        result_native = ln(x)
    torch.cuda.synchronize()
    native_time = (time.time() - start) / num_iterations * 1000

    print(f"Triton:         {triton_time:6.3f} ms")
    print(f"PyTorch (unfused): {torch_time:6.3f} ms")
    print(f"PyTorch (native):  {native_time:6.3f} ms")
    print(f"Speedup vs unfused: {torch_time / triton_time:.2f}x")
    print(f"Speedup vs native:  {native_time / triton_time:.2f}x")


if __name__ == "__main__":
    print("Fused LayerNorm in Triton")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available. This example requires a GPU.")
        exit(1)

    test_correctness()

    benchmark(1024, 512)
    benchmark(4096, 1024)
    benchmark(8192, 2048)

    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("1. LayerNorm naturally fuses: mean, var, normalize, affine")
    print("2. 2-4x speedup vs unfused PyTorch operations")
    print("3. Competitive with PyTorch native (which is also fused)")
    print("=" * 70)
