"""
Fused Add-Multiply Operation

This example demonstrates kernel fusion by combining:
    output = (x + y) * z

Why fuse?
- Reduces memory traffic (2 reads instead of 3)
- Eliminates intermediate storage
- Better cache utilization
- 2-3x speedup for memory-bound operations
"""

import torch
import triton
import triton.language as tl
import time


@triton.jit
def fused_add_mul_kernel(
    x_ptr, y_ptr, z_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: output = (x + y) * z

    Computed in single pass without intermediate storage.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load all inputs once
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = tl.load(z_ptr + offsets, mask=mask)

    # Fused computation
    output = (x + y) * z

    # Write result
    tl.store(output_ptr + offsets, output, mask=mask)


def triton_fused_add_mul(x, y, z):
    """Fused implementation."""
    assert x.shape == y.shape == z.shape
    output = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    fused_add_mul_kernel[grid](x, y, z, output, n_elements, BLOCK_SIZE)

    return output


def pytorch_unfused_add_mul(x, y, z):
    """Unfused implementation (separate kernels)."""
    temp = x + y  # Kernel 1: add
    output = temp * z  # Kernel 2: multiply
    return output


def analyze_memory_traffic():
    """
    Compare memory traffic between fused and unfused versions.
    """
    print("\nMemory Traffic Analysis")
    print("=" * 70)

    n = 1_000_000
    element_size = 4  # float32

    print(f"Operation: output = (x + y) * z")
    print(f"Elements: {n:,}")
    print(f"Element size: {element_size} bytes")
    print()

    print("Unfused (PyTorch default):")
    print("  Kernel 1: temp = x + y")
    print(f"    Read:  x, y = {2 * n * element_size / 1e6:.2f} MB")
    print(f"    Write: temp = {n * element_size / 1e6:.2f} MB")
    print("  Kernel 2: output = temp * z")
    print(f"    Read:  temp, z = {2 * n * element_size / 1e6:.2f} MB")
    print(f"    Write: output = {n * element_size / 1e6:.2f} MB")
    unfused_bytes = 6 * n * element_size
    print(f"  Total: {unfused_bytes / 1e6:.2f} MB")
    print()

    print("Fused (Triton):")
    print("  Kernel: output = (x + y) * z")
    print(f"    Read:  x, y, z = {3 * n * element_size / 1e6:.2f} MB")
    print(f"    Write: output = {n * element_size / 1e6:.2f} MB")
    fused_bytes = 4 * n * element_size
    print(f"  Total: {fused_bytes / 1e6:.2f} MB")
    print()

    reduction = (unfused_bytes - fused_bytes) / unfused_bytes * 100
    print(f"Memory Traffic Reduction: {reduction:.1f}%")
    print(f"Expected Speedup: ~{unfused_bytes / fused_bytes:.2f}x")


def benchmark():
    """
    Benchmark fused vs unfused implementation.
    """
    print("\nBenchmark Results")
    print("=" * 70)

    sizes = [1_000_000, 10_000_000, 100_000_000]

    for size in sizes:
        print(f"\nSize: {size:,} elements")

        x = torch.randn(size, device='cuda')
        y = torch.randn(size, device='cuda')
        z = torch.randn(size, device='cuda')

        # Warmup
        for _ in range(10):
            _ = triton_fused_add_mul(x, y, z)
            _ = pytorch_unfused_add_mul(x, y, z)
        torch.cuda.synchronize()

        # Benchmark fused
        num_iterations = 100
        start = time.time()
        for _ in range(num_iterations):
            result_fused = triton_fused_add_mul(x, y, z)
        torch.cuda.synchronize()
        fused_time = (time.time() - start) / num_iterations * 1000

        # Benchmark unfused
        start = time.time()
        for _ in range(num_iterations):
            result_unfused = pytorch_unfused_add_mul(x, y, z)
        torch.cuda.synchronize()
        unfused_time = (time.time() - start) / num_iterations * 1000

        # Verify correctness
        assert torch.allclose(result_fused, result_unfused, rtol=1e-5)

        # Calculate bandwidth
        bytes_fused = 4 * size * 4  # 3 reads + 1 write
        bytes_unfused = 6 * size * 4  # 4 reads + 2 writes
        bw_fused = bytes_fused / (fused_time * 1e-3) / 1e9
        bw_unfused = bytes_unfused / (unfused_time * 1e-3) / 1e9

        print(f"  Fused:   {fused_time:6.3f} ms  ({bw_fused:6.2f} GB/s)")
        print(f"  Unfused: {unfused_time:6.3f} ms  ({bw_unfused:6.2f} GB/s)")
        print(f"  Speedup: {unfused_time / fused_time:.2f}x")


def explain_fusion_benefits():
    """
    Explain why fusion is beneficial.
    """
    print("\nWhy Kernel Fusion Matters")
    print("=" * 70)

    print("1. Reduced Memory Traffic")
    print("   - Intermediate results stay in registers")
    print("   - No round-trip to DRAM")
    print("   - 33-50% less memory bandwidth")
    print()

    print("2. Improved Cache Utilization")
    print("   - Data loaded once, used multiple times")
    print("   - Better temporal locality")
    print("   - Reduced cache pressure")
    print()

    print("3. Lower Kernel Launch Overhead")
    print("   - One kernel instead of multiple")
    print("   - Less synchronization")
    print("   - Better for small tensors")
    print()

    print("4. When Fusion Helps Most:")
    print("   - Memory-bound operations (low arithmetic intensity)")
    print("   - Element-wise operations")
    print("   - Sequential dependencies")
    print("   - Small to medium tensors")
    print()

    print("5. When Fusion May Not Help:")
    print("   - Compute-bound operations")
    print("   - Intermediate results reused elsewhere")
    print("   - Very large tensors (limited by compute)")


def visualize_execution():
    """
    Visualize execution patterns.
    """
    print("\nExecution Patterns")
    print("=" * 70)

    print("Unfused Execution:")
    print("  ┌─────────────┐")
    print("  │ Kernel 1    │  temp = x + y")
    print("  │ (Add)       │")
    print("  └──────┬──────┘")
    print("         │ temp stored in DRAM")
    print("  ┌──────▼──────┐")
    print("  │ Kernel 2    │  output = temp * z")
    print("  │ (Multiply)  │")
    print("  └─────────────┘")
    print()

    print("Fused Execution:")
    print("  ┌─────────────┐")
    print("  │ Fused Kernel│  output = (x + y) * z")
    print("  │ (Add+Mul)   │  [temp in registers]")
    print("  └─────────────┘")
    print()

    print("Key Difference:")
    print("  - Unfused: 2 kernel launches, temp written to DRAM")
    print("  - Fused: 1 kernel launch, temp stays in registers")


def test_correctness():
    """Test correctness."""
    print("\nTesting Correctness")
    print("=" * 70)

    sizes = [1, 100, 1024, 10000]

    for size in sizes:
        x = torch.randn(size, device='cuda')
        y = torch.randn(size, device='cuda')
        z = torch.randn(size, device='cuda')

        result_fused = triton_fused_add_mul(x, y, z)
        result_unfused = pytorch_unfused_add_mul(x, y, z)

        assert torch.allclose(result_fused, result_unfused, rtol=1e-5), \
            f"Mismatch at size {size}"
        print(f"  Size {size:5d}: PASS")

    print("All tests passed!")


if __name__ == "__main__":
    print("Fused Add-Multiply Operation")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available. This example requires a GPU.")
        exit(1)

    # 1. Explain fusion benefits
    explain_fusion_benefits()

    # 2. Visualize execution
    visualize_execution()

    # 3. Analyze memory traffic
    analyze_memory_traffic()

    # 4. Test correctness
    test_correctness()

    # 5. Benchmark
    benchmark()

    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("1. Fusion reduces memory traffic by eliminating intermediate storage")
    print("2. Expect 1.5-3x speedup for memory-bound operations")
    print("3. Most beneficial for element-wise operation chains")
    print("4. Triton makes fusion easy - just combine operations in one kernel")
    print("=" * 70)
