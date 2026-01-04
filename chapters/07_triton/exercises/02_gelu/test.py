"""
Test Suite for Exercise 02: GELU Activation
"""

import torch
import sys

try:
    from solution import triton_gelu
except ImportError:
    print("Error: Could not import solution. Make sure solution.py is complete.")
    sys.exit(1)


def test_correctness():
    """Test GELU correctness against PyTorch."""
    print("Test 1: Correctness")

    x = torch.randn(10000, device='cuda')
    result = triton_gelu(x)
    expected = torch.nn.functional.gelu(x, approximate='tanh')

    rel_error = ((result - expected).abs() / (expected.abs() + 1e-8)).max()

    assert rel_error < 1e-4, f"Relative error too large: {rel_error}"
    print(f"  PASS (max rel error: {rel_error:.2e})")


def test_edge_cases():
    """Test edge cases."""
    print("Test 2: Edge Cases")

    # Zero
    x = torch.zeros(100, device='cuda')
    result = triton_gelu(x)
    assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)

    # Large positive
    x = torch.ones(100, device='cuda') * 10
    result = triton_gelu(x)
    expected = torch.nn.functional.gelu(x, approximate='tanh')
    assert torch.allclose(result, expected, rtol=1e-4)

    # Large negative
    x = torch.ones(100, device='cuda') * -10
    result = triton_gelu(x)
    expected = torch.nn.functional.gelu(x, approximate='tanh')
    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-6)

    print("  PASS")


def test_large_tensor():
    """Test with large tensor."""
    print("Test 3: Large Tensor")

    x = torch.randn(10_000_000, device='cuda')
    result = triton_gelu(x)
    expected = torch.nn.functional.gelu(x, approximate='tanh')

    rel_error = ((result - expected).abs() / (expected.abs() + 1e-8)).max()
    assert rel_error < 1e-4

    print(f"  PASS (max rel error: {rel_error:.2e})")


def run_all_tests():
    """Run all tests."""
    print("Running Test Suite")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available. Tests require a GPU.")
        return False

    try:
        test_correctness()
        test_edge_cases()
        test_large_tensor()

        print("\nAll tests passed!")
        return True

    except AssertionError as e:
        print(f"\nTest failed: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
