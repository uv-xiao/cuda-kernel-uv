"""
Test Suite for Exercise 01: Fused ReLU Matrix Multiplication
"""

import torch
import sys

# Import student's solution
try:
    from solution import triton_matmul_relu, pytorch_matmul_relu
except ImportError:
    print("Error: Could not import solution. Make sure solution.py is complete.")
    sys.exit(1)


def test_basic_correctness():
    """Test basic correctness."""
    print("Test 1: Basic Correctness")
    
    M, N, K = 64, 64, 64
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    
    result = triton_matmul_relu(a, b)
    expected = pytorch_matmul_relu(a, b)
    
    assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)
    print("  ✓ PASS")


def test_all_negative():
    """Test with all negative inputs (should output zeros)."""
    print("Test 2: All Negative Inputs")
    
    M, N, K = 32, 32, 32
    a = -torch.rand(M, K, device='cuda', dtype=torch.float16)
    b = -torch.rand(K, N, device='cuda', dtype=torch.float16)
    
    result = triton_matmul_relu(a, b)
    
    # All outputs should be zero (ReLU clamps negatives)
    assert (result >= 0).all()
    print("  ✓ PASS")


def test_all_positive():
    """Test with all positive inputs."""
    print("Test 3: All Positive Inputs")
    
    M, N, K = 32, 32, 32
    a = torch.rand(M, K, device='cuda', dtype=torch.float16)
    b = torch.rand(K, N, device='cuda', dtype=torch.float16)
    
    result = triton_matmul_relu(a, b)
    expected = pytorch_matmul_relu(a, b)
    
    assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)
    print("  ✓ PASS")


def test_large_matrix():
    """Test with larger matrices."""
    print("Test 4: Large Matrix")
    
    M, N, K = 1024, 1024, 1024
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    
    result = triton_matmul_relu(a, b)
    expected = pytorch_matmul_relu(a, b)
    
    assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)
    print("  ✓ PASS")


def test_non_square():
    """Test with non-square matrices."""
    print("Test 5: Non-Square Matrices")
    
    M, N, K = 128, 256, 192
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    
    result = triton_matmul_relu(a, b)
    expected = pytorch_matmul_relu(a, b)
    
    assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)
    print("  ✓ PASS")


def run_all_tests():
    """Run all tests."""
    print("Running Test Suite")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Tests require a GPU.")
        return False
    
    try:
        test_basic_correctness()
        test_all_negative()
        test_all_positive()
        test_large_matrix()
        test_non_square()
        
        print("\n" + "=" * 70)
        print("All tests passed! ✓")
        print("=" * 70)
        return True
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
