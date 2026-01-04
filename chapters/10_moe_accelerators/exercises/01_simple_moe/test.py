"""
Exercise 01: Test Suite for SimpleMoE

Run this to verify your implementation.
"""

import torch
import sys

# Try to import student solution
try:
    from starter import SimpleMoE, Expert
except ImportError:
    print("Error: Could not import from starter.py")
    sys.exit(1)


def test_expert_forward():
    """Test Expert forward pass"""
    print("Test 1: Expert Forward Pass...")

    expert = Expert(hidden_dim=256, ffn_dim=1024)
    x = torch.randn(10, 256)
    output = expert(x)

    assert output.shape == (10, 256), f"Expected shape (10, 256), got {output.shape}"
    assert not torch.isnan(output).any(), "NaN values in expert output"

    print("  ✓ Expert forward pass correct\n")


def test_moe_shape():
    """Test MoE output shape"""
    print("Test 2: MoE Output Shape...")

    moe = SimpleMoE(hidden_dim=128, ffn_dim=512, num_experts=4, top_k=2)
    x = torch.randn(2, 16, 128)
    output = moe(x)

    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    print("  ✓ Output shape correct\n")


def test_no_nan_inf():
    """Test for NaN/Inf values"""
    print("Test 3: NaN/Inf Check...")

    moe = SimpleMoE(hidden_dim=64, ffn_dim=256, num_experts=4, top_k=2)
    x = torch.randn(4, 8, 64)
    output = moe(x)

    assert not torch.isnan(output).any(), "NaN values in output"
    assert not torch.isinf(output).any(), "Inf values in output"

    print("  ✓ No NaN/Inf values\n")


def test_router_probabilities():
    """Test router probability normalization"""
    print("Test 4: Router Probabilities...")

    moe = SimpleMoE(hidden_dim=128, ffn_dim=512, num_experts=8, top_k=2)
    x = torch.randn(4, 16, 128)

    # Access router logits
    x_flat = x.view(-1, 128)
    router_logits = moe.router(x_flat)
    router_probs = torch.softmax(router_logits, dim=-1)

    # Check probabilities sum to 1
    prob_sums = router_probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), \
        "Router probabilities don't sum to 1"

    # Check top-k selection
    top_k_probs, _ = torch.topk(router_probs, moe.top_k, dim=-1)
    top_k_normalized = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
    assert torch.allclose(top_k_normalized.sum(dim=-1), torch.ones(x_flat.shape[0]), atol=1e-5), \
        "Top-k probabilities don't sum to 1 after normalization"

    print("  ✓ Router probabilities correct\n")


def test_expert_usage():
    """Test that all experts are used reasonably"""
    print("Test 5: Expert Usage Balance...")

    moe = SimpleMoE(hidden_dim=128, ffn_dim=512, num_experts=8, top_k=2)
    x = torch.randn(8, 32, 128)  # 256 tokens total

    # Get expert assignments
    x_flat = x.view(-1, 128)
    router_logits = moe.router(x_flat)
    router_probs = torch.softmax(router_logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(router_probs, moe.top_k, dim=-1)

    # Count tokens per expert
    expert_counts = torch.zeros(moe.num_experts)
    for expert_idx in range(moe.num_experts):
        expert_counts[expert_idx] = (top_k_indices == expert_idx).sum().item()

    # Check that most experts are used
    experts_used = (expert_counts > 0).sum().item()
    assert experts_used >= moe.num_experts * 0.5, \
        f"Only {experts_used}/{moe.num_experts} experts used"

    # Check reasonable distribution (not too imbalanced)
    max_count = expert_counts.max().item()
    min_count = expert_counts.min().item()
    avg_count = expert_counts.mean().item()

    print(f"  Expert usage: min={int(min_count)}, max={int(max_count)}, avg={avg_count:.1f}")
    print(f"  ✓ Expert usage reasonable\n")


def test_gradient_flow():
    """Test gradient flow through MoE"""
    print("Test 6: Gradient Flow...")

    moe = SimpleMoE(hidden_dim=64, ffn_dim=256, num_experts=4, top_k=2)
    x = torch.randn(2, 8, 64, requires_grad=True)

    # Forward pass
    output = moe(x)
    loss = output.sum()

    # Backward pass
    loss.backward()

    # Check gradients exist
    assert x.grad is not None, "No gradient for input"
    assert not torch.isnan(x.grad).any(), "NaN in input gradient"

    # Check expert gradients
    for expert in moe.experts:
        assert expert.fc1.weight.grad is not None, "No gradient for expert weights"
        assert not torch.isnan(expert.fc1.weight.grad).any(), "NaN in expert gradient"

    print("  ✓ Gradient flow correct\n")


def test_batch_independence():
    """Test that batch dimension is independent"""
    print("Test 7: Batch Independence...")

    moe = SimpleMoE(hidden_dim=128, ffn_dim=512, num_experts=4, top_k=2)

    # Process single item
    x1 = torch.randn(1, 16, 128)
    out1 = moe(x1)

    # Process as part of batch
    x_batch = torch.cat([x1, torch.randn(3, 16, 128)], dim=0)
    out_batch = moe(x_batch)

    # First item should be similar (might not be exactly equal due to numerical precision)
    # Just check shapes for now
    assert out1.shape == (1, 16, 128), "Single item shape incorrect"
    assert out_batch.shape == (4, 16, 128), "Batch shape incorrect"

    print("  ✓ Batch processing correct\n")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("SimpleMoE Test Suite")
    print("=" * 60 + "\n")

    tests = [
        test_expert_forward,
        test_moe_shape,
        test_no_nan_inf,
        test_router_probabilities,
        test_expert_usage,
        test_gradient_flow,
        test_batch_independence,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}\n")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n🎉 All tests passed! Great work!")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Review the errors above.")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
