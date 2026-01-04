#!/usr/bin/env python3
"""Test custom sparse attention implementation"""

import torch
import matplotlib.pyplot as plt


def test_mask_properties(mask, pattern_name):
    """Test that mask has expected properties."""
    print(f"\nTesting {pattern_name} mask properties...")

    # Check it's boolean
    assert mask.dtype == torch.bool, "Mask should be boolean"

    # Check it's square
    assert mask.shape[0] == mask.shape[1], "Mask should be square"

    # Check it's not all zeros or all ones
    assert 0 < mask.sum() < mask.numel(), "Mask should be partially sparse"

    # Check causal property (if applicable)
    # For document QA, question tokens can attend to future
    # But document tokens should be causal or local

    print("✓ All property checks passed!")


def compare_with_dense(sparse_attn_fn, Q, K, V):
    """Compare sparse attention output with dense (for correctness)."""
    import torch.nn.functional as F

    print("\nComparing with dense attention...")

    # Dense attention
    Q_multi = Q.unsqueeze(1)
    K_multi = K.unsqueeze(1)
    V_multi = V.unsqueeze(1)

    output_dense = F.scaled_dot_product_attention(Q_multi, K_multi, V_multi)
    output_dense = output_dense.squeeze(1)

    # Sparse attention
    output_sparse = sparse_attn_fn(Q, K, V)

    # They won't match exactly (different patterns), but check validity
    assert output_sparse.shape == output_dense.shape, "Shape mismatch!"
    assert not torch.isnan(output_sparse).any(), "NaN in sparse output!"
    assert not torch.isinf(output_sparse).any(), "Inf in sparse output!"

    print("✓ Sparse attention produces valid output")
    print(f"  Dense mean: {output_dense.mean():.6f}")
    print(f"  Sparse mean: {output_sparse.mean():.6f}")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing Custom Sparse Attention")
    print("=" * 60)

    # Test with solution if available
    try:
        from solution import DocumentQASparseAttention

        seq_len = 512
        pattern = DocumentQASparseAttention(num_question_tokens=32, local_window=128)

        mask = pattern.create_mask(seq_len)
        test_mask_properties(mask, "Document QA")

        # Test attention computation
        batch_size = 2
        dim = 64

        Q = torch.randn(batch_size, seq_len, dim)
        K = torch.randn(batch_size, seq_len, dim)
        V = torch.randn(batch_size, seq_len, dim)

        compare_with_dense(pattern.compute_attention, Q, K, V)

        print("\n✓ All tests passed!")

    except ImportError:
        print("Solution not found. Implement your pattern first!")
