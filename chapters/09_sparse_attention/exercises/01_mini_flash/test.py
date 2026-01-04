#!/usr/bin/env python3
"""Test Mini FlashAttention implementation"""

import torch
import torch.nn.functional as F
import subprocess
import numpy as np

def test_correctness():
    """Test that implementation matches PyTorch."""
    print("Testing Mini FlashAttention correctness...")

    batch_size = 1
    seq_len = 256
    head_dim = 64

    torch.manual_seed(42)
    Q = torch.randn(batch_size, seq_len, head_dim)
    K = torch.randn(batch_size, seq_len, head_dim)
    V = torch.randn(batch_size, seq_len, head_dim)

    # PyTorch reference
    Q_multi = Q.unsqueeze(1)  # Add head dimension
    K_multi = K.unsqueeze(1)
    V_multi = V.unsqueeze(1)

    expected = F.scaled_dot_product_attention(Q_multi, K_multi, V_multi)
    expected = expected.squeeze(1)

    print(f"Expected output shape: {expected.shape}")
    print(f"Expected mean: {expected.mean():.6f}, std: {expected.std():.6f}")

    # TODO: Run your CUDA implementation and compare
    print("\nRun your CUDA implementation and compare with expected output")
    print("Max allowed error: 1e-5")

if __name__ == '__main__':
    test_correctness()
