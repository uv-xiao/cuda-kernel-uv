#!/usr/bin/env python3
"""
Solution: Document QA Sparse Attention Pattern
"""

import torch
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt


class DocumentQASparseAttention:
    """
    Sparse pattern for document question answering.

    Pattern:
    - Question tokens (first K) attend to ALL tokens
    - Document tokens attend to:
      - All question tokens
      - Local window around self
    """

    def __init__(self, num_question_tokens=64, local_window=256):
        self.num_question = num_question_tokens
        self.local_window = local_window

    def create_mask(self, seq_len):
        """Create sparse mask for document QA."""
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)

        # Question tokens attend to ALL
        mask[:self.num_question, :] = True

        # All tokens attend to question tokens
        mask[:, :self.num_question] = True

        # Document tokens: local window
        for i in range(self.num_question, seq_len):
            start = max(self.num_question, i - self.local_window // 2)
            end = min(seq_len, i + self.local_window // 2 + 1)
            mask[i, start:end] = True

        return mask

    def compute_attention(self, Q, K, V, mask=None):
        """
        Compute sparse attention.
        For simplicity, use masked dense attention.
        """
        batch_size, seq_len, dim = Q.shape

        if mask is None:
            mask = self.create_mask(seq_len).to(Q.device)

        # Compute scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (dim ** 0.5)

        # Apply mask
        scores = scores.masked_fill(~mask.unsqueeze(0), float('-inf'))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Handle NaN from all -inf
        attn_weights = torch.nan_to_num(attn_weights, 0.0)

        # Apply to values
        output = torch.matmul(attn_weights, V)

        return output


def visualize_pattern(mask, title='Document QA Sparse Pattern'):
    """Visualize the pattern."""
    plt.figure(figsize=(12, 10))
    plt.imshow(mask.float().cpu().numpy(), cmap='RdYlGn', aspect='auto')
    plt.title(title)
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')

    # Mark question region
    num_q = 64
    plt.axhline(y=num_q, color='blue', linestyle='--', linewidth=2, label='Question End')
    plt.axvline(x=num_q, color='blue', linestyle='--', linewidth=2)

    plt.colorbar(label='Attend (green) / Masked (red)')
    plt.legend()
    plt.savefig('document_qa_pattern.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to document_qa_pattern.png")


def analyze_sparsity(mask):
    """Analyze sparsity."""
    total = mask.numel()
    active = mask.sum().item()
    sparsity = 100 * (1 - active / total)

    print("Sparsity Analysis:")
    print(f"  Total pairs: {total:,}")
    print(f"  Active pairs: {active:,}")
    print(f"  Sparsity: {sparsity:.1f}%")

    active_per_row = mask.sum(dim=1).float()
    print(f"\nPer-query statistics:")
    print(f"  Question tokens attend to: {active_per_row[:64].mean():.0f} tokens")
    print(f"  Document tokens attend to: {active_per_row[64:].mean():.0f} tokens")


def benchmark_performance():
    """Benchmark sparse vs dense."""
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 4
    seq_len = 4096
    dim = 128

    Q = torch.randn(batch_size, seq_len, dim, device=device)
    K = torch.randn(batch_size, seq_len, dim, device=device)
    V = torch.randn(batch_size, seq_len, dim, device=device)

    # Dense attention
    print("\n1. Dense Attention")
    Q_multi = Q.unsqueeze(1)
    K_multi = K.unsqueeze(1)
    V_multi = V.unsqueeze(1)

    # Warmup
    for _ in range(5):
        _ = F.scaled_dot_product_attention(Q_multi, K_multi, V_multi)
    if device == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(20):
        output_dense = F.scaled_dot_product_attention(Q_multi, K_multi, V_multi)
    if device == 'cuda':
        torch.cuda.synchronize()
    time_dense = (time.time() - start) / 20 * 1000

    print(f"   Time: {time_dense:.2f} ms")

    # Sparse attention
    print("\n2. Sparse Attention (Document QA)")
    sparse_attn = DocumentQASparseAttention(num_question_tokens=64, local_window=256)

    for _ in range(5):
        _ = sparse_attn.compute_attention(Q, K, V)
    if device == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(20):
        output_sparse = sparse_attn.compute_attention(Q, K, V)
    if device == 'cuda':
        torch.cuda.synchronize()
    time_sparse = (time.time() - start) / 20 * 1000

    print(f"   Time: {time_sparse:.2f} ms")
    print(f"   Speedup: {time_dense / time_sparse:.2f}x")

    # Note: This uses masked dense attention, not truly optimized sparse
    print("\nNote: Using masked dense attention. True sparse implementation")
    print("      with index-based gather would be much faster!")


if __name__ == '__main__':
    print("=" * 60)
    print("Document QA Sparse Attention - Solution")
    print("=" * 60)

    seq_len = 2048
    pattern = DocumentQASparseAttention(num_question_tokens=64, local_window=256)

    # Create and visualize
    mask = pattern.create_mask(seq_len)
    visualize_pattern(mask)
    analyze_sparsity(mask)

    # Benchmark
    benchmark_performance()
