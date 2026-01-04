#!/usr/bin/env python3
"""
DeepSeek Sparse Attention - Full Implementation

Combines lightning indexer with efficient sparse attention computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math

from lightning_indexer import LightningIndexer


class DeepSeekSparseAttention(nn.Module):
    """
    DeepSeek-style sparse attention with dynamic token selection.

    Pipeline:
    1. Lightning Indexer: Select top-k relevant tokens
    2. Gather: Extract selected K, V
    3. Sparse Attention: Compute attention only on selected tokens
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        approx_rank=128,
        topk=512,
        dropout=0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.topk = topk
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Lightning indexer for token selection
        self.indexer = LightningIndexer(dim, approx_rank)

        # Standard attention projections
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, dim]
            mask: Optional [batch, seq_len, seq_len]

        Returns:
            output: [batch, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # [batch, seq_len, 3*dim]
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Now: [batch, num_heads, seq_len, head_dim]

        # For indexer, use mean across heads (or use full q, k)
        q_mean = q.mean(dim=1)  # [batch, seq_len, head_dim]
        k_mean = k.mean(dim=1)

        # Step 1: Lightning indexer - select top-k tokens
        indices, _ = self.indexer(q_mean, k_mean, self.topk, mask)
        # indices: [batch, seq_len, topk]

        # Step 2: Gather selected K and V
        # Expand indices for multi-head
        indices_expanded = indices.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        # [batch, num_heads, seq_len, topk]

        indices_for_gather = indices_expanded.unsqueeze(-1).expand(
            -1, -1, -1, -1, self.head_dim
        )
        # [batch, num_heads, seq_len, topk, head_dim]

        k_expanded = k.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
        # [batch, num_heads, seq_len, seq_len, head_dim]

        k_selected = torch.gather(k_expanded, 3, indices_for_gather)
        # [batch, num_heads, seq_len, topk, head_dim]

        v_expanded = v.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
        v_selected = torch.gather(v_expanded, 3, indices_for_gather)

        # Step 3: Compute sparse attention scores
        # q: [batch, num_heads, seq_len, head_dim]
        # k_selected: [batch, num_heads, seq_len, topk, head_dim]

        q_expanded = q.unsqueeze(3)  # [batch, num_heads, seq_len, 1, head_dim]
        scores = (q_expanded * k_selected).sum(dim=-1)  # [batch, num_heads, seq_len, topk]
        scores = scores * self.scale

        # Softmax over selected tokens
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Step 4: Apply attention to values
        attn_weights_expanded = attn_weights.unsqueeze(-1)
        # [batch, num_heads, seq_len, topk, 1]

        output = (attn_weights_expanded * v_selected).sum(dim=3)
        # [batch, num_heads, seq_len, head_dim]

        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.dim)
        output = self.out_proj(output)

        return output


def benchmark_sparse_vs_dense():
    """Compare sparse attention with dense attention."""
    print("=" * 70)
    print("DeepSeek Sparse Attention Benchmark")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    batch_size = 4
    seq_len = 4096
    dim = 512
    num_heads = 8
    topk = 512

    x = torch.randn(batch_size, seq_len, dim, device=device)

    # Dense attention (PyTorch built-in)
    print("1. Dense Attention (PyTorch SDPA)")
    q = torch.randn(batch_size, num_heads, seq_len, dim // num_heads, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, dim // num_heads, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, dim // num_heads, device=device)

    # Warmup
    for _ in range(10):
        _ = F.scaled_dot_product_attention(q, k, v)
    if device == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    num_iters = 50
    for _ in range(num_iters):
        output_dense = F.scaled_dot_product_attention(q, k, v)
    if device == 'cuda':
        torch.cuda.synchronize()
    time_dense = (time.time() - start) / num_iters * 1000

    flops_dense = 4 * batch_size * num_heads * seq_len * seq_len * (dim // num_heads)
    print(f"   Time: {time_dense:.2f} ms")
    print(f"   FLOPs: {flops_dense / 1e9:.2f} GFLOPs\n")

    # Sparse attention
    print(f"2. Sparse Attention (k={topk})")
    sparse_attn = DeepSeekSparseAttention(
        dim=dim,
        num_heads=num_heads,
        approx_rank=128,
        topk=topk
    ).to(device)

    # Warmup
    for _ in range(10):
        _ = sparse_attn(x)
    if device == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_iters):
        output_sparse = sparse_attn(x)
    if device == 'cuda':
        torch.cuda.synchronize()
    time_sparse = (time.time() - start) / num_iters * 1000

    flops_sparse = 4 * batch_size * num_heads * seq_len * topk * (dim // num_heads)
    print(f"   Time: {time_sparse:.2f} ms")
    print(f"   FLOPs: {flops_sparse / 1e9:.2f} GFLOPs")
    print(f"   Speedup: {time_dense / time_sparse:.2f}x")
    print(f"   FLOP reduction: {flops_dense / flops_sparse:.2f}x\n")

    # Memory comparison
    mem_dense = batch_size * num_heads * seq_len * seq_len * 4 / 1024**2
    mem_sparse = batch_size * num_heads * seq_len * topk * 4 / 1024**2
    print("Memory Usage:")
    print(f"   Dense attention matrix: {mem_dense:.2f} MB")
    print(f"   Sparse attention matrix: {mem_sparse:.2f} MB")
    print(f"   Reduction: {mem_dense / mem_sparse:.2f}x\n")


def test_correctness():
    """Test sparse attention produces reasonable outputs."""
    print("=" * 70)
    print("Correctness Test")
    print("=" * 70)

    batch_size, seq_len, dim = 2, 128, 256

    x = torch.randn(batch_size, seq_len, dim)

    model = DeepSeekSparseAttention(dim=dim, num_heads=8, topk=64)
    output = model(x)

    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    assert not torch.isnan(output).any(), "NaN in output!"
    assert not torch.isinf(output).any(), "Inf in output!"

    print("✓ All checks passed!")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"  Output mean: {output.mean():.3f}, std: {output.std():.3f}\n")


if __name__ == '__main__':
    test_correctness()
    benchmark_sparse_vs_dense()
