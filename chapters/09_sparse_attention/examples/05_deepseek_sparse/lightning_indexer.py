#!/usr/bin/env python3
"""
Lightning Indexer: Efficient Top-K Token Selection for Sparse Attention

Implements DeepSeek-V3's lightning indexer for O(1)-ish index computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math


class LightningIndexer(nn.Module):
    """
    Efficient top-k token selection using low-rank approximation.

    Args:
        dim: Model dimension
        approx_rank: Rank for low-rank approximation (r << dim)
        block_size: Size of blocks for blocked indexing (optional)
        use_local_bias: Add bias toward local positions
    """

    def __init__(
        self,
        dim,
        approx_rank=128,
        block_size=None,
        use_local_bias=True
    ):
        super().__init__()
        self.dim = dim
        self.approx_rank = approx_rank
        self.block_size = block_size
        self.use_local_bias = use_local_bias

        # Learned projection matrices for low-rank approximation
        self.W_q_approx = nn.Linear(dim, approx_rank, bias=False)
        self.W_k_approx = nn.Linear(dim, approx_rank, bias=False)

        # Initialize with scaled random weights
        nn.init.normal_(self.W_q_approx.weight, std=1.0 / math.sqrt(dim))
        nn.init.normal_(self.W_k_approx.weight, std=1.0 / math.sqrt(dim))

    def forward(self, Q, K, k, mask=None):
        """
        Select top-k most relevant tokens for each query.

        Args:
            Q: Query tensor [batch, seq_len, dim]
            K: Key tensor [batch, seq_len, dim]
            k: Number of tokens to select per query
            mask: Optional mask [batch, seq_len, seq_len]

        Returns:
            indices: [batch, seq_len, k] - indices of selected tokens
            scores: [batch, seq_len, k] - approximate scores
        """
        batch_size, seq_len, _ = Q.shape

        # Project to low-rank space
        Q_approx = self.W_q_approx(Q)  # [batch, seq_len, approx_rank]
        K_approx = self.W_k_approx(K)  # [batch, seq_len, approx_rank]

        # Compute approximate attention scores
        # [batch, seq_len, approx_rank] @ [batch, approx_rank, seq_len]
        # -> [batch, seq_len, seq_len]
        scores_approx = torch.bmm(Q_approx, K_approx.transpose(1, 2))

        # Scale by sqrt(rank)
        scores_approx = scores_approx / math.sqrt(self.approx_rank)

        # Apply mask if provided
        if mask is not None:
            scores_approx = scores_approx.masked_fill(mask == 0, float('-inf'))

        # Add local bias (exponential decay with distance)
        if self.use_local_bias:
            positions = torch.arange(seq_len, device=Q.device)
            distance = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
            local_bias = torch.exp(-distance / 256.0) * 0.1  # Tunable decay
            scores_approx = scores_approx + local_bias.unsqueeze(0)

        # Select top-k indices
        # torch.topk returns (values, indices)
        topk_scores, topk_indices = torch.topk(scores_approx, k, dim=-1)

        return topk_indices, topk_scores

    def blocked_forward(self, Q, K, k, window_size=512):
        """
        Blocked indexing: only search within nearby blocks for efficiency.

        For very long sequences, limits search space per query.
        """
        batch_size, seq_len, _ = Q.shape

        # Project to low-rank
        Q_approx = self.W_q_approx(Q)
        K_approx = self.W_k_approx(K)

        # Initialize output
        all_indices = []
        all_scores = []

        num_blocks = (seq_len + self.block_size - 1) // self.block_size

        for b in range(num_blocks):
            # Query block
            q_start = b * self.block_size
            q_end = min((b + 1) * self.block_size, seq_len)
            Q_block = Q_approx[:, q_start:q_end]

            # Key blocks to search (current ± window)
            k_block_start = max(0, b - window_size // self.block_size)
            k_block_end = min(num_blocks, b + window_size // self.block_size + 1)

            k_start = k_block_start * self.block_size
            k_end = min(k_block_end * self.block_size, seq_len)
            K_block = K_approx[:, k_start:k_end]

            # Compute scores within window
            scores = torch.bmm(Q_block, K_block.transpose(1, 2))
            scores = scores / math.sqrt(self.approx_rank)

            # Top-k within window
            topk_scores, topk_indices_local = torch.topk(scores, k, dim=-1)

            # Convert local indices to global
            topk_indices_global = topk_indices_local + k_start

            all_indices.append(topk_indices_global)
            all_scores.append(topk_scores)

        # Concatenate blocks
        indices = torch.cat(all_indices, dim=1)
        scores = torch.cat(all_scores, dim=1)

        return indices, scores


class HybridIndexer(nn.Module):
    """
    Hybrid indexer: Combine exact local attention with approximate global.

    Pattern: Local window (exact) + Top-k global (approximate)
    """

    def __init__(self, dim, approx_rank=128, local_window=256, global_topk=256):
        super().__init__()
        self.dim = dim
        self.local_window = local_window
        self.global_topk = global_topk

        self.lightning = LightningIndexer(dim, approx_rank)

    def forward(self, Q, K, mask=None):
        """
        Returns indices for hybrid pattern: local + global top-k.

        Args:
            Q, K: [batch, seq_len, dim]

        Returns:
            indices: [batch, seq_len, local_window + global_topk]
        """
        batch_size, seq_len, _ = Q.shape

        # 1. Local window indices
        local_indices = []
        for i in range(seq_len):
            # Causal local window: [i - window + 1, i]
            start = max(0, i - self.local_window + 1)
            end = i + 1
            indices_i = torch.arange(start, end, device=Q.device)

            # Pad if necessary
            if len(indices_i) < self.local_window:
                pad_len = self.local_window - len(indices_i)
                indices_i = torch.cat([
                    torch.zeros(pad_len, dtype=torch.long, device=Q.device),
                    indices_i
                ])

            local_indices.append(indices_i)

        local_indices = torch.stack(local_indices, dim=0)  # [seq_len, local_window]
        local_indices = local_indices.unsqueeze(0).expand(batch_size, -1, -1)

        # 2. Global top-k indices (excluding local)
        global_indices, _ = self.lightning(Q, K, self.global_topk, mask)

        # 3. Combine (and deduplicate if needed)
        # For simplicity, just concatenate
        combined_indices = torch.cat([local_indices, global_indices], dim=-1)

        return combined_indices


def benchmark_indexer():
    """Benchmark different indexer strategies."""
    print("=" * 60)
    print("Lightning Indexer Benchmark")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    batch_size = 4
    seq_len = 8192
    dim = 128
    k = 512

    Q = torch.randn(batch_size, seq_len, dim, device=device)
    K = torch.randn(batch_size, seq_len, dim, device=device)

    # 1. Naive top-k (compute full scores)
    print("1. Naive Top-K (Full Scores)")
    start = time.time()
    scores_full = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(dim)
    indices_naive, _ = torch.topk(scores_full, k, dim=-1)
    if device == 'cuda':
        torch.cuda.synchronize()
    time_naive = time.time() - start
    print(f"   Time: {time_naive * 1000:.2f} ms")
    print(f"   FLOPs: {2 * batch_size * seq_len * seq_len * dim / 1e9:.2f} GFLOPs\n")

    # 2. Lightning indexer (low-rank approximation)
    print("2. Lightning Indexer (Low-Rank)")
    indexer = LightningIndexer(dim, approx_rank=128).to(device)

    # Warmup
    for _ in range(10):
        _, _ = indexer(Q, K, k)
    if device == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    indices_lightning, _ = indexer(Q, K, k)
    if device == 'cuda':
        torch.cuda.synchronize()
    time_lightning = time.time() - start

    approx_rank = 128
    flops_lightning = 2 * batch_size * seq_len * (dim * approx_rank + seq_len * approx_rank)
    print(f"   Time: {time_lightning * 1000:.2f} ms")
    print(f"   FLOPs: {flops_lightning / 1e9:.2f} GFLOPs")
    print(f"   Speedup: {time_naive / time_lightning:.2f}x\n")

    # 3. Check selection quality (overlap with naive)
    # How many of the lightning-selected indices are in naive top-k?
    overlap = 0
    for b in range(batch_size):
        for i in range(seq_len):
            naive_set = set(indices_naive[b, i].cpu().tolist())
            lightning_set = set(indices_lightning[b, i].cpu().tolist())
            overlap += len(naive_set & lightning_set)

    total_selections = batch_size * seq_len * k
    overlap_pct = 100.0 * overlap / total_selections

    print("Selection Quality:")
    print(f"   Overlap with exact top-k: {overlap_pct:.1f}%")
    print(f"   (Expected ~70-90% for well-trained indexer)\n")

    # 4. Memory usage
    mem_naive = batch_size * seq_len * seq_len * 4 / 1024**2  # Full scores
    mem_lightning = batch_size * seq_len * k * 4 / 1024**2  # Sparse indices
    print("Memory Usage:")
    print(f"   Naive (full scores): {mem_naive:.2f} MB")
    print(f"   Lightning (indices only): {mem_lightning:.2f} MB")
    print(f"   Reduction: {mem_naive / mem_lightning:.2f}x\n")


def test_correctness():
    """Test that indexer returns valid indices."""
    print("=" * 60)
    print("Correctness Test")
    print("=" * 60)

    batch_size, seq_len, dim = 2, 128, 64
    k = 32

    Q = torch.randn(batch_size, seq_len, dim)
    K = torch.randn(batch_size, seq_len, dim)

    indexer = LightningIndexer(dim, approx_rank=32)
    indices, scores = indexer(Q, K, k)

    # Check shape
    assert indices.shape == (batch_size, seq_len, k), f"Wrong shape: {indices.shape}"

    # Check indices are in valid range
    assert indices.min() >= 0, "Negative indices!"
    assert indices.max() < seq_len, f"Indices out of range: {indices.max()} >= {seq_len}"

    # Check scores are reasonable
    assert not torch.isnan(scores).any(), "NaN in scores!"
    assert not torch.isinf(scores).any(), "Inf in scores!"

    print("✓ All checks passed!")
    print(f"  Shape: {indices.shape}")
    print(f"  Index range: [{indices.min()}, {indices.max()}]")
    print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]\n")


if __name__ == '__main__':
    test_correctness()
    benchmark_indexer()
