#!/usr/bin/env python3
"""
Token Selection Strategies for Sparse Attention

Compares different approaches to selecting which tokens to attend to.
"""

import torch
import torch.nn.functional as F
import time
import math


def exact_topk(scores, k):
    """Standard exact top-k selection."""
    return torch.topk(scores, k, dim=-1)


def threshold_selection(scores, threshold):
    """Select all tokens above a threshold (variable sparsity)."""
    mask = scores > threshold
    # For fixed output size, take top-k among those passing threshold
    return mask


def nucleus_selection(scores, p=0.9):
    """
    Nucleus (top-p) sampling approach.
    Select smallest set of tokens whose cumulative probability >= p.
    """
    probs = F.softmax(scores, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative_probs <= p

    # Include at least one token
    mask[:, :, 0] = True

    # Convert back to original order
    return sorted_indices, mask


def local_plus_topk(seq_len, local_window, k_global):
    """
    Hybrid: local window + global top-k.
    Returns indices pattern [seq_len, local_window + k_global].
    """
    device = 'cpu'
    all_indices = []

    for i in range(seq_len):
        # Local window (causal)
        local_start = max(0, i - local_window + 1)
        local_end = i + 1
        local_idx = torch.arange(local_start, local_end, device=device)

        # Placeholder for global (would be selected by scores)
        # In practice, this is combined with score-based selection
        all_indices.append(local_idx)

    return all_indices


def benchmark_selection_strategies():
    """Compare different selection strategies."""
    print("=" * 60)
    print("Token Selection Strategy Benchmark")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 4
    seq_len = 4096
    k = 512

    scores = torch.randn(batch_size, seq_len, seq_len, device=device)

    strategies = [
        ("Exact Top-K", lambda: torch.topk(scores, k, dim=-1)),
        ("Threshold (>0)", lambda: threshold_selection(scores, 0.0)),
    ]

    for name, fn in strategies:
        # Warmup
        for _ in range(5):
            _ = fn()
        if device == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(20):
            result = fn()
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.time() - start) / 20 * 1000

        print(f"\n{name}:")
        print(f"  Time: {elapsed:.2f} ms")


class AdaptiveTokenSelector(torch.nn.Module):
    """
    Adaptive selection: vary k based on query importance.

    Some queries need more context (higher k), others less.
    """

    def __init__(self, dim, base_k=256, max_k=512):
        super().__init__()
        self.base_k = base_k
        self.max_k = max_k

        # Learned importance predictor
        self.importance_net = torch.nn.Linear(dim, 1)

    def forward(self, Q, scores):
        """
        Args:
            Q: [batch, seq_len, dim]
            scores: [batch, seq_len, seq_len]

        Returns:
            indices: [batch, seq_len, k] where k varies per query
        """
        # Predict importance for each query
        importance = torch.sigmoid(self.importance_net(Q)).squeeze(-1)  # [batch, seq_len]

        # Scale k based on importance
        k_per_query = (self.base_k + (self.max_k - self.base_k) * importance).long()

        # For simplicity, use max_k and mask out extras
        # In practice, would use variable-length tensors or padding
        indices, values = torch.topk(scores, self.max_k, dim=-1)

        return indices, k_per_query


if __name__ == '__main__':
    benchmark_selection_strategies()

    # Test adaptive selector
    print("\n" + "=" * 60)
    print("Adaptive Token Selector Test")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch, seq_len, dim = 2, 128, 64

    Q = torch.randn(batch, seq_len, dim, device=device)
    scores = torch.randn(batch, seq_len, seq_len, device=device)

    selector = AdaptiveTokenSelector(dim, base_k=32, max_k=64).to(device)
    indices, k_values = selector(Q, scores)

    print(f"Indices shape: {indices.shape}")
    print(f"K values range: [{k_values.min()}, {k_values.max()}]")
    print(f"Mean k: {k_values.float().mean():.1f}")
