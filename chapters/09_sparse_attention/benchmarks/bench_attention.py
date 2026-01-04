#!/usr/bin/env python3
"""
Comprehensive Attention Benchmark

Compares naive, FlashAttention, and sparse attention across different configurations.
"""

import torch
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def benchmark_attention_variants(seq_lengths, batch_size=4, num_heads=8, head_dim=64):
    """
    Benchmark different attention implementations across sequence lengths.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running benchmarks on {device}")
    print(f"Config: batch={batch_size}, heads={num_heads}, head_dim={head_dim}\n")

    results = {
        'seq_lengths': seq_lengths,
        'pytorch_sdpa': [],
        'memory_pytorch': [],
    }

    for seq_len in seq_lengths:
        print(f"Sequence length: {seq_len}")

        # Create inputs
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

        # PyTorch SDPA
        for _ in range(10):
            _ = F.scaled_dot_product_attention(q, k, v)
        if device == 'cuda':
            torch.cuda.synchronize()

        start = time.time()
        for _ in range(20):
            output = F.scaled_dot_product_attention(q, k, v)
        if device == 'cuda':
            torch.cuda.synchronize()
        time_sdpa = (time.time() - start) / 20 * 1000

        results['pytorch_sdpa'].append(time_sdpa)

        # Memory (attention matrix)
        mem = batch_size * num_heads * seq_len * seq_len * 4 / 1024**2
        results['memory_pytorch'].append(mem)

        print(f"  PyTorch SDPA: {time_sdpa:.2f} ms, Memory: {mem:.1f} MB\n")

    return results


def plot_results(results, output_dir='./'):
    """Plot benchmark results."""
    seq_lengths = results['seq_lengths']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Time comparison
    ax1.plot(seq_lengths, results['pytorch_sdpa'], 'o-', label='PyTorch SDPA', linewidth=2)
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Attention Time vs Sequence Length')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Memory comparison
    ax2.plot(seq_lengths, results['memory_pytorch'], 'o-', label='Dense', linewidth=2, color='C0')

    # Theoretical sparse memory (k=512)
    k = 512
    sparse_mem = [s * k * 4 / 1024 for s in seq_lengths]
    ax2.plot(seq_lengths, sparse_mem, 's--', label=f'Sparse (k={k})', linewidth=2, color='C1')

    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Memory (MB)')
    ax2.set_title('Attention Matrix Memory')
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'attention_benchmark.png', dpi=150)
    print(f"Saved plot to {output_dir}/attention_benchmark.png")


def complexity_analysis():
    """Analyze and compare computational complexity."""
    print("=" * 70)
    print("Complexity Analysis: O(L²) vs O(Lk)")
    print("=" * 70)

    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    d = 128
    k = 512

    print(f"Head dimension d = {d}")
    print(f"Sparse parameter k = {k}\n")

    print(f"{'Seq Len':>8} | {'Dense FLOPs':>12} | {'Sparse FLOPs':>12} | {'Speedup':>8}")
    print("-" * 70)

    for L in seq_lengths:
        dense_flops = 4 * L * L * d  # QK^T + softmax + PV
        sparse_flops = 4 * L * k * d

        speedup = dense_flops / sparse_flops

        print(f"{L:8d} | {dense_flops/1e9:10.2f} G | {sparse_flops/1e9:10.2f} G | {speedup:7.1f}x")

    print()


if __name__ == '__main__':
    # Complexity analysis
    complexity_analysis()

    # Benchmarks
    print("=" * 70)
    print("Running Benchmarks")
    print("=" * 70)

    seq_lengths = [256, 512, 1024, 2048, 4096]

    results = benchmark_attention_variants(seq_lengths)

    # Plot results
    plot_results(results, output_dir='.')

    print("\nBenchmark complete!")
