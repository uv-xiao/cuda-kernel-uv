#!/usr/bin/env python3
"""
Memory Usage Analysis for Attention Mechanisms

Analyzes actual GPU memory usage for different attention implementations.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def measure_memory_usage(batch_size, seq_len, dim, num_heads):
    """
    Measure actual GPU memory usage for attention.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory measurement")
        return None

    device = 'cuda'
    head_dim = dim // num_heads

    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Allocate inputs
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    # Measure before attention
    mem_before = torch.cuda.memory_allocated()

    # Run attention
    output = F.scaled_dot_product_attention(q, k, v)

    # Measure peak
    mem_peak = torch.cuda.max_memory_allocated()

    mem_used = (mem_peak - mem_before) / 1024**2  # MB

    # Cleanup
    del q, k, v, output
    torch.cuda.empty_cache()

    return mem_used


def analyze_memory_scaling():
    """Analyze how memory scales with sequence length."""
    print("=" * 70)
    print("Memory Scaling Analysis")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    batch_size = 4
    num_heads = 8
    dim = 512

    seq_lengths = [256, 512, 1024, 2048]
    measured_memory = []
    theoretical_attn_memory = []

    for seq_len in seq_lengths:
        mem = measure_memory_usage(batch_size, seq_len, dim, num_heads)
        measured_memory.append(mem)

        # Theoretical: attention matrix size
        attn_mem = batch_size * num_heads * seq_len * seq_len * 4 / 1024**2
        theoretical_attn_memory.append(attn_mem)

        print(f"Seq len {seq_len:4d}: Measured {mem:6.1f} MB, "
              f"Theoretical attention matrix {attn_mem:6.1f} MB")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, measured_memory, 'o-', label='Measured Peak Usage', linewidth=2)
    plt.plot(seq_lengths, theoretical_attn_memory, 's--',
             label='Theoretical Attention Matrix', linewidth=2)

    plt.xlabel('Sequence Length')
    plt.ylabel('Memory (MB)')
    plt.title('GPU Memory Usage vs Sequence Length')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('memory_scaling.png', dpi=150)
    print("\nSaved plot to memory_scaling.png")


def compare_attention_memory():
    """Compare memory for dense vs sparse attention."""
    print("\n" + "=" * 70)
    print("Dense vs Sparse Memory Comparison")
    print("=" * 70)

    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    batch_size = 4
    num_heads = 8
    k_sparse = 512

    print(f"\nBatch size: {batch_size}, Num heads: {num_heads}")
    print(f"Sparse k: {k_sparse}\n")

    print(f"{'Seq Len':>8} | {'Dense (MB)':>12} | {'Sparse (MB)':>12} | {'Reduction':>10}")
    print("-" * 70)

    for L in seq_lengths:
        # Dense attention matrix
        dense_mem = batch_size * num_heads * L * L * 4 / 1024**2

        # Sparse attention matrix
        sparse_mem = batch_size * num_heads * L * k_sparse * 4 / 1024**2

        reduction = dense_mem / sparse_mem

        print(f"{L:8d} | {dense_mem:10.1f} | {sparse_mem:10.1f} | {reduction:9.1f}x")


def kv_cache_analysis():
    """Analyze KV cache memory for different approaches."""
    print("\n" + "=" * 70)
    print("KV Cache Memory Analysis")
    print("=" * 70)

    seq_len = 32768
    batch_size = 1  # Typical for inference
    num_heads = 32
    head_dim = 128
    num_layers = 40

    # Standard KV cache
    kv_standard = 2 * batch_size * num_heads * seq_len * head_dim * 4 / 1024**2
    kv_standard_total = kv_standard * num_layers

    # MLA (DeepSeek): compressed latent
    latent_dim = 512
    kv_mla = 2 * batch_size * seq_len * latent_dim * 4 / 1024**2
    kv_mla_total = kv_mla * num_layers

    print(f"Sequence length: {seq_len}")
    print(f"Batch size: {batch_size}")
    print(f"Num heads: {num_heads}, Head dim: {head_dim}")
    print(f"Num layers: {num_layers}")
    print(f"MLA latent dim: {latent_dim}\n")

    print("Per-layer KV cache:")
    print(f"  Standard: {kv_standard:.1f} MB")
    print(f"  MLA: {kv_mla:.1f} MB")
    print(f"  Reduction: {kv_standard / kv_mla:.1f}x\n")

    print("Total KV cache (all layers):")
    print(f"  Standard: {kv_standard_total:.1f} MB ({kv_standard_total/1024:.2f} GB)")
    print(f"  MLA: {kv_mla_total:.1f} MB ({kv_mla_total/1024:.2f} GB)")
    print(f"  Reduction: {kv_standard_total / kv_mla_total:.1f}x")


if __name__ == '__main__':
    analyze_memory_scaling()
    compare_attention_memory()
    kv_cache_analysis()
