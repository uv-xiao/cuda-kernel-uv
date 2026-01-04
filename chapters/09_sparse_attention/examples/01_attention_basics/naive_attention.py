#!/usr/bin/env python3
"""
Naive Attention Implementation in PyTorch
Reference implementation for testing and understanding
"""

import torch
import torch.nn.functional as F
import math
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np


def naive_attention_pytorch(Q, K, V, mask=None, is_causal=False, return_attention=False):
    """
    Standard scaled dot-product attention.

    Args:
        Q: Query tensor [batch_size, num_heads, seq_len, head_dim]
        K: Key tensor [batch_size, num_heads, seq_len, head_dim]
        V: Value tensor [batch_size, num_heads, seq_len, head_dim]
        mask: Optional mask [batch_size, seq_len, seq_len] or broadcastable
        is_causal: If True, apply causal (lower triangular) mask
        return_attention: If True, return attention weights as well

    Returns:
        output: [batch_size, num_heads, seq_len, head_dim]
        attention_weights: [batch_size, num_heads, seq_len, seq_len] (if return_attention=True)
    """
    batch_size, num_heads, seq_len, head_dim = Q.shape

    # Step 1: Compute attention scores QK^T
    # [B, H, L, d] @ [B, H, d, L] -> [B, H, L, L]
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # Step 2: Scale by sqrt(d_k)
    scores = scores / math.sqrt(head_dim)

    # Step 3: Apply mask
    if is_causal:
        # Create lower triangular mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device))
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 4: Softmax
    attention_weights = F.softmax(scores, dim=-1)

    # Handle NaN from -inf in softmax (all positions masked)
    attention_weights = torch.nan_to_num(attention_weights, 0.0)

    # Step 5: Apply attention to values
    # [B, H, L, L] @ [B, H, L, d] -> [B, H, L, d]
    output = torch.matmul(attention_weights, V)

    if return_attention:
        return output, attention_weights
    return output


def naive_attention_manual(Q, K, V, mask=None, is_causal=False):
    """
    Manual implementation showing explicit loops (for educational purposes).
    Much slower than PyTorch's optimized operations.
    """
    batch_size, num_heads, seq_len, head_dim = Q.shape
    output = torch.zeros_like(Q)

    scale = 1.0 / math.sqrt(head_dim)

    for b in range(batch_size):
        for h in range(num_heads):
            # Compute attention scores
            scores = torch.zeros(seq_len, seq_len, device=Q.device)
            for i in range(seq_len):
                for j in range(seq_len):
                    # Skip if causal and j > i
                    if is_causal and j > i:
                        scores[i, j] = float('-inf')
                        continue

                    # Dot product
                    score = 0.0
                    for d in range(head_dim):
                        score += Q[b, h, i, d] * K[b, h, j, d]
                    scores[i, j] = score * scale

            # Softmax
            attention = F.softmax(scores, dim=-1)

            # Weighted sum
            for i in range(seq_len):
                for d in range(head_dim):
                    value = 0.0
                    for j in range(seq_len):
                        value += attention[i, j] * V[b, h, j, d]
                    output[b, h, i, d] = value

    return output


def test_correctness():
    """Test that our implementation matches PyTorch's built-in."""
    print("=" * 60)
    print("Testing Correctness")
    print("=" * 60)

    torch.manual_seed(42)
    batch_size, num_heads, seq_len, head_dim = 2, 4, 64, 32

    Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Our implementation
    output_ours = naive_attention_pytorch(Q, K, V)

    # PyTorch's implementation
    output_torch = F.scaled_dot_product_attention(Q, K, V)

    # Compare
    max_diff = (output_ours - output_torch).abs().max().item()
    mean_diff = (output_ours - output_torch).abs().mean().item()

    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")

    if max_diff < 1e-5:
        print("✓ Correctness test PASSED")
    else:
        print("✗ Correctness test FAILED")

    # Test causal attention
    print("\nTesting causal attention...")
    output_ours_causal = naive_attention_pytorch(Q, K, V, is_causal=True)
    output_torch_causal = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

    max_diff_causal = (output_ours_causal - output_torch_causal).abs().max().item()
    print(f"Max difference (causal): {max_diff_causal:.2e}")

    if max_diff_causal < 1e-5:
        print("✓ Causal attention test PASSED")
    else:
        print("✗ Causal attention test FAILED")


def benchmark_attention(batch_size, num_heads, seq_len, head_dim, device='cuda'):
    """Benchmark attention performance."""
    print("=" * 60)
    print(f"Benchmarking Attention")
    print("=" * 60)
    print(f"Config: batch={batch_size}, heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}")
    print(f"Device: {device}")

    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    # Warmup
    for _ in range(10):
        _ = naive_attention_pytorch(Q, K, V)

    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark our implementation
    num_iters = 100
    start = time.time()
    for _ in range(num_iters):
        output = naive_attention_pytorch(Q, K, V)
    if device == 'cuda':
        torch.cuda.synchronize()
    end = time.time()

    time_ours = (end - start) / num_iters * 1000  # ms

    # Benchmark PyTorch's implementation
    start = time.time()
    for _ in range(num_iters):
        output = F.scaled_dot_product_attention(Q, K, V)
    if device == 'cuda':
        torch.cuda.synchronize()
    end = time.time()

    time_torch = (end - start) / num_iters * 1000  # ms

    print(f"\nResults:")
    print(f"  Our implementation: {time_ours:.3f} ms")
    print(f"  PyTorch SDPA: {time_torch:.3f} ms")
    print(f"  Ratio: {time_ours / time_torch:.2f}x")

    # Memory usage
    qkv_memory = 3 * batch_size * num_heads * seq_len * head_dim * 4 / 1024**2  # MB
    attn_memory = batch_size * num_heads * seq_len * seq_len * 4 / 1024**2  # MB

    print(f"\nMemory (theoretical):")
    print(f"  QKV tensors: {qkv_memory:.2f} MB")
    print(f"  Attention matrix: {attn_memory:.2f} MB")
    print(f"  Total: {qkv_memory + attn_memory:.2f} MB")

    # FLOPs analysis
    # QK^T: 2 * B * H * L^2 * d (multiply-add)
    # Softmax: ~5 * B * H * L^2 (exp, sum, div, etc.)
    # PV: 2 * B * H * L^2 * d
    flops = 4 * batch_size * num_heads * seq_len * seq_len * head_dim + \
            5 * batch_size * num_heads * seq_len * seq_len
    tflops = flops / (time_ours * 1e-3) / 1e12

    print(f"\nCompute:")
    print(f"  FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"  Throughput: {tflops:.2f} TFLOPs/s")


def visualize_attention_pattern(seq_len=64, head_dim=32):
    """Visualize attention patterns."""
    print("=" * 60)
    print("Visualizing Attention Patterns")
    print("=" * 60)

    torch.manual_seed(42)
    Q = torch.randn(1, 1, seq_len, head_dim)
    K = torch.randn(1, 1, seq_len, head_dim)
    V = torch.randn(1, 1, seq_len, head_dim)

    # Compute attention
    _, attention = naive_attention_pytorch(Q, K, V, return_attention=True)
    attention = attention[0, 0].cpu().numpy()  # [seq_len, seq_len]

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Regular attention
    im0 = axes[0].imshow(attention, cmap='viridis', aspect='auto')
    axes[0].set_title('Regular Attention')
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    plt.colorbar(im0, ax=axes[0])

    # Causal attention
    _, attention_causal = naive_attention_pytorch(Q, K, V, is_causal=True, return_attention=True)
    attention_causal = attention_causal[0, 0].cpu().numpy()
    im1 = axes[1].imshow(attention_causal, cmap='viridis', aspect='auto')
    axes[1].set_title('Causal Attention')
    axes[1].set_xlabel('Key Position')
    axes[1].set_ylabel('Query Position')
    plt.colorbar(im1, ax=axes[1])

    # Attention entropy (measure of how focused the attention is)
    # Higher entropy = more uniform attention, lower = more focused
    entropy = -np.sum(attention * np.log(attention + 1e-9), axis=1)
    axes[2].plot(entropy)
    axes[2].set_title('Attention Entropy per Query')
    axes[2].set_xlabel('Query Position')
    axes[2].set_ylabel('Entropy (nats)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('attention_patterns.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to attention_patterns.png")
    print(f"Mean attention entropy: {entropy.mean():.3f} nats")


def scaling_analysis():
    """Analyze how performance scales with sequence length."""
    print("=" * 60)
    print("Scaling Analysis")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping scaling analysis")
        return

    batch_size, num_heads, head_dim = 4, 8, 64
    seq_lengths = [128, 256, 512, 1024, 2048]

    times = []
    memories = []

    for seq_len in seq_lengths:
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')

        # Warmup
        for _ in range(5):
            _ = naive_attention_pytorch(Q, K, V)
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(20):
            output = naive_attention_pytorch(Q, K, V)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / 20 * 1000

        times.append(elapsed)

        # Memory
        attn_memory = batch_size * num_heads * seq_len * seq_len * 4 / 1024**2
        memories.append(attn_memory)

        print(f"seq_len={seq_len:4d}: {elapsed:6.2f} ms, {attn_memory:7.2f} MB")

    # Plot scaling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Time scaling
    ax1.plot(seq_lengths, times, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Time vs Sequence Length')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=2)

    # Memory scaling
    ax2.plot(seq_lengths, memories, 'o-', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Attention Matrix Memory (MB)')
    ax2.set_title('Memory vs Sequence Length')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log', base=2)

    plt.tight_layout()
    plt.savefig('scaling_analysis.png', dpi=150, bbox_inches='tight')
    print("\nSaved scaling analysis to scaling_analysis.png")
    print("Note: Both time and memory scale as O(L²)")


def main():
    parser = argparse.ArgumentParser(description='Naive Attention Implementation')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--head_dim', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test', action='store_true', help='Run correctness tests')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmarks')
    parser.add_argument('--visualize', action='store_true', help='Visualize attention patterns')
    parser.add_argument('--scaling', action='store_true', help='Run scaling analysis')
    parser.add_argument('--all', action='store_true', help='Run all tests')

    args = parser.parse_args()

    if args.all:
        args.test = args.benchmark = args.visualize = args.scaling = True

    if not (args.test or args.benchmark or args.visualize or args.scaling):
        # Default: run everything
        args.test = args.benchmark = args.visualize = args.scaling = True

    if args.test:
        test_correctness()
        print()

    if args.benchmark:
        benchmark_attention(args.batch_size, args.num_heads, args.seq_len,
                          args.head_dim, args.device)
        print()

    if args.visualize:
        visualize_attention_pattern()
        print()

    if args.scaling:
        scaling_analysis()
        print()


if __name__ == '__main__':
    main()
