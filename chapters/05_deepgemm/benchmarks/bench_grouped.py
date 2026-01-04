#!/usr/bin/env python3
"""Benchmark grouped GEMM for MoE workloads"""

import torch
import time
import argparse
import csv
import numpy as np


def simulate_token_distribution(total_tokens, num_experts, balance='uniform'):
    """Simulate token distribution across experts"""
    if balance == 'uniform':
        base = total_tokens // num_experts
        remainder = total_tokens % num_experts
        return [base + (1 if i < remainder else 0) for i in range(num_experts)]
    elif balance == 'imbalanced':
        # Zipf-like distribution
        ranks = np.arange(1, num_experts + 1)
        probs = 1.0 / ranks
        probs = probs / probs.sum()
        counts = (probs * total_tokens).astype(int)
        # Adjust to exact total
        counts[0] += total_tokens - counts.sum()
        return counts.tolist()
    elif balance == 'skewed':
        # One expert gets most tokens
        counts = [total_tokens // (num_experts * 2)] * num_experts
        counts[0] = total_tokens - sum(counts[1:])
        return counts
    else:
        raise ValueError(f"Unknown balance: {balance}")


def benchmark_grouped_gemm(token_counts, hidden_dim, expert_dim, warmup=10, iters=100):
    """Benchmark grouped GEMM (simplified)"""
    num_experts = len(token_counts)

    # Create grouped inputs
    grouped_inputs = []
    expert_weights = []

    for count in token_counts:
        if count > 0:
            grouped_inputs.append(
                torch.randn(count, hidden_dim, dtype=torch.bfloat16, device='cuda')
            )
        else:
            grouped_inputs.append(
                torch.empty(0, hidden_dim, dtype=torch.bfloat16, device='cuda')
            )

    for _ in range(num_experts):
        expert_weights.append(
            torch.randn(hidden_dim, expert_dim, dtype=torch.bfloat16, device='cuda')
        )

    # Warmup
    for _ in range(warmup):
        outputs = []
        for inp, weight in zip(grouped_inputs, expert_weights):
            if inp.size(0) > 0:
                outputs.append(torch.matmul(inp, weight))
            else:
                outputs.append(torch.empty(0, expert_dim, dtype=torch.bfloat16, device='cuda'))
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(iters):
        outputs = []
        for inp, weight in zip(grouped_inputs, expert_weights):
            if inp.size(0) > 0:
                outputs.append(torch.matmul(inp, weight))
            else:
                outputs.append(torch.empty(0, expert_dim, dtype=torch.bfloat16, device='cuda'))
    torch.cuda.synchronize()
    elapsed = time.time() - start

    time_ms = elapsed / iters * 1000

    # Calculate useful FLOPs
    useful_flops = sum(2 * count * hidden_dim * expert_dim for count in token_counts)
    tflops = (useful_flops / 1e12) / (time_ms / 1000)

    return time_ms, tflops, useful_flops


def benchmark_padded_gemm(token_counts, hidden_dim, expert_dim, warmup=10, iters=100):
    """Benchmark padded batched GEMM"""
    num_experts = len(token_counts)
    max_tokens = max(token_counts)

    # Create padded batch
    batch_input = torch.zeros(num_experts, max_tokens, hidden_dim,
                              dtype=torch.bfloat16, device='cuda')

    for i, count in enumerate(token_counts):
        if count > 0:
            batch_input[i, :count] = torch.randn(count, hidden_dim,
                                                 dtype=torch.bfloat16, device='cuda')

    expert_weights = torch.randn(num_experts, hidden_dim, expert_dim,
                                dtype=torch.bfloat16, device='cuda')

    # Warmup
    for _ in range(warmup):
        output = torch.matmul(batch_input, expert_weights)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(iters):
        output = torch.matmul(batch_input, expert_weights)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    time_ms = elapsed / iters * 1000

    # Calculate total FLOPs (including padding waste)
    total_flops = 2 * num_experts * max_tokens * hidden_dim * expert_dim
    useful_flops = sum(2 * count * hidden_dim * expert_dim for count in token_counts)
    tflops = (total_flops / 1e12) / (time_ms / 1000)

    efficiency = useful_flops / total_flops * 100

    return time_ms, tflops, total_flops, efficiency


def run_benchmark_suite(num_experts_list, hidden_dim, expert_dim, total_tokens, output_file=None):
    """Run complete benchmark suite"""

    print(f"{'='*80}")
    print(f"Grouped GEMM Benchmark for MoE")
    print(f"{'='*80}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Hidden dim: {hidden_dim}, Expert dim: {expert_dim}")
    print(f"Total tokens: {total_tokens}")
    print(f"{'='*80}\n")

    results = []

    for num_experts in num_experts_list:
        print(f"\n{'='*80}")
        print(f"Experts: {num_experts}")
        print(f"{'='*80}\n")

        for balance in ['uniform', 'imbalanced', 'skewed']:
            token_counts = simulate_token_distribution(total_tokens, num_experts, balance)

            print(f"{balance.capitalize()} distribution:")
            print(f"  Token counts: {token_counts}")
            print(f"  Max/Min ratio: {max(token_counts) / max(min(token_counts), 1):.1f}x")

            # Grouped GEMM
            time_grouped, tflops_grouped, useful_flops = \
                benchmark_grouped_gemm(token_counts, hidden_dim, expert_dim)

            # Padded GEMM
            time_padded, tflops_padded, total_flops, efficiency = \
                benchmark_padded_gemm(token_counts, hidden_dim, expert_dim)

            speedup = time_padded / time_grouped

            print(f"\n  Grouped GEMM:")
            print(f"    Time: {time_grouped:.3f} ms")
            print(f"    Throughput: {tflops_grouped:.1f} TFLOPS")
            print(f"    Useful FLOPs: {useful_flops/1e9:.2f} GFLOPS")

            print(f"\n  Padded Batched GEMM:")
            print(f"    Time: {time_padded:.3f} ms")
            print(f"    Throughput: {tflops_padded:.1f} TFLOPS")
            print(f"    Total FLOPs: {total_flops/1e9:.2f} GFLOPS")
            print(f"    Compute efficiency: {efficiency:.1f}%")

            print(f"\n  Speedup: {speedup:.2f}x\n")

            results.append({
                'num_experts': num_experts,
                'balance': balance,
                'token_counts': str(token_counts),
                'time_grouped_ms': time_grouped,
                'tflops_grouped': tflops_grouped,
                'time_padded_ms': time_padded,
                'tflops_padded': tflops_padded,
                'efficiency_%': efficiency,
                'speedup': speedup
            })

    # Summary
    print(f"\n{'='*80}")
    print(f"Summary")
    print(f"{'='*80}")
    print(f"{'Experts':<10} {'Balance':<15} {'Efficiency':<12} {'Speedup':<10}")
    print(f"{'-'*80}")

    for r in results:
        print(f"{r['num_experts']:<10} {r['balance']:<15} "
              f"{r['efficiency_%']:<12.1f}% {r['speedup']:<10.2f}x")

    # Save to file
    if output_file:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {output_file}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark Grouped GEMM for MoE')
    parser.add_argument('--experts', type=str, default='8,16,32',
                       help='Comma-separated expert counts (default: 8,16,32)')
    parser.add_argument('--hidden', type=int, default=2048,
                       help='Hidden dimension (default: 2048)')
    parser.add_argument('--expert-dim', type=int, default=None,
                       help='Expert dimension (default: 4*hidden)')
    parser.add_argument('--tokens', type=int, default=1024,
                       help='Total tokens (default: 1024)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file (default: None)')
    parser.add_argument('--warmup', type=int, default=10,
                       help='Warmup iterations (default: 10)')
    parser.add_argument('--iters', type=int, default=100,
                       help='Benchmark iterations (default: 100)')

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        return

    num_experts_list = [int(e) for e in args.experts.split(',')]
    expert_dim = args.expert_dim or args.hidden * 4

    run_benchmark_suite(num_experts_list, args.hidden, expert_dim,
                       args.tokens, args.output)


if __name__ == '__main__':
    main()
