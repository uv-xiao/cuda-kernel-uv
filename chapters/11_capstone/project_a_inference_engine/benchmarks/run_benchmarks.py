"""
Comprehensive Benchmarking Suite for Inference Engine

This script benchmarks:
1. Attention kernels (Flash vs Reference)
2. MoE layers
3. End-to-end inference
"""

import torch
import time
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import sys
import os

# Add parent directories to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'starter'))
sys.path.append(os.path.join(parent_dir, 'reference'))


def benchmark_function(
    func,
    *args,
    num_iterations: int = 100,
    warmup: int = 10,
    **kwargs
) -> Dict:
    """Benchmark a function"""
    # Warmup
    for _ in range(warmup):
        _ = func(*args, **kwargs)

    # Synchronize
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        output = func(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    # Calculate metrics
    total_time = end - start
    avg_time = total_time / num_iterations
    std_time = 0  # Could compute std if needed

    return {
        'avg_time_ms': avg_time * 1000,
        'std_time_ms': std_time * 1000,
        'throughput_per_s': 1.0 / avg_time,
    }


def benchmark_attention(args):
    """Benchmark attention implementations"""
    print("=" * 80)
    print("ATTENTION BENCHMARKS")
    print("=" * 80)

    try:
        from starter.attention import AttentionLayer
        from reference.inference_engine import ReferenceAttention
    except ImportError as e:
        print(f"Could not import attention modules: {e}")
        return []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = []

    # Test configurations
    configs = [
        {'batch': 1, 'seq_len': 128, 'heads': 8, 'head_dim': 64},
        {'batch': 4, 'seq_len': 512, 'heads': 16, 'head_dim': 64},
        {'batch': 16, 'seq_len': 2048, 'heads': 16, 'head_dim': 64},
        {'batch': 32, 'seq_len': 4096, 'heads': 32, 'head_dim': 128},
    ]

    for config in configs:
        batch = config['batch']
        seq_len = config['seq_len']
        num_heads = config['heads']
        head_dim = config['head_dim']
        hidden_dim = num_heads * head_dim

        print(f"\nConfig: batch={batch}, seq_len={seq_len}, heads={num_heads}, head_dim={head_dim}")

        # Create input
        x = torch.randn(batch, seq_len, hidden_dim, device=device)

        # Reference implementation
        ref_attn = ReferenceAttention(hidden_dim, num_heads).to(device)
        ref_attn.eval()

        with torch.no_grad():
            try:
                ref_stats = benchmark_function(
                    ref_attn, x,
                    num_iterations=args.num_iterations,
                    warmup=args.warmup,
                )
                print(f"  Reference: {ref_stats['avg_time_ms']:.2f} ms")
            except Exception as e:
                print(f"  Reference failed: {e}")
                ref_stats = {'avg_time_ms': float('inf')}

        # Custom implementation
        try:
            custom_attn = AttentionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                use_flash=True,
            ).to(device)
            custom_attn.eval()

            with torch.no_grad():
                custom_stats = benchmark_function(
                    custom_attn, x,
                    num_iterations=args.num_iterations,
                    warmup=args.warmup,
                )
                print(f"  Custom:    {custom_stats['avg_time_ms']:.2f} ms")

                # Speedup
                speedup = ref_stats['avg_time_ms'] / custom_stats['avg_time_ms']
                print(f"  Speedup:   {speedup:.2f}x")

        except NotImplementedError:
            print("  Custom: Not implemented yet")
            custom_stats = {'avg_time_ms': float('inf')}
            speedup = 0

        # Record results
        results.append({
            'type': 'attention',
            'batch': batch,
            'seq_len': seq_len,
            'num_heads': num_heads,
            'head_dim': head_dim,
            'reference_time_ms': ref_stats['avg_time_ms'],
            'custom_time_ms': custom_stats['avg_time_ms'],
            'speedup': speedup,
        })

    return results


def benchmark_moe(args):
    """Benchmark MoE implementations"""
    print("\n" + "=" * 80)
    print("MoE BENCHMARKS")
    print("=" * 80)

    try:
        from starter.moe import MoELayer
    except ImportError as e:
        print(f"Could not import MoE module: {e}")
        return []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = []

    # Test configurations
    configs = [
        {'batch': 4, 'seq_len': 128, 'hidden': 1024, 'intermediate': 4096, 'experts': 8, 'top_k': 2},
        {'batch': 16, 'seq_len': 512, 'hidden': 2048, 'intermediate': 8192, 'experts': 16, 'top_k': 2},
        {'batch': 32, 'seq_len': 2048, 'hidden': 4096, 'intermediate': 11008, 'experts': 32, 'top_k': 2},
    ]

    for config in configs:
        batch = config['batch']
        seq_len = config['seq_len']
        hidden_dim = config['hidden']
        intermediate_dim = config['intermediate']
        num_experts = config['experts']
        top_k = config['top_k']

        print(f"\nConfig: batch={batch}, seq_len={seq_len}, hidden={hidden_dim}, "
              f"experts={num_experts}, top_k={top_k}")

        # Create input
        x = torch.randn(batch, seq_len, hidden_dim, device=device)

        # MoE layer
        try:
            moe = MoELayer(
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_experts=num_experts,
                num_experts_per_token=top_k,
            ).to(device)
            moe.eval()

            with torch.no_grad():
                stats = benchmark_function(
                    moe, x,
                    num_iterations=args.num_iterations,
                    warmup=args.warmup,
                )
                print(f"  Time: {stats['avg_time_ms']:.2f} ms")

                # Calculate throughput
                num_tokens = batch * seq_len
                tokens_per_sec = num_tokens / (stats['avg_time_ms'] / 1000)
                print(f"  Throughput: {tokens_per_sec:.1f} tokens/s")

        except Exception as e:
            print(f"  MoE failed: {e}")
            stats = {'avg_time_ms': float('inf')}
            tokens_per_sec = 0

        # Record results
        results.append({
            'type': 'moe',
            'batch': batch,
            'seq_len': seq_len,
            'hidden_dim': hidden_dim,
            'num_experts': num_experts,
            'top_k': top_k,
            'time_ms': stats['avg_time_ms'],
            'throughput': tokens_per_sec,
        })

    return results


def benchmark_end_to_end(args):
    """Benchmark end-to-end inference"""
    print("\n" + "=" * 80)
    print("END-TO-END BENCHMARKS")
    print("=" * 80)

    try:
        from starter.inference_engine import InferenceEngine, TransformerConfig
        from reference.inference_engine import ReferenceInferenceEngine
    except ImportError as e:
        print(f"Could not import inference engine: {e}")
        return []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = []

    # Test configurations (different model sizes)
    configs = [
        {
            'name': 'Small',
            'vocab_size': 32000,
            'hidden_dim': 768,
            'num_layers': 12,
            'num_heads': 12,
            'intermediate_dim': 3072,
            'batch': 4,
            'seq_len': 512,
        },
        {
            'name': 'Medium',
            'vocab_size': 32000,
            'hidden_dim': 2048,
            'num_layers': 24,
            'num_heads': 16,
            'intermediate_dim': 8192,
            'batch': 8,
            'seq_len': 1024,
        },
    ]

    for config in configs:
        print(f"\nModel: {config['name']}")
        print(f"  Layers: {config['num_layers']}, Hidden: {config['hidden_dim']}")
        print(f"  Batch: {config['batch']}, Seq Len: {config['seq_len']}")

        # Create input
        batch = config['batch']
        seq_len = config['seq_len']
        input_ids = torch.randint(0, config['vocab_size'], (batch, seq_len), device=device)

        # Reference implementation
        ref_model = ReferenceInferenceEngine(
            vocab_size=config['vocab_size'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            intermediate_dim=config['intermediate_dim'],
        ).to(device)
        ref_model.eval()

        with torch.no_grad():
            try:
                ref_stats = benchmark_function(
                    ref_model, input_ids,
                    num_iterations=10,
                    warmup=2,
                )
                print(f"  Reference: {ref_stats['avg_time_ms']:.2f} ms")
            except Exception as e:
                print(f"  Reference failed: {e}")
                ref_stats = {'avg_time_ms': float('inf')}

        # Custom implementation
        try:
            custom_config = TransformerConfig(
                vocab_size=config['vocab_size'],
                hidden_dim=config['hidden_dim'],
                num_layers=config['num_layers'],
                num_heads=config['num_heads'],
                head_dim=config['hidden_dim'] // config['num_heads'],
                intermediate_dim=config['intermediate_dim'],
                use_moe=False,
                use_flash_attention=True,
            )

            custom_model = InferenceEngine(custom_config).to(device)
            custom_model.eval()

            with torch.no_grad():
                custom_stats = benchmark_function(
                    custom_model, input_ids,
                    num_iterations=10,
                    warmup=2,
                )
                print(f"  Custom:    {custom_stats['avg_time_ms']:.2f} ms")

                speedup = ref_stats['avg_time_ms'] / custom_stats['avg_time_ms']
                print(f"  Speedup:   {speedup:.2f}x")

        except NotImplementedError:
            print("  Custom: Not implemented yet")
            custom_stats = {'avg_time_ms': float('inf')}
            speedup = 0

        # Record results
        results.append({
            'type': 'end_to_end',
            'model': config['name'],
            'num_layers': config['num_layers'],
            'hidden_dim': config['hidden_dim'],
            'batch': batch,
            'seq_len': seq_len,
            'reference_time_ms': ref_stats['avg_time_ms'],
            'custom_time_ms': custom_stats['avg_time_ms'],
            'speedup': speedup,
        })

    return results


def visualize_results(results: List[Dict], output_dir: str = '.'):
    """Create visualizations of benchmark results"""
    print("\nGenerating visualizations...")

    # Separate by type
    attention_results = [r for r in results if r['type'] == 'attention']
    moe_results = [r for r in results if r['type'] == 'moe']
    e2e_results = [r for r in results if r['type'] == 'end_to_end']

    # Plot attention benchmarks
    if attention_results:
        fig, ax = plt.subplots(figsize=(10, 6))

        seq_lens = [r['seq_len'] for r in attention_results]
        ref_times = [r['reference_time_ms'] for r in attention_results]
        custom_times = [r['custom_time_ms'] for r in attention_results]

        x = np.arange(len(seq_lens))
        width = 0.35

        ax.bar(x - width/2, ref_times, width, label='Reference', alpha=0.8)
        ax.bar(x + width/2, custom_times, width, label='Custom', alpha=0.8)

        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Time (ms)')
        ax.set_title('Attention Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(seq_lens)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'attention_benchmark.png'))
        print(f"  Saved: {output_dir}/attention_benchmark.png")

    # Plot MoE benchmarks
    if moe_results:
        fig, ax = plt.subplots(figsize=(10, 6))

        labels = [f"B={r['batch']}, S={r['seq_len']}" for r in moe_results]
        times = [r['time_ms'] for r in moe_results]

        ax.bar(labels, times, alpha=0.8)
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Time (ms)')
        ax.set_title('MoE Layer Performance')
        ax.grid(axis='y', alpha=0.3)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'moe_benchmark.png'))
        print(f"  Saved: {output_dir}/moe_benchmark.png")

    print("Visualization complete!")


def main():
    parser = argparse.ArgumentParser(description='Benchmark inference engine')
    parser.add_argument('--benchmark', choices=['attention', 'moe', 'e2e', 'all'], default='all',
                        help='Which benchmark to run')
    parser.add_argument('--num-iterations', type=int, default=100,
                        help='Number of iterations for benchmarking')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Number of warmup iterations')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='Output file for results')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots')

    args = parser.parse_args()

    print("Starting benchmarks...")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    results = []

    # Run benchmarks
    if args.benchmark in ['attention', 'all']:
        results.extend(benchmark_attention(args))

    if args.benchmark in ['moe', 'all']:
        results.extend(benchmark_moe(args))

    if args.benchmark in ['e2e', 'all']:
        results.extend(benchmark_end_to_end(args))

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Visualize if requested
    if args.visualize and results:
        output_dir = os.path.dirname(args.output) or '.'
        visualize_results(results, output_dir)

    print("\nBenchmarking complete!")


if __name__ == "__main__":
    main()
