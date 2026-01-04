"""
Unified Benchmark Script for All GEMM Implementations

This script benchmarks all four GEMM implementations:
- CUDA C++
- Triton
- TileLang
- CUTLASS

And compares against cuBLAS baseline.
"""

import torch
import time
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import subprocess
import os


def benchmark_function(func, *args, num_iterations=100, warmup=10, **kwargs) -> Dict:
    """Benchmark a function and return timing statistics"""

    # Warmup
    for _ in range(warmup):
        _ = func(*args, **kwargs)

    # Synchronize
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        result = func(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    # Calculate metrics
    total_time = end - start
    avg_time = total_time / num_iterations

    return {
        'avg_time_ms': avg_time * 1000,
        'avg_time_s': avg_time,
    }


def gemm_cublas(A, B, C, alpha=1.0, beta=0.0):
    """Baseline GEMM using cuBLAS (via PyTorch)"""
    return torch.addmm(C, A, B, alpha=alpha, beta=beta)


def gemm_cuda(A, B, C, alpha=1.0, beta=0.0):
    """CUDA C++ GEMM implementation"""
    # TODO: Import and call your CUDA implementation
    # Example:
    # from gemm_cuda_module import gemm_cuda
    # return gemm_cuda(A, B, C, alpha, beta)

    raise NotImplementedError("CUDA implementation not available")


def gemm_triton(A, B, C, alpha=1.0, beta=0.0):
    """Triton GEMM implementation"""
    try:
        from gemm_triton import gemm_triton as triton_impl
        return triton_impl(A, B, C.clone(), alpha, beta)
    except ImportError:
        raise NotImplementedError("Triton implementation not available")


def gemm_tilelang(A, B, C, alpha=1.0, beta=0.0):
    """TileLang GEMM implementation"""
    # TODO: Import and call your TileLang implementation
    # Example:
    # from gemm_tilelang import gemm_tilelang as tilelang_impl
    # return tilelang_impl(A, B, C, alpha, beta)

    raise NotImplementedError("TileLang implementation not available")


def gemm_cutlass(A, B, C, alpha=1.0, beta=0.0):
    """CUTLASS GEMM implementation"""
    # TODO: Import and call your CUTLASS implementation
    # Example:
    # from gemm_cutlass_module import gemm_cutlass
    # return gemm_cutlass(A, B, C, alpha, beta)

    raise NotImplementedError("CUTLASS implementation not available")


def verify_correctness(impl_func, impl_name, A, B, C, alpha=1.0, beta=0.0, rtol=1e-3, atol=1e-5):
    """Verify correctness against cuBLAS"""
    print(f"\nVerifying {impl_name}...")

    try:
        # Run implementation
        result = impl_func(A, B, C.clone(), alpha, beta)

        # Run cuBLAS reference
        reference = gemm_cublas(A, B, C.clone(), alpha, beta)

        # Check correctness
        max_error = torch.max(torch.abs(result - reference)).item()
        rel_error = max_error / (torch.max(torch.abs(reference)).item() + 1e-8)

        is_correct = torch.allclose(result, reference, rtol=rtol, atol=atol)

        print(f"  Max absolute error: {max_error:.6e}")
        print(f"  Max relative error: {rel_error:.6e}")
        print(f"  Correctness: {'PASS' if is_correct else 'FAIL'}")

        return is_correct

    except NotImplementedError:
        print(f"  Not implemented yet")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def benchmark_gemm_config(
    M: int, N: int, K: int,
    implementations: Dict,
    dtype=torch.float32,
    num_iterations=100,
    warmup=10,
) -> Dict:
    """Benchmark all implementations for a given configuration"""

    print(f"\n{'='*80}")
    print(f"Configuration: M={M}, N={N}, K={K}, dtype={dtype}")
    print(f"{'='*80}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create random matrices
    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(K, N, device=device, dtype=dtype)
    C = torch.zeros(M, N, device=device, dtype=dtype)

    alpha, beta = 1.0, 0.0

    results = {
        'M': M, 'N': N, 'K': K,
        'dtype': str(dtype),
    }

    # Benchmark each implementation
    for impl_name, impl_func in implementations.items():
        print(f"\nBenchmarking {impl_name}...")

        try:
            stats = benchmark_function(
                impl_func, A, B, C.clone(), alpha, beta,
                num_iterations=num_iterations,
                warmup=warmup,
            )

            # Calculate FLOPS
            flops = 2 * M * N * K  # Multiply-add operations
            tflops = flops / (stats['avg_time_s'] * 1e12)

            # Store results
            results[f'{impl_name}_time_ms'] = stats['avg_time_ms']
            results[f'{impl_name}_tflops'] = tflops

            print(f"  Time: {stats['avg_time_ms']:.2f} ms")
            print(f"  Performance: {tflops:.2f} TFLOPS")

        except NotImplementedError:
            print(f"  Not implemented")
            results[f'{impl_name}_time_ms'] = float('inf')
            results[f'{impl_name}_tflops'] = 0.0
        except Exception as e:
            print(f"  Error: {e}")
            results[f'{impl_name}_time_ms'] = float('inf')
            results[f'{impl_name}_tflops'] = 0.0

    # Calculate speedups relative to cuBLAS
    cublas_time = results.get('cuBLAS_time_ms', float('inf'))

    for impl_name in implementations.keys():
        if impl_name != 'cuBLAS':
            impl_time = results.get(f'{impl_name}_time_ms', float('inf'))
            if impl_time < float('inf') and cublas_time < float('inf'):
                speedup = cublas_time / impl_time
                pct_of_cublas = (results[f'{impl_name}_tflops'] /
                                results['cuBLAS_tflops'] * 100)
                results[f'{impl_name}_speedup'] = speedup
                results[f'{impl_name}_pct_cublas'] = pct_of_cublas

                print(f"\n{impl_name} vs cuBLAS:")
                print(f"  Speedup: {speedup:.2f}x")
                print(f"  % of cuBLAS: {pct_of_cublas:.1f}%")

    return results


def run_comprehensive_benchmark(args):
    """Run comprehensive benchmarks across all configurations"""

    # Define implementations to test
    implementations = {
        'cuBLAS': gemm_cublas,
        'CUDA': gemm_cuda,
        'Triton': gemm_triton,
        'TileLang': gemm_tilelang,
        'CUTLASS': gemm_cutlass,
    }

    # Test configurations
    if args.quick:
        # Quick test
        configs = [
            (128, 128, 128),
            (1024, 1024, 1024),
            (4096, 4096, 4096),
        ]
        num_iterations = 20
        warmup = 5
    else:
        # Comprehensive test
        configs = [
            # Small
            (128, 128, 128),
            (256, 256, 256),
            (512, 512, 512),
            # Medium
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            # Large
            (4096, 4096, 4096),
            (8192, 8192, 8192),
            # Rectangular
            (8192, 512, 2048),
            (4096, 1024, 4096),
            (2048, 4096, 1024),
        ]
        num_iterations = args.num_iterations
        warmup = args.warmup

    # Run benchmarks
    all_results = []

    for M, N, K in configs:
        results = benchmark_gemm_config(
            M, N, K,
            implementations,
            dtype=torch.float32,
            num_iterations=num_iterations,
            warmup=warmup,
        )
        all_results.append(results)

    # Save results
    output_file = args.output
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to {output_file}")

    return all_results


def visualize_results(results: List[Dict], output_dir='.'):
    """Create visualizations of benchmark results"""

    print("\nGenerating visualizations...")

    # Extract data
    configs = [f"M=N=K={r['M']}" if r['M'] == r['N'] == r['K']
               else f"{r['M']}x{r['N']}x{r['K']}"
               for r in results]

    implementations = ['CUDA', 'Triton', 'TileLang', 'CUTLASS']

    # Plot 1: Performance (TFLOPS) comparison
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(configs))
    width = 0.15

    for i, impl in enumerate(implementations):
        tflops = [r.get(f'{impl}_tflops', 0) for r in results]
        ax.bar(x + i * width, tflops, width, label=impl, alpha=0.8)

    # cuBLAS reference line
    cublas_tflops = [r.get('cuBLAS_tflops', 0) for r in results]
    ax.plot(x + 1.5 * width, cublas_tflops, 'k--', linewidth=2, label='cuBLAS (reference)')

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Performance (TFLOPS)')
    ax.set_title('GEMM Performance Comparison')
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=150)
    print(f"  Saved: {output_dir}/performance_comparison.png")

    # Plot 2: Percentage of cuBLAS
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, impl in enumerate(implementations):
        pct = [r.get(f'{impl}_pct_cublas', 0) for r in results]
        ax.bar(x + i * width, pct, width, label=impl, alpha=0.8)

    ax.axhline(y=100, color='k', linestyle='--', linewidth=2, label='cuBLAS (100%)')
    ax.set_xlabel('Configuration')
    ax.set_ylabel('% of cuBLAS Performance')
    ax.set_title('Performance Relative to cuBLAS')
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pct_cublas_comparison.png'), dpi=150)
    print(f"  Saved: {output_dir}/pct_cublas_comparison.png")

    # Plot 3: Execution time comparison
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, impl in enumerate(implementations):
        times = [r.get(f'{impl}_time_ms', float('inf')) for r in results]
        times = [t if t < float('inf') else 0 for t in times]
        ax.bar(x + i * width, times, width, label=impl, alpha=0.8)

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Time (ms)')
    ax.set_title('GEMM Execution Time')
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_comparison.png'), dpi=150)
    print(f"  Saved: {output_dir}/time_comparison.png")

    print("Visualization complete!")


def main():
    parser = argparse.ArgumentParser(description='Benchmark GEMM implementations')
    parser.add_argument('--verify', action='store_true',
                        help='Run correctness verification')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with fewer configurations')
    parser.add_argument('--num-iterations', type=int, default=100,
                        help='Number of iterations for benchmarking')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Number of warmup iterations')
    parser.add_argument('--output', type=str, default='gemm_benchmark_results.json',
                        help='Output file for results')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots')

    args = parser.parse_args()

    print("GEMM Implementation Comparison Benchmark")
    print("=" * 80)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    else:
        print("WARNING: CUDA not available, using CPU")
    print("=" * 80)

    # Correctness verification
    if args.verify:
        print("\n" + "=" * 80)
        print("CORRECTNESS VERIFICATION")
        print("=" * 80)

        M, N, K = 256, 256, 256
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        A = torch.randn(M, K, device=device, dtype=torch.float32)
        B = torch.randn(K, N, device=device, dtype=torch.float32)
        C = torch.zeros(M, N, device=device, dtype=torch.float32)

        implementations = {
            'CUDA': gemm_cuda,
            'Triton': gemm_triton,
            'TileLang': gemm_tilelang,
            'CUTLASS': gemm_cutlass,
        }

        for impl_name, impl_func in implementations.items():
            verify_correctness(impl_func, impl_name, A, B, C)

    # Benchmarking
    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARKING")
    print("=" * 80)

    results = run_comprehensive_benchmark(args)

    # Visualization
    if args.visualize:
        output_dir = os.path.dirname(args.output) or '.'
        visualize_results(results, output_dir)

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
