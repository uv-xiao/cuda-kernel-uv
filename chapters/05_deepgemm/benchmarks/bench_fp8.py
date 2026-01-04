#!/usr/bin/env python3
"""Benchmark FP8 vs BF16 GEMM performance"""

import torch
import time
import argparse
import csv
from pathlib import Path


def benchmark_gemm(M, N, K, dtype, warmup=10, iters=100):
    """Benchmark single GEMM operation"""
    A = torch.randn(M, K, dtype=dtype, device='cuda')
    B = torch.randn(K, N, dtype=dtype, device='cuda')

    # Warmup
    for _ in range(warmup):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(iters):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    time_ms = elapsed / iters * 1000
    flops = 2 * M * N * K
    tflops = (flops / 1e12) / (time_ms / 1000)

    # Memory bandwidth
    bytes_transferred = (M * K + K * N + M * N) * torch.finfo(dtype).bits / 8
    bandwidth_gb = (bytes_transferred / 1e9) / (time_ms / 1000)

    return {
        'time_ms': time_ms,
        'tflops': tflops,
        'bandwidth_gb_s': bandwidth_gb
    }


def run_benchmark_suite(sizes, output_file=None):
    """Run complete benchmark suite"""

    print(f"{'='*80}")
    print(f"FP8 vs BF16 GEMM Benchmark")
    print(f"{'='*80}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability()}")
    print(f"{'='*80}\n")

    results = []

    # Check FP8 support
    has_fp8 = torch.cuda.get_device_capability()[0] >= 9
    dtypes = [torch.bfloat16, torch.float16]
    if has_fp8:
        print("FP8 Tensor Cores detected (SM 9.0+)\n")
        # Note: PyTorch doesn't have native FP8 yet, so we benchmark BF16/FP16
        # In production, use CUTLASS or DeepGEMM for true FP8
    else:
        print("FP8 Tensor Cores NOT available (need SM 9.0+)\n")

    print(f"{'Size':<12} {'Dtype':<10} {'Time (ms)':<12} {'TFLOPS':<12} {'GB/s':<12}")
    print(f"{'-'*80}")

    for size in sizes:
        M = N = K = size

        for dtype in dtypes:
            result = benchmark_gemm(M, N, K, dtype)

            print(f"{size:<12} {str(dtype).split('.')[-1]:<10} "
                  f"{result['time_ms']:<12.3f} {result['tflops']:<12.1f} "
                  f"{result['bandwidth_gb_s']:<12.1f}")

            results.append({
                'size': size,
                'M': M, 'N': N, 'K': K,
                'dtype': str(dtype),
                **result
            })

        print()

    # Calculate speedups
    print(f"\n{'='*80}")
    print(f"Speedup Analysis (FP16 baseline)")
    print(f"{'='*80}")
    print(f"{'Size':<12} {'BF16/FP16':<15} {'Theoretical FP8*':<15}")
    print(f"{'-'*80}")

    for size in sizes:
        fp16_result = [r for r in results if r['size'] == size and 'float16' in r['dtype']][0]
        bf16_result = [r for r in results if r['size'] == size and 'bfloat16' in r['dtype']][0]

        bf16_speedup = fp16_result['time_ms'] / bf16_result['time_ms']

        # Theoretical FP8 speedup (2x over BF16 on H100)
        theoretical_fp8_speedup = 2.0

        print(f"{size:<12} {bf16_speedup:<15.2f}x {theoretical_fp8_speedup:<15.2f}x")

    print(f"\n* Theoretical FP8 speedup assumes H100/H200 with FP8 Tensor Cores")
    print(f"  Actual FP8 requires CUTLASS/DeepGEMM or similar library\n")

    # Save to file
    if output_file:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to {output_file}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark FP8 vs BF16 GEMM')
    parser.add_argument('--sizes', type=str, default='1024,2048,4096',
                       help='Comma-separated matrix sizes (default: 1024,2048,4096)')
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

    sizes = [int(s) for s in args.sizes.split(',')]
    run_benchmark_suite(sizes, args.output)


if __name__ == '__main__':
    main()
