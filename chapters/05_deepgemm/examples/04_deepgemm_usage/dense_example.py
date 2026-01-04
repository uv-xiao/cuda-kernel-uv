#!/usr/bin/env python3
"""
DeepGEMM Dense FP8 GEMM Example

This example demonstrates:
1. Basic FP8 quantization
2. Dense FP8 GEMM computation
3. Performance comparison with BF16
"""

import torch
import time
import argparse

# Note: This is a reference implementation showing DeepGEMM API usage
# Actual DeepGEMM must be installed separately from:
# https://github.com/deepseek-ai/DeepSeek-V3/tree/main/inference/kernels/DeepGEMM

try:
    import deepgemm
    HAS_DEEPGEMM = True
except ImportError:
    HAS_DEEPGEMM = False
    print("Warning: DeepGEMM not installed. Using reference implementation.")
    print("Install from: https://github.com/deepseek-ai/DeepSeek-V3/")


# ============================================================================
# Reference FP8 Implementation (for demonstration)
# ============================================================================

class ReferenceFP8GEMM:
    """Reference implementation mimicking DeepGEMM API"""

    @staticmethod
    def quantize_fp8(tensor, dtype='e4m3', block_size=None):
        """Quantize BF16/FP32 tensor to FP8"""
        if block_size is None:
            # Per-tensor quantization
            max_val = tensor.abs().max()
            scale = max_val / 448.0 if dtype == 'e4m3' else max_val / 57344.0
            scale = max(scale, 1e-6)

            # Simulated FP8 (stored as int8)
            quantized = (tensor / scale).clamp(-448 if dtype == 'e4m3' else -57344,
                                                448 if dtype == 'e4m3' else 57344)
            quantized = quantized.to(torch.int8)

            return quantized, scale
        else:
            # Fine-grained quantization
            numel = tensor.numel()
            num_blocks = (numel + block_size - 1) // block_size

            tensor_flat = tensor.flatten()
            quantized = torch.zeros_like(tensor_flat, dtype=torch.int8)
            scales = torch.zeros(num_blocks, dtype=torch.float32, device=tensor.device)

            for i in range(num_blocks):
                start = i * block_size
                end = min(start + block_size, numel)
                block = tensor_flat[start:end]

                max_val = block.abs().max()
                scale = max_val / 448.0
                scale = max(scale, 1e-6)
                scales[i] = scale

                quantized[start:end] = (block / scale).clamp(-448, 448).to(torch.int8)

            return quantized.reshape(tensor.shape), scales

    @staticmethod
    def fp8_gemm(A_fp8, B_fp8, scale_A, scale_B, out_dtype=torch.bfloat16):
        """Simulated FP8 GEMM (actually uses BF16 for reference)"""
        # Dequantize
        A = A_fp8.to(torch.float32) * scale_A
        B = B_fp8.to(torch.float32) * scale_B

        # Compute GEMM
        C = torch.matmul(A, B)

        return C.to(out_dtype)


# ============================================================================
# Example Functions
# ============================================================================

def run_dense_gemm_example(M, N, K, use_fp8=True, warmup=10, iters=100):
    """Run dense GEMM with FP8 or BF16"""

    print(f"\n{'='*60}")
    print(f"Dense GEMM: {M}x{N}x{K}")
    print(f"Precision: {'FP8 E4M3' if use_fp8 else 'BF16'}")
    print(f"{'='*60}\n")

    # Initialize matrices
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    B = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')

    if use_fp8:
        # Quantize to FP8
        print("Quantizing tensors to FP8...")

        if HAS_DEEPGEMM:
            A_fp8, scale_A = deepgemm.quantize_fp8(A, dtype='e4m3')
            B_fp8, scale_B = deepgemm.quantize_fp8(B, dtype='e4m3')
        else:
            ref_api = ReferenceFP8GEMM()
            A_fp8, scale_A = ref_api.quantize_fp8(A, dtype='e4m3')
            B_fp8, scale_B = ref_api.quantize_fp8(B, dtype='e4m3')

        print(f"  Scale A: {scale_A:.6f}")
        print(f"  Scale B: {scale_B:.6f}")

        # Warmup
        for _ in range(warmup):
            if HAS_DEEPGEMM:
                C = deepgemm.fp8_gemm(A_fp8, B_fp8, scale_A, scale_B,
                                     out_dtype=torch.bfloat16)
            else:
                C = ref_api.fp8_gemm(A_fp8, B_fp8, scale_A, scale_B,
                                    out_dtype=torch.bfloat16)
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(iters):
            if HAS_DEEPGEMM:
                C = deepgemm.fp8_gemm(A_fp8, B_fp8, scale_A, scale_B,
                                     out_dtype=torch.bfloat16)
            else:
                C = ref_api.fp8_gemm(A_fp8, B_fp8, scale_A, scale_B,
                                    out_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        elapsed = time.time() - start

    else:
        # BF16 GEMM
        for _ in range(warmup):
            C = torch.matmul(A, B)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(iters):
            C = torch.matmul(A, B)
        torch.cuda.synchronize()
        elapsed = time.time() - start

    # Calculate metrics
    time_per_iter = elapsed / iters * 1000  # ms
    flops = 2 * M * N * K  # 2 * M * N * K for GEMM
    tflops = (flops / 1e12) / (time_per_iter / 1000)

    print(f"\nPerformance:")
    print(f"  Time per iteration: {time_per_iter:.3f} ms")
    print(f"  Throughput: {tflops:.1f} TFLOPS")

    # Verify correctness if FP8
    if use_fp8:
        C_ref = torch.matmul(A, B)
        error = (C - C_ref).abs().mean().item()
        max_error = (C - C_ref).abs().max().item()
        print(f"\nAccuracy (vs BF16):")
        print(f"  Mean absolute error: {error:.6f}")
        print(f"  Max absolute error: {max_error:.6f}")

    return time_per_iter, tflops


def compare_fp8_vs_bf16(M, N, K):
    """Compare FP8 and BF16 GEMM performance"""

    print(f"\n{'='*60}")
    print(f"FP8 vs BF16 Comparison")
    print(f"{'='*60}")

    # Run FP8
    time_fp8, tflops_fp8 = run_dense_gemm_example(M, N, K, use_fp8=True)

    # Run BF16
    time_bf16, tflops_bf16 = run_dense_gemm_example(M, N, K, use_fp8=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"FP8:  {time_fp8:.3f} ms ({tflops_fp8:.1f} TFLOPS)")
    print(f"BF16: {time_bf16:.3f} ms ({tflops_bf16:.1f} TFLOPS)")
    print(f"Speedup: {time_bf16/time_fp8:.2f}x")
    print(f"{'='*60}\n")


def test_fine_grained_scaling():
    """Demonstrate fine-grained scaling"""

    print(f"\n{'='*60}")
    print(f"Fine-Grained Scaling Example")
    print(f"{'='*60}\n")

    M, K = 2048, 2048
    block_size = 128

    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')

    # Add some outliers
    A[0, :10] = 100.0
    A[512, :10] = -100.0

    print(f"Matrix shape: {M}x{K}")
    print(f"Block size: {block_size}")
    print(f"Number of blocks: {(M*K + block_size - 1) // block_size}\n")

    # Per-tensor quantization
    ref_api = ReferenceFP8GEMM()
    A_fp8_coarse, scale_coarse = ref_api.quantize_fp8(A)
    A_dequant_coarse = A_fp8_coarse.to(torch.float32) * scale_coarse

    error_coarse = (A_dequant_coarse - A.to(torch.float32)).abs().mean().item()

    # Fine-grained quantization
    A_fp8_fine, scales_fine = ref_api.quantize_fp8(A, block_size=block_size)
    A_flat = A.flatten()
    A_fp8_flat = A_fp8_fine.flatten()

    A_dequant_fine = torch.zeros_like(A_flat, dtype=torch.float32)
    num_blocks = (A_flat.numel() + block_size - 1) // block_size

    for i in range(num_blocks):
        start = i * block_size
        end = min(start + block_size, A_flat.numel())
        A_dequant_fine[start:end] = A_fp8_flat[start:end].to(torch.float32) * scales_fine[i]

    A_dequant_fine = A_dequant_fine.reshape(A.shape)
    error_fine = (A_dequant_fine - A.to(torch.float32)).abs().mean().item()

    print("Per-Tensor Quantization:")
    print(f"  Scale: {scale_coarse:.6f}")
    print(f"  Mean absolute error: {error_coarse:.6f}\n")

    print("Fine-Grained Quantization:")
    print(f"  Scale range: [{scales_fine.min():.6f}, {scales_fine.max():.6f}]")
    print(f"  Mean absolute error: {error_fine:.6f}\n")

    print(f"Error reduction: {error_coarse/error_fine:.2f}x")


def main():
    parser = argparse.ArgumentParser(description='DeepGEMM Dense FP8 GEMM Example')
    parser.add_argument('--size', type=int, default=2048,
                       help='Matrix size (default: 2048)')
    parser.add_argument('--warmup', type=int, default=10,
                       help='Warmup iterations (default: 10)')
    parser.add_argument('--iters', type=int, default=100,
                       help='Benchmark iterations (default: 100)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare FP8 vs BF16')
    parser.add_argument('--fine-grained', action='store_true',
                       help='Test fine-grained scaling')

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        return

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability()}")

    if args.fine_grained:
        test_fine_grained_scaling()
    elif args.compare:
        compare_fp8_vs_bf16(args.size, args.size, args.size)
    else:
        run_dense_gemm_example(args.size, args.size, args.size,
                              warmup=args.warmup, iters=args.iters)


if __name__ == '__main__':
    main()
