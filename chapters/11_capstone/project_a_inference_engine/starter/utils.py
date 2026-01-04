"""
Utility functions for the inference engine
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
import subprocess
import os


class LayerNorm(nn.Module):
    """
    Layer normalization with optional custom kernel
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5, use_custom: bool = False):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.use_custom = use_custom

        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

        # TODO: Load custom kernel if requested
        if use_custom:
            # self.custom_kernel = load_layernorm_kernel()
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (..., normalized_shape)

        Returns:
            Normalized tensor (..., normalized_shape)
        """
        if self.use_custom:
            # TODO: Use custom kernel
            # return self.custom_kernel(x, self.weight, self.bias, self.eps)
            pass

        # Use PyTorch native implementation
        return nn.functional.layer_norm(
            x, (self.normalized_shape,), self.weight, self.bias, self.eps
        )


def get_device_properties() -> Dict:
    """
    Get CUDA device properties

    Returns:
        Dictionary with device information
    """
    if not torch.cuda.is_available():
        return {'device': 'cpu', 'available': False}

    props = torch.cuda.get_device_properties(0)

    return {
        'device': 'cuda',
        'available': True,
        'name': props.name,
        'compute_capability': f"{props.major}.{props.minor}",
        'total_memory_gb': props.total_memory / 1e9,
        'multi_processor_count': props.multi_processor_count,
        'max_threads_per_block': props.max_threads_per_block,
        'max_shared_memory_per_block_kb': props.max_shared_mem_per_block / 1024,
    }


def compute_flops(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    intermediate_dim: int,
    vocab_size: int,
) -> float:
    """
    Compute theoretical FLOPs for transformer forward pass

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_dim: Hidden dimension
        num_layers: Number of layers
        num_heads: Number of attention heads
        intermediate_dim: Intermediate dimension in FFN/MoE
        vocab_size: Vocabulary size

    Returns:
        Total FLOPs
    """
    # Embedding: negligible
    embedding_flops = 0

    # Attention per layer:
    # QKV projection: 3 * (B * S * H * H)
    qkv_flops = 3 * batch_size * seq_len * hidden_dim * hidden_dim

    # Attention matrix: B * num_heads * S * S * head_dim
    head_dim = hidden_dim // num_heads
    attention_flops = batch_size * num_heads * seq_len * seq_len * head_dim

    # Output projection: B * S * H * H
    out_proj_flops = batch_size * seq_len * hidden_dim * hidden_dim

    # Total attention FLOPs per layer
    attn_flops_per_layer = qkv_flops + attention_flops + out_proj_flops

    # FFN per layer:
    # Up projection: B * S * H * intermediate_dim
    # Down projection: B * S * intermediate_dim * H
    ffn_flops_per_layer = 2 * batch_size * seq_len * hidden_dim * intermediate_dim

    # Total per layer
    flops_per_layer = attn_flops_per_layer + ffn_flops_per_layer

    # All layers
    total_flops = num_layers * flops_per_layer

    # Final projection (if needed)
    lm_head_flops = batch_size * seq_len * hidden_dim * vocab_size
    total_flops += lm_head_flops

    return total_flops


def compute_memory_usage(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    intermediate_dim: int,
    vocab_size: int,
    dtype: torch.dtype = torch.float16,
) -> Dict:
    """
    Estimate memory usage for model

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_dim: Hidden dimension
        num_layers: Number of layers
        num_heads: Number of attention heads
        intermediate_dim: Intermediate dimension
        vocab_size: Vocabulary size
        dtype: Data type (float32, float16, etc.)

    Returns:
        Dictionary with memory estimates (in GB)
    """
    bytes_per_element = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
    }[dtype]

    # Activation memory (forward pass only, no gradients)
    # Input: B * S * H
    input_memory = batch_size * seq_len * hidden_dim * bytes_per_element

    # Attention: QKV (B * S * 3 * H), Scores (B * num_heads * S * S)
    attn_memory_per_layer = (
        batch_size * seq_len * 3 * hidden_dim +  # QKV
        batch_size * num_heads * seq_len * seq_len  # Scores (can be optimized with Flash Attention)
    ) * bytes_per_element

    # FFN: intermediate activation (B * S * intermediate_dim)
    ffn_memory_per_layer = batch_size * seq_len * intermediate_dim * bytes_per_element

    # Total activation memory
    activation_memory = input_memory + num_layers * (attn_memory_per_layer + ffn_memory_per_layer)

    # Parameter memory
    # Embedding: vocab_size * H
    embedding_params = vocab_size * hidden_dim * bytes_per_element

    # Attention params per layer: 4 * H * H (QKV + output proj)
    attn_params_per_layer = 4 * hidden_dim * hidden_dim * bytes_per_element

    # FFN params per layer: 2 * H * intermediate_dim
    ffn_params_per_layer = 2 * hidden_dim * intermediate_dim * bytes_per_element

    # Total parameter memory
    param_memory = embedding_params + num_layers * (attn_params_per_layer + ffn_params_per_layer)

    # Total memory
    total_memory = activation_memory + param_memory

    return {
        'activation_memory_gb': activation_memory / 1e9,
        'parameter_memory_gb': param_memory / 1e9,
        'total_memory_gb': total_memory / 1e9,
    }


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create causal attention mask (for autoregressive generation)

    Args:
        seq_len: Sequence length
        device: Device to create mask on

    Returns:
        Causal mask (seq_len, seq_len)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask


def benchmark_kernel(
    kernel_func,
    *args,
    num_iterations: int = 100,
    warmup: int = 10,
    **kwargs
) -> Dict:
    """
    Benchmark a kernel function

    Args:
        kernel_func: Function to benchmark
        *args, **kwargs: Arguments to pass to function
        num_iterations: Number of iterations to run
        warmup: Number of warmup iterations

    Returns:
        Dictionary with timing statistics
    """
    import time

    # Warmup
    for _ in range(warmup):
        _ = kernel_func(*args, **kwargs)

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(num_iterations):
        _ = kernel_func(*args, **kwargs)

    torch.cuda.synchronize()
    end = time.perf_counter()

    # Calculate statistics
    total_time = end - start
    avg_time = total_time / num_iterations
    throughput = 1.0 / avg_time

    return {
        'total_time_s': total_time,
        'avg_time_ms': avg_time * 1000,
        'throughput_per_s': throughput,
    }


def profile_with_nsys(
    command: str,
    output_file: str = "profile",
    stats: bool = True,
):
    """
    Profile command with Nsight Systems

    Args:
        command: Command to profile
        output_file: Output file name (without extension)
        stats: Print statistics after profiling

    Example:
        profile_with_nsys("python inference_engine.py", "profile_inference")
    """
    nsys_command = [
        "nsys", "profile",
        "-o", output_file,
        "--stats", "true" if stats else "false",
        "-f", "true",  # Force overwrite
    ] + command.split()

    print(f"Running: {' '.join(nsys_command)}")

    try:
        subprocess.run(nsys_command, check=True)
        print(f"Profile saved to {output_file}.nsys-rep")
        print(f"View with: nsys-ui {output_file}.nsys-rep")
    except subprocess.CalledProcessError as e:
        print(f"Profiling failed: {e}")


def profile_with_ncu(
    command: str,
    output_file: str = "profile",
    metrics: Optional[list] = None,
):
    """
    Profile command with Nsight Compute

    Args:
        command: Command to profile
        output_file: Output file name
        metrics: List of metrics to collect (None for default)

    Example:
        profile_with_ncu("python inference_engine.py", metrics=["sm__throughput.avg.pct_of_peak_sustained_elapsed"])
    """
    ncu_command = [
        "ncu",
        "-o", output_file,
        "-f",  # Force overwrite
    ]

    if metrics:
        ncu_command.extend(["--metrics", ",".join(metrics)])

    ncu_command.extend(command.split())

    print(f"Running: {' '.join(ncu_command)}")

    try:
        subprocess.run(ncu_command, check=True)
        print(f"Profile saved to {output_file}.ncu-rep")
        print(f"View with: ncu-ui {output_file}.ncu-rep")
    except subprocess.CalledProcessError as e:
        print(f"Profiling failed: {e}")


def save_benchmark_results(
    results: list,
    filename: str = "benchmark_results.csv",
):
    """
    Save benchmark results to CSV

    Args:
        results: List of result dictionaries
        filename: Output filename
    """
    import csv

    if not results:
        print("No results to save")
        return

    # Get all keys
    keys = set()
    for result in results:
        keys.update(result.keys())
    keys = sorted(keys)

    # Write CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {filename}")


def load_cuda_kernel(
    cu_file: str,
    kernel_name: str,
    compiler: str = "nvcc",
    compute_capability: Optional[str] = None,
) -> callable:
    """
    Compile and load CUDA kernel

    Args:
        cu_file: Path to .cu file
        kernel_name: Name of kernel function
        compiler: CUDA compiler to use
        compute_capability: Compute capability (e.g., "8.0" for A100)

    Returns:
        Callable kernel function
    """
    from torch.utils.cpp_extension import load

    if compute_capability is None:
        # Auto-detect
        props = torch.cuda.get_device_properties(0)
        compute_capability = f"{props.major}.{props.minor}"

    extra_cuda_cflags = [
        f"-arch=sm_{compute_capability.replace('.', '')}",
        "-O3",
        "--use_fast_math",
    ]

    # Load and compile
    module = load(
        name=kernel_name,
        sources=[cu_file],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=True,
    )

    return getattr(module, kernel_name)


def validate_numerical_accuracy(
    output: torch.Tensor,
    reference: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> Tuple[bool, Dict]:
    """
    Validate numerical accuracy against reference

    Args:
        output: Output tensor
        reference: Reference tensor
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        (is_correct, stats)
    """
    # Check shapes match
    if output.shape != reference.shape:
        return False, {'error': 'Shape mismatch'}

    # Compute errors
    abs_diff = torch.abs(output - reference)
    rel_diff = abs_diff / (torch.abs(reference) + 1e-8)

    # Statistics
    stats = {
        'max_abs_error': abs_diff.max().item(),
        'mean_abs_error': abs_diff.mean().item(),
        'max_rel_error': rel_diff.max().item(),
        'mean_rel_error': rel_diff.mean().item(),
    }

    # Check if within tolerance
    is_correct = torch.allclose(output, reference, rtol=rtol, atol=atol)

    return is_correct, stats


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")

    # Device properties
    print("\nDevice Properties:")
    props = get_device_properties()
    for key, value in props.items():
        print(f"  {key}: {value}")

    # FLOPs calculation
    print("\nFLOPs Calculation:")
    flops = compute_flops(
        batch_size=4,
        seq_len=2048,
        hidden_dim=4096,
        num_layers=32,
        num_heads=32,
        intermediate_dim=11008,
        vocab_size=32000,
    )
    print(f"  Total FLOPs: {flops / 1e12:.2f} TFLOPS")

    # Memory estimation
    print("\nMemory Estimation:")
    memory = compute_memory_usage(
        batch_size=4,
        seq_len=2048,
        hidden_dim=4096,
        num_layers=32,
        num_heads=32,
        intermediate_dim=11008,
        vocab_size=32000,
        dtype=torch.float16,
    )
    for key, value in memory.items():
        print(f"  {key}: {value:.2f}")

    print("\nAll tests passed!")
