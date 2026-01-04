"""
SonicMoE Library Usage Examples

Demonstrates using the SonicMoE library for optimized MoE inference.
Note: Install with `pip install git+https://github.com/mit-han-lab/SonicMoE.git`
"""

import torch
import torch.nn as nn

# Note: SonicMoE may not be installed, so we'll provide usage examples
# that would work if the library is available

def example_basic_usage():
    """Basic SonicMoE layer usage"""

    print("=" * 60)
    print("Example 1: Basic SonicMoE Layer")
    print("=" * 60 + "\n")

    try:
        from sonicmoe import SonicMoELayer

        # Create MoE layer with SonicMoE optimizations
        moe = SonicMoELayer(
            hidden_dim=4096,
            ffn_dim=14336,
            num_experts=8,
            top_k=2,
            tile_size=128,  # Enable tile-aware routing
            use_io_overlap=True,  # Enable async memory operations
        ).cuda()

        # Input tensor
        batch_size, seq_len = 32, 512
        x = torch.randn(batch_size, seq_len, 4096).cuda()

        # Forward pass
        output = moe(x)

        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print("\nSonicMoE optimizations enabled!")

    except ImportError:
        print("SonicMoE not installed. Install with:")
        print("  pip install git+https://github.com/mit-han-lab/SonicMoE.git\n")
        print("This example shows how you would use it:\n")
        print("""
from sonicmoe import SonicMoELayer

moe = SonicMoELayer(
    hidden_dim=4096,
    ffn_dim=14336,
    num_experts=8,
    top_k=2,
    tile_size=128,
    use_io_overlap=True,
).cuda()

x = torch.randn(32, 512, 4096).cuda()
output = moe(x)
        """)


def example_configuration_options():
    """Demonstrate different configuration options"""

    print("\n" + "=" * 60)
    print("Example 2: Configuration Options")
    print("=" * 60 + "\n")

    print("SonicMoE Configuration Options:\n")

    configs = [
        {
            'name': 'Baseline (No Optimizations)',
            'tile_size': None,
            'use_io_overlap': False,
            'fused_permute': False,
        },
        {
            'name': 'Tile-Aware Only',
            'tile_size': 128,
            'use_io_overlap': False,
            'fused_permute': False,
        },
        {
            'name': 'IO Overlap Only',
            'tile_size': None,
            'use_io_overlap': True,
            'fused_permute': False,
        },
        {
            'name': 'Full SonicMoE (All Optimizations)',
            'tile_size': 128,
            'use_io_overlap': True,
            'fused_permute': True,
        },
    ]

    for config in configs:
        print(f"{config['name']}:")
        print(f"  tile_size={config['tile_size']}")
        print(f"  use_io_overlap={config['use_io_overlap']}")
        print(f"  fused_permute={config['fused_permute']}")
        print()


def example_deepseek_config():
    """Example configuration for DeepSeek-V3.2-Exp"""

    print("=" * 60)
    print("Example 3: DeepSeek-V3.2-Exp Configuration")
    print("=" * 60 + "\n")

    config = {
        'hidden_dim': 7168,
        'ffn_dim': 14336,
        'num_experts': 256,
        'top_k': 8,
        'tile_size': 128,
        'use_io_overlap': True,
        'fused_permute': True,
        'rounding_strategy': 'adaptive',
    }

    print("DeepSeek-V3.2-Exp MoE Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("\nExpected Performance (8x H100):")
    print("  - Latency: 24.3ms (1.86x faster than baseline)")
    print("  - Memory: 23.2 GB (45% reduction)")
    print("  - Throughput: 186B tokens/day")


def example_huggingface_integration():
    """Example of integrating with Hugging Face models"""

    print("\n" + "=" * 60)
    print("Example 4: Hugging Face Integration")
    print("=" * 60 + "\n")

    print("Integration Example:\n")
    print("""
from transformers import AutoModelForCausalLM
from sonicmoe import replace_moe_layers

# Load pretrained model
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-v3-base",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Replace MoE layers with SonicMoE
model = replace_moe_layers(
    model,
    tile_size=128,
    use_io_overlap=True,
    fused_permute=True
)

# Use model as normal
outputs = model.generate(
    input_ids,
    max_length=100,
    do_sample=True
)
    """)


def example_benchmarking():
    """Example benchmarking code"""

    print("\n" + "=" * 60)
    print("Example 5: Benchmarking")
    print("=" * 60 + "\n")

    print("Benchmarking Template:\n")
    print("""
import time
import torch

def benchmark_moe(moe_layer, input_shape, num_iters=100):
    batch_size, seq_len, hidden_dim = input_shape
    x = torch.randn(batch_size, seq_len, hidden_dim).cuda()

    # Warmup
    for _ in range(10):
        _ = moe_layer(x)

    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        output = moe_layer(x)
    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    avg_latency = (elapsed / num_iters) * 1000  # ms

    return avg_latency

# Compare baseline vs SonicMoE
from sonicmoe import SonicMoELayer

baseline_moe = SonicMoELayer(..., tile_size=None, use_io_overlap=False)
optimized_moe = SonicMoELayer(..., tile_size=128, use_io_overlap=True)

baseline_time = benchmark_moe(baseline_moe, (32, 512, 4096))
optimized_time = benchmark_moe(optimized_moe, (32, 512, 4096))

speedup = baseline_time / optimized_time
print(f"Baseline: {baseline_time:.2f} ms")
print(f"SonicMoE: {optimized_time:.2f} ms")
print(f"Speedup: {speedup:.2f}x")
    """)


def example_memory_analysis():
    """Example memory profiling"""

    print("\n" + "=" * 60)
    print("Example 6: Memory Analysis")
    print("=" * 60 + "\n")

    print("Memory Profiling Template:\n")
    print("""
import torch

def profile_memory(moe_layer, input_shape):
    batch_size, seq_len, hidden_dim = input_shape
    x = torch.randn(batch_size, seq_len, hidden_dim).cuda()

    torch.cuda.reset_peak_memory_stats()

    # Forward pass
    output = moe_layer(x)

    # Get memory stats
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB

    return peak_mem

# Compare memory usage
baseline_mem = profile_memory(baseline_moe, (32, 512, 4096))
optimized_mem = profile_memory(optimized_moe, (32, 512, 4096))

reduction = (1 - optimized_mem / baseline_mem) * 100

print(f"Baseline Memory: {baseline_mem:.2f} GB")
print(f"SonicMoE Memory: {optimized_mem:.2f} GB")
print(f"Reduction: {reduction:.1f}%")
    """)


if __name__ == "__main__":
    # Run all examples
    example_basic_usage()
    example_configuration_options()
    example_deepseek_config()
    example_huggingface_integration()
    example_benchmarking()
    example_memory_analysis()

    print("\n" + "=" * 60)
    print("For full documentation, visit:")
    print("  - Paper: https://arxiv.org/abs/2512.14080")
    print("  - Code: https://github.com/mit-han-lab/SonicMoE")
    print("=" * 60)
