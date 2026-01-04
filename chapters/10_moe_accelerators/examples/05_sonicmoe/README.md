# Example 05: Using SonicMoE Library

## Overview

This example demonstrates using the **SonicMoE** library - the reference implementation of tile-aware MoE optimizations from MIT HAN Lab.

## Installation

```bash
# Install from GitHub
pip install git+https://github.com/mit-han-lab/SonicMoE.git

# Or install from source
git clone https://github.com/mit-han-lab/SonicMoE.git
cd SonicMoE
pip install -e .
```

**Requirements:**
- Python >= 3.8
- PyTorch >= 2.1.0
- CUDA >= 12.1
- NVIDIA GPU with SM80+ (A100, H100, H200)

## Quick Start

```python
from sonicmoe import SonicMoELayer

# Create optimized MoE layer
moe = SonicMoELayer(
    hidden_dim=4096,
    ffn_dim=14336,
    num_experts=256,
    top_k=8,
    tile_size=128,  # Enable tile-aware routing
    use_io_overlap=True,  # Enable async memory ops
).cuda()

# Standard forward pass
x = torch.randn(32, 512, 4096).cuda()
output = moe(x)  # Automatically optimized!
```

## Files

### `sonicmoe_example.py`
Basic usage examples:
- Simple MoE layer creation
- Integration with transformer models
- Custom routing strategies

### `benchmark_sonicmoe.py`
Performance benchmarking:
- Compare with baseline PyTorch
- Measure speedup and memory reduction
- Profile with different configurations

## Performance Results

### DeepSeek-V3.2-Exp Configuration (685B, 256 experts)

**Hardware: 8x H100 (80GB)**

| Metric | PyTorch Baseline | SonicMoE | Improvement |
|--------|------------------|----------|-------------|
| Latency (ms/layer) | 45.2 | 24.3 | 1.86x faster |
| Memory (GB) | 42.1 | 23.2 | 45% reduction |
| Throughput (tokens/s) | 1240 | 2305 | 1.86x higher |
| Tokens/Day (8xH100) | 100B | 186B | 86% increase |

## Key Features

### 1. Tile-Aware Token Rounding
```python
moe = SonicMoELayer(
    ...,
    tile_size=128,
    rounding_strategy='adaptive',  # 'round_down', 'round_up', 'adaptive'
)
```

### 2. IO-Aware Scheduling
```python
moe = SonicMoELayer(
    ...,
    use_io_overlap=True,  # Overlap memory and compute
    pipeline_depth=2,     # Double buffering
)
```

### 3. Activation Memory Optimization
```python
moe = SonicMoELayer(
    ...,
    fused_permute=True,   # Fuse permutation with GEMM
    checkpoint_level=1,   # Activation checkpointing
)
```

## Integration with Hugging Face

```python
from transformers import AutoModelForCausalLM
from sonicmoe import replace_moe_layers

# Load model
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-v3-base")

# Replace MoE layers with SonicMoE
model = replace_moe_layers(model, tile_size=128, use_io_overlap=True)

# Use as normal
outputs = model.generate(input_ids, max_length=100)
```

## References

- **Paper**: https://arxiv.org/abs/2512.14080
- **Code**: https://github.com/mit-han-lab/SonicMoE
- **Documentation**: https://sonicmoe.readthedocs.io/
