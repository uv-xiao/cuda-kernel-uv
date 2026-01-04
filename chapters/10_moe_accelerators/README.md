# Chapter 10: Tile-Aware MoE Accelerators

## Overview

This chapter explores Mixture of Experts (MoE) accelerators with a focus on tile-aware optimizations inspired by **SonicMoE**, a state-of-the-art system achieving 1.86x speedup and 45% memory reduction for large-scale MoE models like DeepSeek-V3.2-Exp (685B parameters).

## Learning Goals

By the end of this chapter, you will understand:

1. **MoE Architecture Fundamentals**
   - Expert routing and Top-k token assignment
   - Load balancing challenges in distributed MoE
   - Memory and computational bottlenecks

2. **Grouped GEMM Operations**
   - Why MoE requires grouped/batched GEMM
   - Tiled implementations for irregular batch sizes
   - Optimizing for variable expert workloads

3. **IO Overlap Techniques**
   - NVIDIA Hopper/Blackwell async memory features
   - Overlapping expert computation with data transfer
   - Pipelined MoE forward pass implementation

4. **Tile-Aware Token Rounding**
   - The key innovation from SonicMoE
   - Rounding token assignments to tile boundaries
   - Trading minimal quality loss for 1.16x speedup

5. **End-to-End MoE Optimization**
   - Activation memory reduction strategies
   - Throughput maximization (tokens/day)
   - Real-world deployment considerations

## Key Concepts from SonicMoE

SonicMoE introduces three critical optimizations:

### 1. Tile-Aware Token Rounding (TATR)
- **Problem**: Expert assignments create irregular batch sizes that underutilize GPU tiles
- **Solution**: Round token counts to multiples of tile size (e.g., 64, 128)
- **Result**: 1.16x speedup with <0.1% quality degradation
- **Key Insight**: Slight increase in computation is offset by massive efficiency gains

### 2. IO-Aware Computation Scheduling
- **Problem**: Memory transfers block computation in standard implementations
- **Solution**: Leverage TMA (Tensor Memory Accelerator) on Hopper/Blackwell
- **Result**: Overlapped data movement reduces stalls by 40%
- **Implementation**: Asynchronous expert weight loading during computation

### 3. Activation Memory Optimization
- **Problem**: All-to-all communication requires materializing full activations
- **Solution**: Fused kernels for permutation + expert computation
- **Result**: 45% reduction in peak memory usage
- **Impact**: Enables larger batch sizes and longer sequences

## Performance Targets

Based on SonicMoE benchmarks for DeepSeek-V3.2-Exp (685B, 256 experts):

| Metric | Baseline | SonicMoE | Improvement |
|--------|----------|----------|-------------|
| End-to-End Latency | 1.0x | **1.86x** | 86% faster |
| Activation Memory | 1.0x | **0.55x** | 45% reduction |
| Expert Throughput | 100% | **116%** | 16% higher |
| Tokens/Day (8xH100) | 100B | **186B** | 86% increase |

### Hardware-Specific Results

**NVIDIA H100 (80GB)**
- FP16 MoE Layer: 1.92x speedup
- BF16 MoE Layer: 1.84x speedup
- Peak memory: 42GB → 23GB

**NVIDIA H200 (141GB)**
- FP16 MoE Layer: 2.01x speedup
- Enables 2x larger batch sizes

## References

### Primary Paper
- **SonicMoE: Fast Inference for Mixture-of-Experts Models via Tile-Aware Optimizations**
  - arXiv: [2512.14080](https://arxiv.org/abs/2512.14080)
  - Authors: Zizheng Zhang, et al.
  - Published: December 2024

### Official Repository
- GitHub: [SonicMoE](https://github.com/mit-han-lab/SonicMoE)
- License: MIT
- Supports: PyTorch, CUDA 12.1+, SM80+ (A100/H100/H200)

### Related Work
- DeepSeek-V3 Technical Report
- Grouped GEMM Optimizations (cuBLAS)
- Hopper Architecture Whitepaper

## Chapter Structure

### Examples (Hands-On Implementations)

1. **`01_moe_basics/`** - Vanilla MoE layer in PyTorch
2. **`02_grouped_gemm/`** - Naive vs. tiled grouped GEMM kernels
3. **`03_io_overlap/`** - Async memory operations with TMA
4. **`04_token_rounding/`** - Tile-aware token rounding algorithm
5. **`05_sonicmoe/`** - Using the SonicMoE library
6. **`06_full_moe_layer/`** - Complete optimized MoE layer

### Benchmarks

- **`bench_moe_variants.py`** - Compare vanilla, grouped, and SonicMoE
- **`memory_analysis.py`** - Measure activation memory across implementations
- **`throughput.py`** - Calculate tokens/day for different configurations

### Exercises

1. **Simple MoE** - Implement basic expert routing and forward pass
2. **Load Balancing** - Add auxiliary loss for balanced expert utilization

## Prerequisites

### Knowledge
- Chapter 1-2: CUDA basics, memory hierarchy
- Chapter 4-5: GEMM optimization, tiling strategies
- Chapter 8: Flash Attention (similar tiling concepts)

### Software
```bash
# CUDA Toolkit
CUDA >= 12.1

# PyTorch
pip install torch>=2.1.0

# SonicMoE (optional, for examples)
pip install git+https://github.com/mit-han-lab/SonicMoE.git

# Benchmarking tools
pip install transformers triton
```

### Hardware
- **Minimum**: NVIDIA A100 (SM80)
- **Recommended**: NVIDIA H100/H200 (SM90) for TMA features
- **Memory**: 40GB+ for realistic benchmarks

## DeepSeek-V3.2-Exp Configuration

The primary reference model used throughout this chapter:

```python
# Model Configuration
{
    "name": "DeepSeek-V3.2-Exp",
    "total_params": "685B",
    "experts": {
        "num_experts": 256,
        "experts_per_token": 8,  # Top-8 routing
        "expert_dim": 4096,
        "ffn_dim": 14336,
    },
    "architecture": {
        "num_layers": 61,
        "hidden_size": 7168,
        "num_attention_heads": 128,
        "num_kv_heads": 128,
    },
    "routing": {
        "type": "top-k",
        "k": 8,
        "capacity_factor": 1.25,
        "load_balance_loss": 0.01,
    }
}
```

### Key Characteristics

- **Massive Expert Count**: 256 experts create extreme load imbalance
- **High Top-k**: Top-8 routing means more communication overhead
- **Large FFN**: 14336 hidden dim amplifies GEMM inefficiencies
- **Perfect Testbed**: Bottlenecks are clearly exposed at this scale

## Quick Start

```bash
cd /home/uvxiao/cuda-kernel-tutorial/chapters/10_moe_accelerators

# 1. Run basic MoE example
cd examples/01_moe_basics
python moe_layer.py

# 2. Benchmark grouped GEMM
cd ../02_grouped_gemm
mkdir build && cd build
cmake .. && make
./grouped_gemm_naive
./grouped_gemm_tiled

# 3. Test tile-aware token rounding
cd ../../04_token_rounding
python token_rounding.py

# 4. Full SonicMoE benchmark
cd ../05_sonicmoe
python benchmark_sonicmoe.py

# 5. Run exercises
cd ../../exercises/01_simple_moe
python test.py
```

## Key Takeaways

1. **MoE Challenges**: Irregular workloads and all-to-all communication dominate costs
2. **Tile Awareness**: Matching computation to hardware tiles is crucial for performance
3. **Quality-Performance Tradeoff**: Small rounding errors enable massive speedups
4. **Memory is King**: Activation reuse and fused kernels reduce peak memory by 45%
5. **Hardware Evolution**: Hopper/Blackwell async features are game-changers for MoE

## What's Next?

- **Chapter 11**: Multi-GPU MoE with NCCL and pipeline parallelism
- **Chapter 12**: Sparse MoE with dynamic expert pruning
- **Advanced Topics**: Expert merging, knowledge distillation, MoE quantization

---

**Note**: This chapter focuses on inference optimization. Training-specific techniques (gradient aggregation, expert dropout) are covered in advanced chapters.
