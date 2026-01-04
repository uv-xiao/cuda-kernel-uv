# Chapter 10 Quick Start Guide

## What You'll Learn

This chapter teaches you how to optimize Mixture of Experts (MoE) models for inference, focusing on the techniques from **SonicMoE** that achieve:
- **1.86x speedup** in end-to-end latency
- **45% reduction** in activation memory
- **186B tokens/day** on 8x H100 GPUs (vs 100B baseline)

## Prerequisites

```bash
# CUDA and PyTorch
CUDA >= 12.1
pip install torch>=2.1.0

# Optional: SonicMoE library
pip install git+https://github.com/mit-han-lab/SonicMoE.git

# Build tools for CUDA examples
cmake, nvcc
```

## 30-Minute Quick Start

### Step 1: Understand Basic MoE (5 min)

```bash
cd examples/01_moe_basics
python moe_layer.py
```

**Key Insight**: MoE layers route tokens to different experts, creating irregular workloads.

### Step 2: See the Grouped GEMM Problem (5 min)

```bash
cd ../02_grouped_gemm
mkdir build && cd build
cmake .. && make
./grouped_gemm_naive
```

**Key Insight**: Sequential expert processing wastes GPU capacity. Need parallel grouped GEMM.

### Step 3: Apply Tile-Aware Token Rounding (10 min)

```bash
cd ../../04_token_rounding
python token_rounding.py
python routing_comparison.py
```

**Key Insight**: Rounding token counts to tile boundaries (e.g., 128) gives perfect GPU utilization with minimal quality loss.

### Step 4: Run Complete Benchmark (10 min)

```bash
cd ../../benchmarks
python bench_moe_variants.py
python throughput.py --config deepseek-v3 --hardware h100 --num-gpus 8
```

**Key Insight**: Combined optimizations give 1.86x speedup and massive cost savings.

## Learning Path

### Beginner (4-6 hours)
1. **Example 01**: Basic MoE architecture (1 hour)
2. **Exercise 01**: Implement simple MoE forward pass (1 hour)
3. **Example 04**: Tile-aware token rounding (1 hour)
4. **Exercise 02**: Add load balancing (1-2 hours)

### Intermediate (8-12 hours)
1. **Example 02**: Grouped GEMM kernels (2-3 hours)
2. **Example 03**: IO overlap with TMA (2-3 hours)
3. **Example 05**: Using SonicMoE library (1 hour)
4. **Benchmarks**: Performance analysis (2-3 hours)

### Advanced (16+ hours)
1. **Example 06**: Full optimized MoE layer (4-6 hours)
2. Custom CUDA kernels for your use case (6-8 hours)
3. Multi-GPU extensions (4-6 hours)

## Key Concepts

### 1. MoE Architecture
```
Input → Router → Top-k Experts → Weighted Sum → Output
```
- Each token routed to k out of N experts
- Creates irregular, imbalanced workloads

### 2. Grouped GEMM
```python
# Instead of:
for expert in experts:
    output[expert] = expert(input[expert])  # Sequential!

# Use:
grouped_gemm(all_expert_inputs, all_expert_weights)  # Parallel!
```

### 3. Tile-Aware Token Rounding
```python
# Before: 142 tokens → 2 tiles (71% util)
# After:  128 tokens → 1 tile (100% util)

rounded = (count // TILE_SIZE) * TILE_SIZE
# Drop lowest-probability tokens, gain massive speedup
```

### 4. IO Overlap
```python
# Overlap expert weight loading with computation
async_load(expert_weights[i+1])  # Non-blocking
compute(expert[i])                # Parallel execution
```

## Common Issues

### Issue 1: Low GPU Utilization
**Symptom**: GPU usage < 50% during MoE forward pass
**Solution**: Use grouped GEMM to process experts in parallel

### Issue 2: High Memory Usage
**Symptom**: OOM errors with large batch sizes
**Solution**: Apply tile-aware rounding and activation checkpointing

### Issue 3: Load Imbalance
**Symptom**: Some experts get 10x more tokens than others
**Solution**: Add auxiliary load balancing loss during training

## Performance Targets

### DeepSeek-V3.2-Exp (685B, 256 experts)

| Metric | Target | How to Achieve |
|--------|--------|----------------|
| Latency | <25ms/layer | Grouped GEMM + Tile-aware + IO overlap |
| Memory | <25GB | Fused kernels + activation reuse |
| Throughput | >2000 tok/s | All optimizations combined |

### Your Custom Model

Estimate speedup:
```python
baseline_util = 0.45  # Typical for vanilla MoE
optimized_util = 0.78  # With SonicMoE optimizations
speedup = optimized_util / baseline_util  # ≈ 1.73x

# Add IO overlap benefit
final_speedup = speedup * 1.15  # ≈ 2.0x
```

## Verification Checklist

- [ ] Basic MoE forward pass works
- [ ] Grouped GEMM faster than sequential
- [ ] Tile-aware rounding improves utilization to >95%
- [ ] IO overlap reduces latency by 10-15%
- [ ] Load balancing keeps expert usage within 2x range
- [ ] End-to-end speedup > 1.5x

## Next Steps

After completing this chapter:

1. **Apply to Your Model**: Integrate optimizations into your codebase
2. **Multi-GPU Scaling**: Chapter 11 covers tensor/pipeline parallelism
3. **Quantization**: Chapter 12 adds INT8/FP8 support
4. **Production Deployment**: Monitor and tune for your workload

## Resources

### Papers
- SonicMoE: https://arxiv.org/abs/2512.14080
- GShard: https://arxiv.org/abs/2006.16668
- Switch Transformer: https://arxiv.org/abs/2101.03961

### Code
- SonicMoE: https://github.com/mit-han-lab/SonicMoE
- DeepSeek-V3: https://github.com/deepseek-ai/DeepSeek-V3

### Hardware Docs
- NVIDIA Hopper Architecture: https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/
- TMA Programming Guide: https://docs.nvidia.com/cuda/hopper-tuning-guide/

## Getting Help

1. **Check README.md**: Each example has detailed documentation
2. **Run Tests**: `python test.py` in exercise directories
3. **Review Solutions**: Compare with `solution.py` after attempting
4. **Profile**: Use `nsys` to visualize performance bottlenecks

## Estimated Time Investment

- **Quick Start**: 30 minutes
- **Complete Chapter**: 8-12 hours
- **Mastery**: 20-30 hours (including exercises and experimentation)

Good luck! The techniques in this chapter are used in production at DeepSeek, Mistral, and other leading AI labs.
