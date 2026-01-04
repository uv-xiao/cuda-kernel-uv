# Chapter 05: DeepGEMM & Advanced GEMM Patterns

## Overview

This chapter explores advanced GEMM (General Matrix Multiply) patterns critical for modern large language models, with a focus on FP8 quantization and Mixture-of-Experts (MoE) architectures. We study **DeepGEMM** from DeepSeek, a production-grade library that exemplifies clean, efficient GEMM kernel design.

DeepGEMM implements three fundamental GEMM patterns:
1. **Dense GEMM** - Standard matrix multiplication with FP8 support
2. **Grouped GEMM** - Batched GEMM for MoE layers
3. **Masked Grouped GEMM** - Grouped GEMM with dynamic token routing

## Learning Goals

By the end of this chapter, you will:

- Understand FP8 data formats (E4M3, E5M2) and their tradeoffs
- Implement quantization/dequantization with fine-grained scaling
- Design grouped GEMM kernels for MoE workloads
- Handle variable-size expert batches efficiently
- Apply DeepGEMM design principles to your own kernels
- Benchmark FP8 vs BF16 performance characteristics

## Why DeepGEMM?

DeepGEMM represents the "clean and efficient" design philosophy:

**Clean Design:**
- Minimal abstraction layers - easy to understand and modify
- Clear separation of concerns (data types, scaling, GEMM logic)
- Well-structured templates that expose key optimization points
- Comprehensive test coverage

**Efficient Implementation:**
- FP8 delivers ~2x throughput over BF16 on modern GPUs (H100, H200)
- Grouped GEMM eliminates padding overhead in MoE
- Fine-grained scaling maintains accuracy with minimal overhead
- Optimized for real-world LLM serving workloads

## Key Concepts

### 1. FP8 Data Types

Modern GPUs (NVIDIA Hopper+, AMD MI300+) support two FP8 formats:

| Format | Exponent | Mantissa | Range | Use Case |
|--------|----------|----------|-------|----------|
| E4M3 | 4 bits | 3 bits | [-448, 448] | Activations, weights |
| E5M2 | 5 bits | 2 bits | [-57344, 57344] | Gradients (wider range) |

**Key tradeoff:** E4M3 has better precision in normal range, E5M2 handles outliers better.

### 2. Fine-Grained Scaling

Instead of single scale factor per tensor, use per-block scaling:

```
Quantized = round(Original / scale_factor)
Dequantized = Quantized * scale_factor
```

Fine-grained scaling (e.g., per 128 elements) maintains accuracy while enabling FP8:
- Handles outliers within small regions
- Minimal memory overhead (~1% for scale factors)
- Hardware-friendly: scales stored in registers/shared memory

### 3. Grouped GEMM for MoE

Mixture-of-Experts layers route tokens to different expert networks:

```
Traditional approach (padding):
[Expert 0: 150 tokens] → pad to 256
[Expert 1: 80 tokens]  → pad to 256
[Expert 2: 200 tokens] → pad to 256
Total compute: 3 * 256 = 768 (28% waste)

Grouped GEMM (no padding):
Batch = [Expert 0, Expert 1, Expert 2]
Total compute: 150 + 80 + 200 = 430
```

**Grouped GEMM benefits:**
- Eliminates padding waste (20-40% in typical MoE)
- Single kernel launch for all experts
- Better GPU utilization on unbalanced loads

### 4. Masked Grouped GEMM

Extension of grouped GEMM with dynamic token masking:
- Support for top-k expert selection (k=1, 2, etc.)
- Handle variable expert assignments per token
- Efficient for sparse MoE patterns

## DeepGEMM Repository

- **GitHub:** https://github.com/deepseek-ai/DeepSeek-V3/tree/main/inference/kernels/DeepGEMM
- **Paper:** DeepSeek-V3 Technical Report (Section on Inference Optimization)
- **Key Files:**
  - `include/deepgemm/` - Core GEMM interfaces
  - `src/` - Kernel implementations
  - `python/` - PyTorch bindings

## Prerequisites

Before starting this chapter, you should:

1. **CUTLASS Fundamentals (Chapter 04):**
   - Understand CUTLASS architecture (threadblock, warp, thread tiles)
   - Know how to use CUTLASS templates for GEMM
   - Familiar with epilogue fusion patterns

2. **Memory Optimization:**
   - Shared memory tiling strategies
   - Cooperative loading patterns
   - Bank conflict avoidance

3. **Basic Linear Algebra:**
   - Matrix multiplication mechanics
   - Batched/grouped operations
   - Block matrix operations

4. **CUDA Programming:**
   - Template metaprogramming in CUDA
   - Warp-level primitives
   - Async copy (cp.async)

## Chapter Structure

### Examples

1. **01_fp8_basics/** - FP8 data type handling
   - FP8 format specifications
   - Conversion kernels
   - Range and precision analysis

2. **02_quantization/** - Quantization techniques
   - Per-tensor quantization
   - Fine-grained (per-block) scaling
   - Dynamic vs static quantization

3. **03_grouped_gemm/** - MoE GEMM patterns
   - Basic grouped GEMM
   - Variable expert sizes
   - Load balancing strategies

4. **04_deepgemm_usage/** - Using DeepGEMM library
   - Dense FP8 GEMM examples
   - MoE integration
   - Performance tuning

### Benchmarks

- FP8 vs BF16 performance comparison
- Grouped GEMM vs padded batched GEMM
- Scaling factor granularity impact
- MoE load balancing analysis

### Exercises

1. **Simple Grouped GEMM** - Implement basic grouped GEMM
2. **FP8 Quantization** - Add fine-grained scaling
3. **MoE Kernel** - Complete MoE forward pass with routing

## Performance Targets

On NVIDIA H100 (80GB SXM):

| Operation | Size | Precision | Target Throughput |
|-----------|------|-----------|-------------------|
| Dense GEMM | 4096x4096x4096 | FP8 | >800 TFLOPS |
| Dense GEMM | 4096x4096x4096 | BF16 | >400 TFLOPS |
| Grouped GEMM (8 experts) | 2048x2048x2048 avg | FP8 | >700 TFLOPS |

Your implementations should achieve:
- FP8: >85% of cuBLAS FP8 performance
- Grouped GEMM: >1.3x speedup vs padded approach
- Fine-grained scaling: <5% overhead vs uniform scaling

## Real-World Applications

DeepGEMM patterns are used in:

1. **LLM Inference:**
   - DeepSeek-V3 (671B parameters, 256 experts)
   - Mixtral (8x7B, 8x22B variants)
   - GPT-4 scale models

2. **Training:**
   - FP8 training with mixed precision
   - Expert parallelism in MoE
   - Gradient accumulation optimization

3. **Serving Systems:**
   - vLLM with FP8 support
   - TensorRT-LLM MoE kernels
   - Custom inference frameworks

## Additional Resources

### Papers
- "FP8 Formats for Deep Learning" (NVIDIA, 2022)
- "DeepSeek-V3 Technical Report" (DeepSeek, 2024)
- "Switch Transformers" (Google, 2021) - MoE architecture

### Documentation
- NVIDIA FP8 Programming Guide
- CUTLASS FP8 GEMM Documentation
- cuBLASLt Grouped GEMM API

### Code References
- DeepGEMM: https://github.com/deepseek-ai/DeepSeek-V3/
- CUTLASS FP8 Examples: https://github.com/NVIDIA/cutlass/tree/main/examples
- Megatron-LM FP8 Training: https://github.com/NVIDIA/Megatron-LM

## Design Philosophy: Clean and Efficient

DeepGEMM demonstrates key principles for production kernels:

### 1. Simplicity Over Abstraction
```cpp
// Good: Direct, understandable
template<typename Element, int BlockSize>
__global__ void quantize_kernel(Element* output, float* input, float* scales) {
    // Clear quantization logic
}

// Avoid: Over-engineered for this use case
template<typename PolicyType>
class QuantizationEngine {
    // Multiple inheritance, factories, etc.
};
```

### 2. Specialize, Don't Generalize Prematurely
- Start with common case (e.g., 2048x2048 tiles)
- Add specializations for edge cases as needed
- Profile before optimizing

### 3. Hardware-Aware Design
- Use Tensor Cores for FP8 (mma.sync)
- Leverage async copy for bandwidth
- Align to cache line and memory transaction sizes

### 4. Testability
- Unit tests for each component
- Numerical accuracy validation
- Performance regression tests

## Next Steps

After completing this chapter:
- Explore DeepSeek-V3 codebase for production patterns
- Implement custom quantization schemes (e.g., GPTQ, AWQ)
- Study advanced MoE routing strategies
- Build end-to-end LLM inference with FP8

Let's dive into FP8 fundamentals!
