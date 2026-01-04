# Chapter 09: Sparse Attention Kernels

## Overview

This chapter explores efficient implementations of attention mechanisms in CUDA, from the classic O(L²) dense attention to modern sparse variants that achieve O(Lk) complexity. We'll implement FlashAttention, understand its memory-efficient tiling strategy, and dive into DeepSeek's innovative sparse attention approach.

## Learning Goals

By the end of this chapter, you will:

1. **Understand the attention mechanism** and its computational/memory bottlenecks
2. **Implement FlashAttention** using tiling and online softmax algorithms
3. **Master memory-efficient attention** through kernel fusion and SRAM optimization
4. **Explore sparse attention patterns** including sliding window and strided attention
5. **Implement DeepSeek's DSA** (Dynamic Sparse Attention) with lightning indexer
6. **Analyze complexity reduction** from O(L²) to O(Lk) and understand the trade-offs

## Key Concepts

### 1. Attention Mechanism Fundamentals

Standard scaled dot-product attention:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Computational complexity**: O(L² · d), where L is sequence length, d is head dimension

**Memory complexity**: O(L²) for storing attention matrix S = QK^T

**Bottleneck**: For long sequences (L > 4096), the L² term dominates both compute and memory.

### 2. FlashAttention: Tiling and Online Softmax

FlashAttention eliminates the need to materialize the full attention matrix by:

- **Tiling**: Breaking Q, K, V into blocks that fit in SRAM (shared memory)
- **Online softmax**: Computing softmax incrementally without storing full QK^T
- **Kernel fusion**: Fusing attention computation into a single kernel
- **Recomputation**: Trading compute for memory in backward pass

**Key innovation**: Reduces HBM access from O(L² · d) to O(L² · d² / M) where M is SRAM size.

**Reference**: [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)

### 3. FlashAttention-2 Improvements

FlashAttention-2 optimizes further:

- **Better parallelism**: Parallelizes over sequence length instead of batch/heads
- **Reduced synchronization**: Minimizes non-matmul FLOPs
- **Work partitioning**: Better GPU utilization through improved tiling

**Reference**: [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691)

### 4. Sparse Attention Patterns

Common sparsity patterns that reduce complexity to O(Lk) where k << L:

- **Local/Sliding Window**: Each token attends to k nearest neighbors
- **Strided/Dilated**: Attend to every k-th token
- **Block-sparse**: Predefined block patterns (used in Sparse Transformers)
- **Random**: Random subset of positions
- **Global tokens**: Some tokens attend to all positions

### 5. DeepSeek Sparse Attention (DSA)

DeepSeek-V3's innovation uses **dynamic** sparse attention:

- **Lightning Indexer**: O(1) index computation for sparse patterns
- **Top-k Selection**: Select k most relevant tokens based on queries
- **Adaptive sparsity**: Different layers use different sparsity patterns
- **Load balancing**: Efficient token distribution across experts (MoE context)

**Complexity**: O(L · d · k) where k is typically 128-512 (vs L = 32768+)

**Reference**: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)

### 6. Complexity Analysis: O(L²) to O(Lk)

**Dense Attention:**
- Compute: O(L² · d) for QK^T and O(L² · d) for softmax(QK^T)V
- Memory: O(L²) for attention matrix
- Example: L=32768, d=128 → 1B FLOPs per head

**Sparse Attention (k nearest neighbors):**
- Compute: O(L · k · d)
- Memory: O(L · k)
- Example: L=32768, k=512, d=128 → 16M FLOPs per head (60× reduction!)

**Trade-offs:**
- Sparse attention may miss long-range dependencies
- Quality depends on how well sparsity pattern matches data
- DeepSeek's dynamic selection helps maintain quality

## Prerequisites

Before starting this chapter, you should understand:

1. **Attention mechanism** - Transformers, self-attention, multi-head attention
2. **CUDA shared memory** - Tiling strategies (see Chapter 03)
3. **Matrix multiplication** - Optimized GEMM kernels (see Chapter 04)
4. **Memory hierarchies** - SRAM vs HBM bandwidth considerations
5. **PyTorch basics** - For reference implementations and testing

## Chapter Structure

### Examples

1. **01_attention_basics/** - Naive attention implementation (baseline)
2. **02_flash_attention_minimal/** - Educational FlashAttention (~100 lines)
3. **03_flash_attention_v2/** - Optimized FlashAttention-2 style kernel
4. **04_sparse_patterns/** - Various sparse attention patterns
5. **05_deepseek_sparse/** - DeepSeek DSA implementation

### Benchmarks

Comprehensive performance comparisons:
- Dense vs Flash vs Sparse attention
- Memory usage analysis
- Scaling behavior with sequence length

### Exercises

1. **Mini FlashAttention** - Implement simplified FlashAttention from scratch
2. **Custom Sparse Mask** - Design and implement your own sparsity pattern

## Key Papers and Resources

### FlashAttention
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) (Dao et al., 2022)
- [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691) (Dao, 2023)
- [flash-attention GitHub](https://github.com/Dao-AILab/flash-attention)
- [flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal) - Educational implementation

### Sparse Attention
- [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) (Child et al., 2019)
- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) (Beltagy et al., 2020)
- [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) (Zaheer et al., 2020)

### DeepSeek
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) (DeepSeek-AI, 2024)
- Focus on Section 3.2: Multi-head Latent Attention (MLA)
- Focus on Appendix: Sparse attention implementation details

### Additional Resources
- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) (Milakov & Gimelshein, 2018)
- [Self-attention Does Not Need O(n²) Memory](https://arxiv.org/abs/2112.05682) (Rabe & Staats, 2021)
- [EfficientAttention: Accurate and Efficient O(1) Attention](https://arxiv.org/abs/2312.08371)

## Memory Efficiency Deep Dive

### Standard Attention Memory Access

For sequence length L, head dimension d, batch size B, number of heads H:

```
HBM Reads/Writes per attention layer:
- Load Q, K, V: 3BHL d (read)
- Store S = QK^T: BHL² (write)
- Load S for softmax: BHL² (read)
- Store P = softmax(S): BHL² (write)
- Load P and V: BHL² + BHL d (read)
- Store O = PV: BHL d (write)

Total: O(BHL² + BHLd) ≈ O(BHL²) for large L
```

### FlashAttention Memory Access

```
HBM Reads/Writes:
- Load Q, K, V: 3BHL d (read)
- Store O: BHL d (write)

Total: O(BHLd)
```

**Speedup**: For L=4096, d=64, this is ~60× reduction in HBM traffic!

## Performance Expectations

### FlashAttention vs Standard Attention

On A100 GPU with L=2048, d=64, batch=16, heads=12:

| Implementation | Time (ms) | Memory (MB) | HBM Access (GB) |
|---------------|-----------|-------------|-----------------|
| PyTorch (naive) | 12.3 | 2048 | 8.4 |
| PyTorch (optimized) | 8.7 | 2048 | 6.1 |
| FlashAttention | 2.1 | 128 | 0.4 |
| FlashAttention-2 | 1.5 | 128 | 0.4 |

### Sparse Attention Scaling

Theoretical FLOPs for different approaches (L=32768, d=128):

| Method | FLOPs per token | Relative |
|--------|-----------------|----------|
| Dense | 1.07B | 1× |
| FlashAttention (dense) | 1.07B | 1× (same FLOPs, better memory) |
| Local (k=512) | 16.8M | 64× |
| DeepSeek DSA (k=512) | 16.8M | 64× |

## Getting Started

1. Start with `examples/01_attention_basics/` to understand the baseline
2. Work through `examples/02_flash_attention_minimal/` to grasp tiling concepts
3. Study `examples/05_deepseek_sparse/` for modern sparse attention
4. Complete exercises to solidify understanding
5. Run benchmarks to see performance differences

## Common Pitfalls

1. **Incorrect online softmax**: Forgetting to track max value leads to numerical instability
2. **Race conditions**: Improper synchronization when updating shared memory
3. **Bank conflicts**: Poor memory access patterns in shared memory
4. **Incorrect masking**: Causal mask implementation errors in autoregressive models
5. **Precision issues**: Using float16 without proper loss scaling

## Next Steps

After mastering sparse attention:
- **Chapter 10**: Advanced kernel fusion techniques
- **Chapter 11**: Multi-GPU attention (distributed transformers)
- **Chapter 12**: INT8/FP8 quantized attention kernels

## Questions to Consider

1. When does FlashAttention provide the most benefit?
2. What sparsity patterns work best for different tasks (NLP vs vision)?
3. How does DeepSeek's dynamic sparse attention maintain quality?
4. What are the trade-offs between recomputation and materialization?
5. How can we extend these techniques to cross-attention?

Happy coding!
