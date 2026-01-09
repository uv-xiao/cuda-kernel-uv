# Chapter 12: LLM Serving Infrastructure Kernels

A deep dive into the kernel architecture of real-world LLM serving systems, using **mini-sglang** and **FlashInfer** as reference implementations.

## Learning Objectives

By the end of this chapter, you will:
- Understand how kernels integrate in production LLM serving pipelines
- Implement custom embedding and KV cache kernels
- Use FlashInfer for optimized attention computation
- Profile and optimize end-to-end serving performance
- Apply distributed communication patterns (NCCL)

## Prerequisites

- **Completed Chapters 1-10** (especially 09: Sparse Attention, 10: MoE)
- Understanding of transformer architecture
- Familiarity with PyTorch and CUDA
- Basic knowledge of tensor parallelism

## Hardware Requirements

- NVIDIA GPU: A100/H100 recommended (8GB+ VRAM minimum)
- Multi-GPU setup for distributed examples

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           LLM SERVING KERNEL PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                         MINI-SGLANG ARCHITECTURE                             │   │
│  │                                                                              │   │
│  │  Request → [Radix Cache] → [Index] → [Transformer Layers] → [LM Head]      │   │
│  │               (CPU)         (GPU)            (GPU)            (GPU)         │   │
│  │                                                                              │   │
│  │  Each Transformer Layer:                                                    │   │
│  │  ┌────────────────────────────────────────────────────────────────────────┐ │   │
│  │  │ QKV Proj → [RoPE] → [Store] → [Attention] → Proj → [RMSNorm] → FFN    │ │   │
│  │  │             (FI)     (Custom)    (FI/FA)             (FI)               │ │   │
│  │  │                                                                         │ │   │
│  │  │ FI = FlashInfer kernel                                                 │ │   │
│  │  │ FA = FlashAttention kernel                                             │ │   │
│  │  └────────────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                              │   │
│  │  [If TP > 1]: NCCL All-Reduce/All-Gather after attention and FFN           │   │
│  │                                                                              │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Chapter Contents

### Example 1: Mini-SGLang Custom Kernels

Learn about the custom CUDA kernels in mini-sglang:
- **Index Kernel**: Warp-level vectorized embedding lookup
- **Store Kernel**: Scatter-write to paged KV cache
- **PyNCCL**: NCCL wrapper for distributed communication

[→ Go to Example 1](examples/01_mini_sglang_kernels/)

### Example 2: FlashInfer Attention

Deep dive into FlashInfer's attention kernels:
- **BatchPrefillWithPagedKVCache**: Prefill phase attention
- **BatchDecodeWithPagedKVCache**: Decode phase attention
- **CUDA Graph support**: Low-latency decode batching

[→ Go to Example 2](examples/02_flashinfer_attention/)

### Example 3: KV Cache Management

Understand paged KV cache and the store kernel:
- Paged memory layout (LayerFirst vs PageFirst)
- Scatter-write optimization
- Radix cache for prefix sharing

[→ Go to Example 3](examples/03_kv_cache_management/)

### Example 4: Distributed Inference

Multi-GPU inference with tensor parallelism:
- NCCL all-reduce for attention/FFN
- NCCL all-gather for vocabulary parallelism
- Symmetric memory optimization

[→ Go to Example 4](examples/04_distributed_inference/)

### Exercises

1. **Custom Embedding Kernel**: Implement vocabulary masking for TP
2. **KV Store Optimization**: Fuse K+V copy with async operations

## Kernel Catalog

### Custom Kernels (mini-sglang)

| Kernel | Location | Purpose |
|--------|----------|---------|
| Index | `kernel/csrc/jit/index.cu` | Embedding lookup |
| Store | `kernel/csrc/jit/store.cu` | KV cache scatter |
| PyNCCL | `kernel/csrc/src/pynccl.cu` | NCCL wrapper |
| Radix | `kernel/csrc/src/radix.cpp` | Prefix matching (CPU) |

### FlashInfer Kernels

| Kernel | API | Purpose |
|--------|-----|---------|
| Prefill | `BatchPrefillWithPagedKVCacheWrapper` | Prefill attention |
| Decode | `BatchDecodeWithPagedKVCacheWrapper` | Decode attention |
| RMSNorm | `flashinfer.rmsnorm` | Layer normalization |
| RoPE | `flashinfer.apply_rope_with_cos_sin_cache_inplace` | Position embedding |

## Time Estimate

- Examples: 4-6 hours total
- Exercises: 4-6 hours total
- Total: 8-12 hours

## Key Resources

### Source Code
- [mini-sglang](https://github.com/sgl-project/mini-sglang) - Lightweight SGLang implementation
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) - GPU kernel library for LLM serving

### Documentation
- [FlashInfer API Docs](https://docs.flashinfer.ai/)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)

### Analysis Reports
- [Mini-SGLang Hands-On Learning Report](../../../../reports/implementations/mini-sglang-hands-on-learning.md)
- [Mini-SGLang Kernel Development Guide](../../../../reports/implementations/mini-sglang-kernel-dev-guide.md)
- [FlashInfer Kernel Development Guide](../../../../reports/implementations/flashinfer-kernel-dev-guide.md)

## Next Steps

After completing this chapter:
- **Chapter 11: Capstone Projects** - Build a complete inference engine
- Contribute optimizations back to mini-sglang or FlashInfer
