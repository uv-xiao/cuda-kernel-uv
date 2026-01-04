# Implementation Plan: CUDA Kernel Tutorial Repository

## Overview

This document outlines the detailed implementation plan for creating a comprehensive CUDA kernel development tutorial repository, progressing from fundamentals to state-of-the-art LLM kernels.

## Resources Collected

### Official Documentation
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [CUTLASS Documentation](https://docs.nvidia.com/cutlass/)
- [Triton Documentation](https://triton-lang.org/)

### Open Source Repositories
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) - FP8 GEMM kernels
- [SonicMoE](https://github.com/Dao-AILab/sonic-moe) - Tile-aware MoE optimization
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) - Memory-efficient attention
- [flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal) - Educational FA implementation
- [TileLang](https://github.com/tile-ai/tilelang) - High-level kernel DSL
- [Triton-Puzzles-Lite](https://github.com/SiriusNEO/Triton-Puzzles-Lite) - Triton exercises
- [LeetCUDA](https://github.com/xlite-dev/LeetCUDA) - CUDA exercises with 200+ kernels

### Tutorial Articles
- [siboehm's CUDA MatMul Optimization](https://siboehm.com/articles/22/CUDA-MMM) - Step-by-step optimization
- [Lei Mao's CUDA Matrix Multiplication](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/)
- [GPU MODE Lecture on Flash Attention](https://christianjmills.com/posts/cuda-mode-notes/lecture-012/)
- [Sebastian Raschka's DeepSeek Technical Tour](https://sebastianraschka.com/blog/2025/technical-deepseek.html)

### Research Papers
- DeepSeek-V3 Technical Report (arXiv:2412.19437)
- SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations (arXiv:2512.14080)
- FlashAttention and FlashAttention-2 papers

---

## Chapter-by-Chapter Plan

### Chapter 01: GPU & CUDA Introduction
**Goal**: Understand GPU architecture and write first CUDA kernel

**Content**:
1. GPU architecture overview (SMs, cores, memory hierarchy)
2. CUDA programming model (threads, blocks, grids)
3. First kernel: vector addition with error checking
4. Memory model: host vs device, cudaMalloc/cudaMemcpy

**Examples**:
- `01_hello_cuda/` - Hello World kernel
- `02_vector_add/` - Vector addition with timing
- `03_error_handling/` - Proper CUDA error checking

**Exercises**:
- Implement vector subtraction
- Implement element-wise multiply
- Experiment with different block sizes

**Reading**:
- CUDA Programming Guide: Chapters 1-3 (Programming Model)
- GPU Architecture whitepaper

---

### Chapter 02: CUDA Basics & Memory
**Goal**: Master thread indexing and memory types

**Content**:
1. Thread/block indexing (1D, 2D, 3D)
2. Global, shared, local, constant memory
3. Synchronization (`__syncthreads`)
4. Memory access patterns

**Examples**:
- `01_2d_indexing/` - 2D grid indexing
- `02_image_processing/` - 2D convolution kernel
- `03_reduction/` - Parallel reduction
- `04_histogram/` - Atomic histogram

**Exercises**:
- Implement matrix transpose (naive)
- Implement parallel prefix sum
- Optimize histogram with shared memory

**Performance Targets**:
- Reduction: > 80% memory bandwidth utilization
- Histogram: Compare atomic vs shared memory approaches

---

### Chapter 03: Profiling & Optimization
**Goal**: Use profilers to diagnose and fix bottlenecks

**Content**:
1. Nsight Compute: metrics, roofline analysis
2. Nsight Systems: timeline analysis
3. Memory coalescing
4. Bank conflicts in shared memory
5. Occupancy optimization

**Examples**:
- `01_matmul_naive/` - Naive matrix multiply
- `02_matmul_coalesced/` - With memory coalescing
- `03_matmul_smem/` - With shared memory tiling
- `04_matmul_optimized/` - Full optimization path

**Exercises**:
- Profile and optimize matrix transpose
- Identify and fix bank conflicts
- Achieve >70% of cuBLAS performance for SGEMM

**Profiler Walkthrough**:
- Interpreting SOL (speed of light) metrics
- Reading roofline charts
- Identifying stall reasons

---

### Chapter 04: CUTLASS and CuTe/CuteDSL
**Goal**: Use CUTLASS for high-performance GEMM

**Content**:
1. CUTLASS architecture overview
2. CuTe abstractions: Layout, Tensor, TiledCopy, TiledMMA
3. CuteDSL (Python) vs CuTe C++
4. Tensor Core usage

**Examples**:
- `01_cutlass_gemm/` - Basic CUTLASS GEMM
- `02_cute_layout/` - CuTe Layout examples
- `03_cute_gemm/` - GEMM using CuTe primitives
- `04_tensor_core/` - WMMA and MMA instructions
- `05_cutedsl_gemm/` - Python DSL approach

**Exercises**:
- Implement batched GEMM
- Implement grouped GEMM
- Compare FP16 vs FP32 vs TF32 performance

**Performance Targets**:
- Achieve >90% of cuBLAS performance
- Understand TFLOPs vs memory bandwidth tradeoffs

---

### Chapter 05: DeepGEMM & Advanced GEMM
**Goal**: Understand FP8 GEMM and MoE GEMM patterns

**Content**:
1. FP8 data types and quantization
2. DeepGEMM architecture
3. Dense GEMM with fine-grained scaling
4. Grouped GEMM for MoE
5. Masked grouped GEMM for inference

**Examples**:
- `01_fp8_basics/` - FP8 data handling
- `02_deepgemm_dense/` - Dense FP8 GEMM
- `03_deepgemm_grouped/` - Grouped GEMM for MoE
- `04_benchmark/` - Performance comparison

**Exercises**:
- Compare FP8 vs BF16 accuracy/performance
- Implement simple grouped GEMM
- Profile DeepGEMM kernels

---

### Chapter 06: Advanced CUDA Features
**Goal**: Master advanced CUDA programming constructs

**Content**:
1. Cooperative groups
2. Warp-level primitives (shuffle, vote, match)
3. Dynamic parallelism
4. Persistent kernels
5. CUDA Graphs

**Examples**:
- `01_warp_primitives/` - Shuffle and vote
- `02_cooperative_groups/` - Grid-wide sync
- `03_reduction_advanced/` - Warp-optimized reduction
- `04_scan/` - Parallel prefix scan
- `05_cuda_graphs/` - Graph-based execution

**Exercises**:
- Implement warp-level reduction
- Implement block-level scan using cooperative groups
- Create CUDA graph for multi-kernel pipeline

---

### Chapter 07: Triton Kernel Design
**Goal**: Write efficient GPU kernels in Triton

**Content**:
1. Triton programming model
2. Block-based computation
3. Autotuning
4. Fusion patterns

**Examples**:
- `01_vector_add/` - First Triton kernel
- `02_softmax/` - Softmax implementation
- `03_matmul/` - Blocked matrix multiply
- `04_fused_attention/` - Fused attention kernel
- `05_autotuning/` - Performance tuning

**Exercises**:
- Triton puzzles (inspired by Triton-Puzzles-Lite)
- Implement fused LayerNorm + Linear
- Compare Triton vs CUDA performance

---

### Chapter 08: TileLang & High-Level DSLs
**Goal**: Use high-level DSLs for kernel development

**Content**:
1. TileLang overview and philosophy
2. Tile-centric decomposition
3. Memory management (shared, fragments)
4. Pipeline stages

**Examples**:
- `01_tilelang_gemm/` - Basic GEMM in TileLang
- `02_flash_attention/` - Attention in TileLang
- `03_mla_decoding/` - MLA implementation
- `04_comparison/` - TileLang vs Triton vs CUDA

**Exercises**:
- Implement linear attention in TileLang
- Explore pipelining configurations
- Profile generated kernels

---

### Chapter 09: Sparse Attention Kernels
**Goal**: Implement efficient sparse attention for LLMs

**Content**:
1. Dense attention review and FlashAttention
2. Sparse attention patterns (local, strided, learned)
3. DeepSeek Sparse Attention (DSA)
4. Lightning indexer and token selection

**Examples**:
- `01_flash_attention_minimal/` - Simplified FA implementation
- `02_flash_attention_v2/` - Optimized FA-2
- `03_sparse_patterns/` - Various sparsity patterns
- `04_deepseek_sparse/` - DSA implementation

**Exercises**:
- Implement mini sparse attention kernel
- Benchmark sparse vs dense attention
- Implement top-k token selection

**Performance Targets**:
- Understand O(L²) to O(Lk) complexity reduction
- Measure memory savings vs dense attention

---

### Chapter 10: Tile-Aware MoE Accelerators
**Goal**: Optimize MoE forward/backward passes

**Content**:
1. MoE architecture review
2. Grouped GEMM for expert computation
3. IO-aware optimization
4. Tile-aware token rounding
5. SonicMoE techniques

**Examples**:
- `01_moe_basics/` - Simple MoE implementation
- `02_grouped_gemm/` - Efficient expert computation
- `03_io_overlap/` - Memory IO optimization
- `04_token_rounding/` - Tile-aware routing
- `05_sonicmoe/` - Full SonicMoE usage

**Exercises**:
- Compare naive vs tile-aware MoE
- Implement expert routing with load balancing
- Profile memory and compute efficiency

**Performance Targets**:
- Understand 1.86x throughput improvement potential
- Achieve 45% memory reduction

---

### Chapter 11: Capstone Projects
**Goal**: Build complete LLM inference components

**Projects**:

**Project A: Mini LLM Inference Engine**
- Custom attention kernel (FA or sparse)
- Custom MoE layer with optimized GEMM
- End-to-end inference pipeline

**Project B: Kernel Comparison Study**
- Implement same operation in CUDA, Triton, TileLang, CUTLASS
- Profile and compare
- Document tradeoffs

**Project C: Custom Sparse Attention**
- Design custom sparsity pattern
- Implement efficient kernel
- Benchmark on real sequences

**Deliverables**:
- Working code with tests
- Profiling report
- Performance comparison chart
- Lessons learned document

---

## Repository Structure

```
cuda-kernel-tutorial/
├── README.md
├── CLAUDE.md
├── PLAN.md
├── LICENSE
├── .gitignore
├── CMakeLists.txt
├── requirements.txt
├── setup.sh
├── chapters/
│   ├── 01_introduction/
│   ├── 02_cuda_basics/
│   ├── 03_profiling/
│   ├── 04_cutlass_cute/
│   ├── 05_deepgemm/
│   ├── 06_advanced_cuda/
│   ├── 07_triton/
│   ├── 08_tilelang/
│   ├── 09_sparse_attention/
│   ├── 10_moe_accelerators/
│   └── 11_capstone/
├── common/
│   ├── include/
│   │   ├── cuda_utils.cuh
│   │   ├── timer.cuh
│   │   └── check.cuh
│   ├── src/
│   └── python/
│       └── utils.py
├── benchmarks/
└── tests/
```

---

## Build & Test Strategy

### CMake Configuration
- Support CUDA 11.8+ and 12.x
- Target architectures: sm_70, sm_80, sm_89, sm_90
- Optional dependencies: CUTLASS, cuBLAS, cuDNN

### Python Environment
- PyTorch 2.0+
- Triton
- TileLang
- NumPy, pytest

### Testing
- Numerical correctness tests
- Performance regression tests
- CI-compatible test scripts

---

## Execution Timeline

The chapters will be generated in parallel where possible:
- **Wave 1**: Chapters 01, 02, 06 (foundational CUDA)
- **Wave 2**: Chapters 03, 04, 05 (profiling and CUTLASS)
- **Wave 3**: Chapters 07, 08 (Triton and TileLang)
- **Wave 4**: Chapters 09, 10 (LLM kernels)
- **Wave 5**: Chapter 11 (Capstone) + testing

---

## Quality Standards

1. All code compiles and runs
2. Each example includes expected output
3. Each exercise has working solution
4. Performance targets are realistic and measured
5. Documentation cites authoritative sources
