# Project A: Mini LLM Inference Engine

Build a minimal but functional LLM inference engine with optimized CUDA kernels for critical operations. This project simulates building production-grade inference infrastructure.

## Project Overview

**Goal**: Implement an end-to-end transformer inference pipeline optimized for small to medium language models (1B-7B parameters).

**Duration**: 2-3 weeks

**Difficulty**: Advanced

## Learning Objectives

By completing this project, you will:
- Integrate multiple kernel types into a cohesive system
- Understand the full inference pipeline for transformer models
- Optimize end-to-end latency and throughput
- Benchmark and profile real-world workloads
- Make tradeoffs between memory and compute

## Problem Statement

Modern LLM inference requires highly optimized kernels for attention and feedforward operations. Your task is to build a mini inference engine that can:

1. Execute multi-head attention with Flash Attention or sparse attention
2. Run Mixture of Experts (MoE) layers with optimized routing and GEMM
3. Complete end-to-end inference for a simple transformer model
4. Achieve competitive performance compared to existing frameworks

## Architecture

Your inference engine should support this model architecture:

```
Input (batch_size, seq_len, hidden_dim)
  |
  v
[Multi-Head Attention]
  - Query/Key/Value projection (GEMM)
  - Attention computation (Flash Attention or Sparse)
  - Output projection (GEMM)
  |
  v
[Add & Norm]
  |
  v
[MoE Layer]
  - Router (TopK selection)
  - Expert GEMMs (batched)
  - Output combination
  |
  v
[Add & Norm]
  |
  v
Output (batch_size, seq_len, hidden_dim)
```

## Required Components

### 1. Attention Kernel (40% of grade)

Implement ONE of the following:

#### Option A: Flash Attention
- Implement Flash Attention v2 algorithm
- Support multi-head attention
- Handle variable sequence lengths
- Optimize memory access patterns

**Target Performance**:
- Achieve >70% of reference Flash Attention throughput
- Support sequence lengths up to 8192
- Memory usage O(N) instead of O(N^2)

#### Option B: Sparse Attention
- Implement a sparse attention pattern (block-sparse, strided, etc.)
- Efficient sparse kernel using block-based approach
- Support variable sparsity ratios

**Target Performance**:
- Achieve >60% of dense attention throughput at 50% sparsity
- Support sequence lengths up to 16384
- Memory usage proportional to sparsity

### 2. MoE Layer (40% of grade)

Implement a Mixture of Experts layer with:

- **Router**: TopK expert selection per token
- **Expert GEMMs**: Batched matrix multiplications
- **Combine**: Weighted combination of expert outputs

**Components**:
```python
class MoELayer:
    def __init__(self, hidden_dim, num_experts, top_k):
        # Initialize router and experts

    def forward(self, x):
        # 1. Route tokens to experts (TopK)
        # 2. Batched GEMM for each expert
        # 3. Combine expert outputs
        return output
```

**Target Performance**:
- Router overhead <5% of total MoE time
- Expert GEMMs achieve >75% of cuBLAS throughput
- Support up to 64 experts with top-k=2 or top-k=8

### 3. End-to-End Pipeline (20% of grade)

Integrate components into a complete inference pipeline:

```python
class InferenceEngine:
    def __init__(self, config):
        # Initialize all layers

    def forward(self, input_ids):
        # Run complete forward pass
        # Return logits
```

**Requirements**:
- Support batch sizes 1, 4, 16, 32
- Support sequence lengths 128, 512, 2048, 8192
- Proper memory management (no leaks)
- Numerical correctness validation

## Deliverables

### 1. Code Implementation

**Directory Structure**:
```
project_a_inference_engine/
├── src/
│   ├── attention.py          # Your attention implementation
│   ├── moe.py                 # Your MoE implementation
│   ├── inference_engine.py   # Main engine
│   └── kernels/
│       ├── attention_kernel.cu
│       ├── moe_kernel.cu
│       └── ...
├── tests/
│   ├── test_attention.py
│   ├── test_moe.py
│   └── test_end_to_end.py
├── benchmarks/
│   ├── benchmark_attention.py
│   ├── benchmark_moe.py
│   └── benchmark_engine.py
├── build.sh
└── README.md
```

**Code Quality Requirements**:
- All kernels compile without warnings
- Python bindings working correctly
- Comprehensive error handling
- Clear code documentation

### 2. Test Suite

Implement tests for:
- **Correctness**: Compare against PyTorch reference
- **Edge cases**: Empty sequences, batch size 1, max sequence length
- **Numerical stability**: Check for NaN/Inf values
- **Memory**: No CUDA memory leaks

**Acceptance Criteria**:
- All tests pass
- Numerical error <1e-3 relative to reference
- No memory leaks detected

### 3. Benchmarks

Provide comprehensive benchmarks:

**Attention Benchmarks**:
```
Sequence Lengths: [128, 512, 1024, 2048, 4096, 8192]
Batch Sizes: [1, 4, 16, 32]
Num Heads: [8, 16, 32]
```

**MoE Benchmarks**:
```
Num Experts: [8, 16, 32, 64]
Top-K: [2, 4, 8]
Hidden Dims: [2048, 4096, 8192]
Batch Sizes: [1, 4, 16, 32]
```

**End-to-End Benchmarks**:
```
Model Configs:
- Small: 12 layers, 768 hidden, 12 heads
- Medium: 24 layers, 1024 hidden, 16 heads
- Large: 32 layers, 4096 hidden, 32 heads
```

**Metrics to Report**:
- Throughput (tokens/second)
- Latency (ms per forward pass)
- Memory usage (GB)
- FLOPS utilization (% of peak)
- Bandwidth utilization (% of peak)

### 4. Technical Report

Write a 3-4 page report covering:

#### Introduction
- Problem statement
- Architecture overview
- Design goals

#### Implementation Details
- Attention kernel design
  - Algorithm choice and justification
  - Memory layout
  - Optimization techniques
- MoE layer design
  - Routing strategy
  - Expert parallelization
  - Load balancing considerations
- Integration challenges

#### Performance Analysis
- Benchmark results with visualizations
- Comparison with baselines:
  - PyTorch native attention
  - Reference Flash Attention
  - cuBLAS for GEMMs
- Profiling insights:
  - Kernel-level bottlenecks
  - Memory bandwidth analysis
  - Compute utilization
- End-to-end performance breakdown

#### Lessons Learned
- What worked well
- What was challenging
- Unexpected findings
- Future optimization opportunities

#### Appendix
- Profiling screenshots
- Detailed benchmark tables
- Build instructions

## Evaluation Rubric

### Correctness (30%)

| Score | Criteria |
|-------|----------|
| 27-30 | All tests pass, numerical error <1e-4, handles all edge cases |
| 24-26 | Minor numerical issues (<1e-3), handles most edge cases |
| 21-23 | Some tests fail, numerical error <1e-2, basic cases work |
| <21   | Significant correctness issues |

### Performance (30%)

**Attention Kernel**:
| Score | Performance vs Reference |
|-------|-------------------------|
| 14-15 | >80% throughput |
| 12-13 | 60-80% throughput |
| 10-11 | 40-60% throughput |
| <10   | <40% throughput |

**MoE Layer**:
| Score | Performance vs cuBLAS |
|-------|----------------------|
| 14-15 | >75% throughput |
| 12-13 | 60-75% throughput |
| 10-11 | 40-60% throughput |
| <10   | <40% throughput |

### Code Quality (20%)

| Score | Criteria |
|-------|----------|
| 18-20 | Excellent structure, clear comments, no code smells |
| 16-17 | Good organization, adequate documentation |
| 14-15 | Acceptable structure, minimal documentation |
| <14   | Poor organization or documentation |

### Analysis & Documentation (20%)

| Score | Criteria |
|-------|----------|
| 18-20 | Thorough analysis, clear visualizations, insightful discussion |
| 16-17 | Good analysis, adequate visualizations, solid discussion |
| 14-15 | Basic analysis, minimal visualizations, superficial discussion |
| <14   | Incomplete or unclear documentation |

## Getting Started

### Prerequisites

```bash
# Required
CUDA Toolkit >= 11.8
Python >= 3.8
PyTorch >= 2.0
NumPy, pytest, matplotlib

# Optional but recommended
Nsight Systems
Nsight Compute
cuBLAS (included in CUDA Toolkit)
```

### Setup

1. Review the starter code:
```bash
cd starter/
python inference_engine.py --help
```

2. Understand the interfaces:
```python
# Read attention.py
# Read moe.py
# Read utils.py
```

3. Run baseline tests:
```bash
pytest tests/ -v
```

4. Start with attention kernel:
```bash
# Implement attention_kernel.cu
# Test with test_attention.py
# Benchmark with benchmark_attention.py
```

### Development Workflow

**Phase 1: Attention (Week 1)**
- Day 1-2: Understand Flash Attention algorithm / Choose sparse pattern
- Day 3-5: Implement naive version, validate correctness
- Day 6-7: Optimize and profile

**Phase 2: MoE (Week 2)**
- Day 1-2: Implement router and basic expert dispatch
- Day 3-4: Optimize expert GEMMs (use CUTLASS or cuBLAS)
- Day 5: Optimize routing and combination
- Day 6-7: Profile and optimize

**Phase 3: Integration (Week 3)**
- Day 1-2: Integrate components, end-to-end testing
- Day 3-4: Final optimizations, comprehensive benchmarks
- Day 5-7: Write report, prepare presentation

## Optimization Tips

### Attention Optimization
1. **Memory Access**: Coalesce global memory access, maximize L2 reuse
2. **Shared Memory**: Use shared memory for Q, K, V tiles
3. **Register Blocking**: Keep partial results in registers
4. **Occupancy**: Balance shared memory usage vs occupancy
5. **Sequence Length**: Handle non-power-of-2 lengths efficiently

### MoE Optimization
1. **Routing**: Use parallel reduction for TopK
2. **Batching**: Group tokens going to same expert
3. **Load Balancing**: Consider auxiliary loss to balance expert usage
4. **GEMM**: Leverage CUTLASS or cuBLAS for expert computations
5. **Memory**: Minimize data movement between routing and expert execution

### Integration Optimization
1. **Kernel Fusion**: Fuse small kernels (Add, Norm) with main operations
2. **Memory Management**: Reuse buffers across layers
3. **Streams**: Use CUDA streams for overlap
4. **Profiling**: Focus on kernels taking >5% of total time

## Common Pitfalls

1. **Not Validating Correctness First**: Always validate before optimizing
2. **Premature Optimization**: Profile before optimizing
3. **Ignoring Memory**: Memory bandwidth is often the bottleneck
4. **Not Testing Edge Cases**: Handle batch_size=1, seq_len not power of 2
5. **Hardcoding Sizes**: Make kernels flexible for different sizes
6. **Memory Leaks**: Always free CUDA memory
7. **Numerical Instability**: Use proper scaling for attention softmax

## Baseline Performance Targets

### H100 GPU (80GB)

**Flash Attention (seq_len=2048, batch=16, heads=16, dim=64)**:
- Target: >250 TFLOPS (>70% of reference Flash Attention)
- Reference Flash Attention: ~350 TFLOPS
- PyTorch native: ~150 TFLOPS

**MoE (hidden=4096, num_experts=32, top_k=2, batch=16, seq_len=2048)**:
- Target: >600 TFLOPS (>75% of cuBLAS)
- cuBLAS baseline: ~800 TFLOPS

**End-to-End (Medium model, batch=8, seq_len=2048)**:
- Target latency: <50ms per forward pass
- Target throughput: >150 tokens/second

### A100 GPU (40GB)

**Flash Attention (seq_len=2048, batch=16, heads=16, dim=64)**:
- Target: >180 TFLOPS
- Reference: ~250 TFLOPS

**MoE (hidden=4096, num_experts=32, top_k=2)**:
- Target: >450 TFLOPS

## Resources

### Papers
- [Flash Attention](https://arxiv.org/abs/2205.14135)
- [Flash Attention 2](https://arxiv.org/abs/2307.08691)
- [Switch Transformers (MoE)](https://arxiv.org/abs/2101.03961)
- [Efficient Sparse Attention](https://arxiv.org/abs/1904.10509)

### Code References
- [Flash Attention Official Implementation](https://github.com/Dao-AILab/flash-attention)
- [PyTorch Attention](https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py)
- [Megatron-LM MoE](https://github.com/NVIDIA/Megatron-LM)

### Tools
- Nsight Systems for timeline profiling
- Nsight Compute for kernel analysis
- CUDA-GDB for debugging
- Compute Sanitizer for memory checking

## FAQ

**Q: Can I use external libraries like CUTLASS for GEMM?**
A: Yes! The focus is on the attention and MoE logic, not reimplementing GEMM from scratch.

**Q: Do I need to implement both Flash Attention AND sparse attention?**
A: No, choose ONE for the attention component.

**Q: How do I validate correctness?**
A: Compare against PyTorch reference implementations with small random inputs.

**Q: What if I can't achieve the target performance?**
A: Document your attempts and analyze why. Understanding bottlenecks is valuable even if you don't hit targets.

**Q: Can I use multiple GPUs?**
A: For this project, focus on single-GPU optimization. Multi-GPU is out of scope.

## Support

If you get stuck:
1. Review Chapter 9 (Sparse Attention) and Chapter 10 (MoE)
2. Check the reference implementation for hints
3. Profile your code to identify bottlenecks
4. Test on smaller problem sizes first
5. Validate correctness before optimizing performance

Good luck building your inference engine!
