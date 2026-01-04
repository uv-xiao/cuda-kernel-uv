# Chapter 03: Profiling & Optimization

## Overview

This chapter teaches you how to profile and optimize CUDA kernels through the practical example of matrix multiplication. You'll learn to use professional profiling tools and apply systematic optimization techniques to achieve near-cuBLAS performance.

## Learning Goals

By the end of this chapter, you will be able to:

1. **Use NVIDIA Profiling Tools**
   - Profile kernels with Nsight Compute (`ncu`)
   - Analyze system-level performance with Nsight Systems (`nsys`)
   - Interpret key performance metrics and bottlenecks

2. **Understand Key Performance Metrics**
   - Memory bandwidth utilization and coalescing efficiency
   - Occupancy and register/shared memory usage
   - Arithmetic intensity and compute throughput
   - Warp efficiency and divergence

3. **Apply Optimization Techniques**
   - Memory access pattern optimization (coalescing)
   - Shared memory tiling to reduce global memory traffic
   - Loop unrolling and instruction-level parallelism
   - Bank conflict avoidance

4. **Follow Systematic Optimization Workflow**
   - Profile to identify bottlenecks
   - Apply targeted optimizations
   - Measure and validate improvements
   - Iterate until performance goals are met

## Key Concepts

### Memory Bandwidth

GPUs are often memory-bound. Understanding and maximizing memory bandwidth utilization is critical:

- **Coalescing**: Combine multiple memory accesses into fewer transactions
- **Bandwidth Utilization**: Achieved bandwidth vs. theoretical peak
- **Memory Hierarchy**: Global, shared, registers, L1/L2 cache

### Occupancy

The ratio of active warps to maximum possible warps:

- Higher occupancy helps hide memory latency
- Limited by registers, shared memory, and block size
- Sweet spot is typically 50-75% (not always 100%)

### Arithmetic Intensity

Ratio of compute operations to memory operations:

```
Arithmetic Intensity = FLOPs / Bytes Transferred
```

- Low intensity → memory-bound (optimize memory access)
- High intensity → compute-bound (optimize ALU utilization)

### Roofline Model

Performance is bounded by either:
- Memory bandwidth: `Performance ≤ Bandwidth × Arithmetic Intensity`
- Compute throughput: `Performance ≤ Peak FLOPS`

## Matrix Multiplication Case Study

We'll optimize matrix multiplication `C = A × B` where all matrices are `N × N`:

### Performance Progression

Following [siboehm's CUDA MatMul worklog](https://siboehm.com/articles/22/CUDA-MMM), we'll progress through:

| Version | Technique | GFLOPS (A100) | % of cuBLAS |
|---------|-----------|---------------|-------------|
| 1. Naive | One thread per element | ~300 | ~1.5% |
| 2. Coalesced | Memory coalescing | ~1,200 | ~6% |
| 3. Tiled | Shared memory (1D) | ~4,500 | ~23% |
| 4. Optimized | 2D tiling + unrolling | ~18,000 | ~90% |
| cuBLAS | NVIDIA's optimized | ~20,000 | 100% |

*Note: GFLOPS targets assume A100 GPU (312 TOPS FP32). Scale expectations for your GPU.*

### Computational Complexity

For `N × N` matrices:
- **FLOPs**: `2 × N³` (N³ multiplies + N³ adds)
- **Memory**: `3 × N²` elements read/written (minimum)
- **Arithmetic Intensity**: `O(N)` with tiling, `O(1)` without

## Chapter Structure

### 1. Examples (Progressive Optimization)

- **01_matmul_naive**: Baseline implementation (1 thread = 1 output)
- **02_matmul_coalesced**: Memory access pattern optimization
- **03_matmul_tiled**: Shared memory tiling (basic)
- **04_matmul_optimized**: Advanced optimizations (2D tiling, unrolling)

Each example includes:
- Fully commented source code
- CMakeLists.txt for building
- README.md with profiling instructions and expected metrics

### 2. Profiling Guide

- **profiling_guide.md**: Step-by-step Nsight Compute/Systems tutorials
- **metrics_explained.md**: Deep dive into performance metrics
- **sample_commands.sh**: Ready-to-run profiling commands

### 3. Exercises

- **01_optimize_transpose**: Apply learned techniques to matrix transpose
- Includes starter code, solution, and tests

## Getting Started

### Prerequisites

1. **Hardware**: CUDA-capable GPU (compute capability 7.0+)
2. **Software**:
   - CUDA Toolkit 11.0+ (includes Nsight tools)
   - CMake 3.18+
   - Python 3.7+ (for testing)

### Verify Nsight Tools Installation

```bash
# Check Nsight Compute
ncu --version

# Check Nsight Systems
nsys --version

# If not found, add to PATH:
export PATH=/usr/local/cuda/bin:$PATH
```

### Build All Examples

```bash
cd /home/uvxiao/cuda-kernel-tutorial/chapters/03_profiling
mkdir build && cd build
cmake ..
make -j
```

### Run and Profile

```bash
# Run naive implementation
./examples/01_matmul_naive/matmul_naive

# Profile with Nsight Compute
ncu --set full ./examples/01_matmul_naive/matmul_naive

# Profile with Nsight Systems
nsys profile --stats=true ./examples/01_matmul_naive/matmul_naive
```

## Optimization Workflow

Follow this systematic approach:

### 1. Baseline Measurement

```bash
# Get baseline performance
./your_kernel

# Profile comprehensively
ncu --set full -o baseline ./your_kernel
```

### 2. Identify Bottlenecks

Look for:
- **Low memory bandwidth utilization** → Optimize access patterns
- **Low occupancy** → Reduce register/shared memory usage
- **Warp stalls** → Improve instruction mix or hide latency
- **Low compute throughput** → Increase arithmetic intensity

### 3. Apply Targeted Optimization

Choose optimization based on bottleneck:
- Memory-bound → Coalescing, caching, tiling
- Compute-bound → ILP, loop unrolling, better ALU usage
- Latency-bound → Increase occupancy

### 4. Measure Impact

```bash
# Profile optimized version
ncu --set full -o optimized ./your_kernel

# Compare metrics
ncu --import baseline.ncu-rep optimized.ncu-rep
```

### 5. Iterate

Repeat steps 2-4 until:
- Performance meets requirements
- Further optimization has diminishing returns
- You're within 80-90% of theoretical peak

## Key Performance Metrics Reference

| Metric | Good Target | Interpretation |
|--------|-------------|----------------|
| Memory Bandwidth Utilization | >70% | Efficient memory access |
| Compute (SM) Throughput | >60% | Efficient computation |
| Occupancy | 50-75% | Good latency hiding |
| Warp Execution Efficiency | >95% | Low divergence |
| Memory Replay Overhead | <10% | Good coalescing |
| Shared Memory Bank Conflicts | <5% | Efficient shared mem |

## Resources

### Official Documentation

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [CUDA Profiler User's Guide](https://docs.nvidia.com/cuda/profiler-users-guide/)

### Performance Guides

- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Kernel Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
- [GPU Architecture Whitepapers](https://www.nvidia.com/en-us/data-center/resources/gpu-architecture/)

### Tutorials and Articles

- [Simon Boehm's CUDA MatMul](https://siboehm.com/articles/22/CUDA-MMM) - Excellent optimization walkthrough
- [NVIDIA's Matrix Multiplication Sample](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/matrixMul)
- [Optimizing Parallel Reduction](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)

### Tools

- [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/) - Command-line profiler
- [Nsight Systems](https://developer.nvidia.com/nsight-systems) - System-wide analysis
- [NVIDIA Visual Profiler](https://developer.nvidia.com/nvidia-visual-profiler) (deprecated, use Nsight)

## Next Steps

1. **Start with examples**: Work through `01_matmul_naive` to `04_matmul_optimized`
2. **Profile each version**: Use the profiling guide to understand metrics
3. **Complete exercises**: Apply techniques to new problems
4. **Experiment**: Try different tile sizes, block dimensions, unroll factors

## Performance Expectations by GPU

Adjust your expectations based on your GPU:

| GPU | Compute Capability | Peak FP32 TFLOPS | Expected MatMul (GFLOPS) |
|-----|-------------------|------------------|---------------------------|
| RTX 3090 | 8.6 | 35.6 | ~28,000 (80%) |
| A100 | 8.0 | 19.5 | ~18,000 (90%) |
| V100 | 7.0 | 15.7 | ~14,000 (90%) |
| RTX 2080 Ti | 7.5 | 13.4 | ~10,000 (75%) |
| GTX 1080 Ti | 6.1 | 11.3 | ~8,000 (70%) |

*Note: Tensor core operations can achieve much higher performance for FP16/INT8.*

## Common Pitfalls

1. **Optimizing without profiling** - Always measure first
2. **Over-optimizing memory-bound code with compute tricks** - Fix the bottleneck
3. **Ignoring memory access patterns** - Coalescing is critical
4. **Excessive shared memory usage** - Can reduce occupancy
5. **Not validating correctness** - Always test after optimization

## Questions?

If you get stuck:
1. Check the README in each example directory
2. Review the profiling guide
3. Compare your metrics with expected values
4. Ensure your kernel logic is correct before optimizing

---

**Ready to start?** Head to `examples/01_matmul_naive/` to begin!
