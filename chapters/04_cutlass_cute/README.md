# Chapter 04: CUTLASS and CuTe

## Overview

This chapter introduces NVIDIA's CUTLASS library and the CuTe (CUDA Template Library) abstraction layer. You'll learn how to leverage these powerful tools to write high-performance GEMM kernels that achieve >90% of cuBLAS performance while maintaining code clarity and composability.

## Learning Goals

By the end of this chapter, you will:

- Understand CUTLASS architecture and design philosophy
- Master CuTe's Layout, Tensor, and Coordinate abstractions
- Implement GEMM kernels using CuTe primitives
- Utilize Tensor Cores via WMMA and MMA instructions
- Achieve >90% cuBLAS performance on modern GPUs
- Compare CuteDSL Python interface with C++ implementation
- Apply layout transformations for optimal memory access patterns

## Prerequisites

Before starting this chapter, you should have:

- **Completed Chapter 03**: Profiling and optimization techniques
- **Strong GEMM understanding**: Matrix multiplication algorithms, tiling strategies
- **Memory hierarchy knowledge**: Shared memory, register blocking, bank conflicts
- **C++ template familiarity**: Template metaprogramming basics
- **Target hardware**: NVIDIA GPU with compute capability 7.0+ (Volta or newer)

## Key Concepts

### 1. CUTLASS Architecture

CUTLASS (CUDA Templates for Linear Algebra Subroutines) is a collection of CUDA C++ template abstractions for implementing high-performance matrix operations. Key components:

- **Hierarchical structure**: Threadblock → Warp → Thread → Instruction
- **Composable primitives**: Layouts, Tensors, TiledCopy, TiledMMA
- **Hardware abstraction**: Portable code across GPU architectures
- **Performance**: Matches or exceeds cuBLAS for many workloads

### 2. CuTe (CUDA Template Library)

CuTe is CUTLASS 3.x's foundational abstraction layer that separates algorithm from data layout:

**Core Abstractions:**

- **Layout**: Maps logical coordinates to physical memory offsets
  ```cpp
  Layout<Shape<_4, _8>, Stride<_8, _1>>  // Row-major 4x8 matrix
  Layout<Shape<_4, _8>, Stride<_1, _4>>  // Column-major 4x8 matrix
  ```

- **Tensor**: Combines data pointer with Layout
  ```cpp
  Tensor tensor = make_tensor(ptr, Layout<Shape<_M, _N>, Stride<_N, _1>>{});
  ```

- **TiledCopy**: Describes multi-threaded memory copy patterns
  ```cpp
  auto copy_atom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, float>{};
  auto tiled_copy = make_tiled_copy(copy_atom, ...);
  ```

- **TiledMMA**: Describes multi-threaded matrix multiply-accumulate
  ```cpp
  using MMA = MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>;
  auto tiled_mma = make_tiled_mma(MMA{}, ...);
  ```

### 3. Layout Algebra

CuTe's layout algebra enables compile-time reasoning about memory access patterns:

```cpp
// Composition: combine transformations
auto layout = composition(layout1, layout2);

// Complement: find orthogonal coordinates
auto complement = complement(layout, shape);

// Logical divide: split dimensions
auto blocked = logical_divide(layout, tile_shape);
```

### 4. Tensor Cores

Modern NVIDIA GPUs include specialized Tensor Core units:

- **Volta (SM70)**: WMMA API, FP16 input/output
- **Turing (SM75)**: INT8, INT4 support
- **Ampere (SM80)**: MMA PTX, BF16, TF32, sparse matrices
- **Hopper (SM90)**: Warpgroup MMA, FP8, Tensor Memory Accelerator (TMA)

**Performance benefit**: 8-16x throughput vs CUDA cores for matrix operations

### 5. CuteDSL vs CuTe C++

| Aspect | CuTe C++ | CuteDSL Python |
|--------|----------|----------------|
| **Language** | C++ templates | Python DSL |
| **Compilation** | nvcc/clang | JIT compilation |
| **Performance** | Native GPU code | Same (generates C++) |
| **Development speed** | Slower (C++ build times) | Faster (Python iteration) |
| **Debugging** | CUDA tools (cuda-gdb, NSight) | Python debugger + CUDA tools |
| **Integration** | Direct CUDA/C++ | PyTorch/JAX interop |
| **Learning curve** | Steeper (templates) | Gentler (Python syntax) |
| **Best for** | Production libraries | Research, prototyping |

## Chapter Structure

### Examples

1. **01_cutlass_setup**: Verify CUTLASS installation and basic usage
2. **02_cute_layout**: Master Layout algebra and transformations
3. **03_cute_gemm**: Implement GEMM using CuTe primitives
4. **04_tensor_cores**: Leverage WMMA and MMA for peak performance
5. **05_cutedsl**: Explore Python DSL interface

### Exercises

1. **01_batched_gemm**: Implement batched matrix multiplication
2. **02_fp16_gemm**: Build FP16 GEMM with Tensor Cores (target: 90%+ cuBLAS)

## References

### Official Documentation

- [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md)
- [CuTe Tutorial](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/01_layout.md)
- [CUTLASS 3.0 Announcement](https://developer.nvidia.com/blog/cutlass-3-0-faster-ai-software-for-nvidia-hopper/)

### Key Papers

- [CUTLASS: Fast Linear Algebra in CUDA C++](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)
- [Dissecting Tensor Cores](https://arxiv.org/abs/2206.02607)

### NVIDIA Developer Resources

- [CUDA Programming Guide - WMMA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
- [PTX ISA - MMA Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions)
- [Tensor Cores Programming](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)

## Performance Targets

| Kernel Type | Target Performance | Notes |
|-------------|-------------------|-------|
| FP32 GEMM | 60-70% of peak FLOPS | Limited by memory bandwidth |
| FP16 GEMM (Tensor Cores) | 90%+ cuBLAS | On Volta/Ampere/Hopper |
| INT8 GEMM (Tensor Cores) | 90%+ cuBLAS | On Turing+ |
| Batched GEMM | 85%+ cuBLAS | Small matrices benefit less |

**Measurement methodology:**
- Measure effective TFLOPS: `(2 * M * N * K) / time_seconds / 1e12`
- Compare against cuBLAS for same problem size
- Use NSight Compute for roofline analysis
- Profile on target architecture (Ampere recommended)

## Development Environment

### Required Software

```bash
# CUDA Toolkit (11.0+)
nvcc --version

# CMake (3.18+)
cmake --version

# Git (for CUTLASS)
git --version
```

### Installing CUTLASS

```bash
# Clone CUTLASS repository
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
git checkout v3.4.0  # Or latest stable version

# Set environment variable
export CUTLASS_DIR=/path/to/cutlass
```

### Building Examples

```bash
cd /home/uvxiao/cuda-kernel-tutorial/chapters/04_cutlass_cute
mkdir build && cd build
cmake .. -DCUTLASS_DIR=$CUTLASS_DIR
make -j$(nproc)
```

## Common Pitfalls

1. **Layout mismatches**: Ensure source and destination layouts are compatible
2. **Alignment requirements**: Tensor Cores require 16-byte aligned addresses
3. **Shape constraints**: MMA instructions have fixed tile shapes (e.g., 16x8x16)
4. **Register pressure**: Excessive register usage limits occupancy
5. **Synchronization**: Missing `__syncthreads()` in shared memory operations

## Tips for Success

1. **Start simple**: Begin with CuTe Layout exercises before jumping to GEMM
2. **Visualize layouts**: Use `print_layout()` to understand memory patterns
3. **Profile early**: Use NSight Compute to identify bottlenecks
4. **Study examples**: CUTLASS repo has extensive examples in `examples/cute/`
5. **Incremental complexity**: Master WMMA before attempting PTX MMA
6. **Compare with cuBLAS**: Always benchmark against the gold standard

## Next Steps

After completing this chapter:

- **Chapter 05**: DeepGEMM - implement state-of-the-art GEMM kernels
- **Advanced topics**: Grouped GEMM, attention kernels, sparse operations
- **Real-world integration**: Build custom PyTorch/JAX operators

## Additional Resources

### Video Tutorials

- [GTC 2023: CUTLASS 3.0 Deep Dive](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51393/)
- [CppCon 2022: Modern CUDA C++ for Performance](https://www.youtube.com/watch?v=XXXmXD4Y9kI)

### Community

- [CUTLASS GitHub Discussions](https://github.com/NVIDIA/cutlass/discussions)
- [NVIDIA Developer Forums - CUDA](https://forums.developer.nvidia.com/c/gpu-accelerated-libraries/cuda-libraries/)

---

**Ready to begin?** Start with `examples/01_cutlass_setup/` to verify your environment!
