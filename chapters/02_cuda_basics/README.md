# Chapter 02: CUDA Basics & Memory

## Overview

This chapter covers the fundamental concepts of CUDA programming, focusing on thread organization, memory hierarchy, and synchronization. Understanding these concepts is critical for writing efficient CUDA kernels.

## Learning Goals

By the end of this chapter, you will be able to:

1. **Thread and Block Indexing**
   - Calculate unique thread indices in 1D, 2D, and 3D configurations
   - Map threads to data elements efficiently
   - Choose optimal grid and block dimensions

2. **Memory Types and Hierarchy**
   - Understand CUDA's memory hierarchy (registers, shared, global, constant, texture)
   - Use shared memory to improve performance
   - Avoid memory access pitfalls (coalescing, bank conflicts)

3. **Synchronization**
   - Use `__syncthreads()` correctly within thread blocks
   - Understand synchronization constraints and deadlocks
   - Implement barrier-based algorithms

4. **Performance Optimization**
   - Optimize memory access patterns
   - Reduce global memory transactions
   - Use shared memory for data reuse

## Key Concepts

### 1. Thread Hierarchy

CUDA organizes threads in a three-level hierarchy:

```
Grid → Blocks → Threads
```

**1D Indexing:**
```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

**2D Indexing:**
```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int idx = row * width + col;
```

**3D Indexing:**
```cuda
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
int idx = z * (width * height) + y * width + x;
```

### 2. Memory Hierarchy

CUDA GPUs have a complex memory hierarchy with different performance characteristics:

| Memory Type | Scope | Lifetime | Speed | Size | Declaration |
|-------------|-------|----------|-------|------|-------------|
| **Registers** | Thread | Thread | Fastest | ~64KB/SM | Automatic variables |
| **Local Memory** | Thread | Thread | Slow (DRAM) | Large | Large arrays, spilled registers |
| **Shared Memory** | Block | Block | Very Fast | 48-164KB/SM | `__shared__` |
| **Global Memory** | Grid | Application | Slow | GBs | `cudaMalloc()` |
| **Constant Memory** | Grid | Application | Fast (cached) | 64KB | `__constant__` |
| **Texture Memory** | Grid | Application | Fast (cached) | - | Texture objects |

**Memory Access Latency (approximate):**
- Registers: 1 cycle
- Shared Memory: ~20-30 cycles
- L1/L2 Cache: ~80-200 cycles
- Global Memory: ~400-800 cycles

### 3. Shared Memory

Shared memory is on-chip memory shared by all threads in a block. It's much faster than global memory and is crucial for optimization.

**Key Features:**
- Low latency (~100x faster than global memory)
- Limited size (48KB-164KB per SM)
- Requires `__syncthreads()` for coordination
- Subject to bank conflicts

**Common Uses:**
- Tiling for matrix operations
- Reduction operations
- Data sharing between threads
- Avoiding redundant global memory accesses

**Bank Conflicts:**
Shared memory is divided into 32 banks. Simultaneous access to the same bank (except same address) causes conflicts.

```cuda
// No conflict - different banks
__shared__ float data[32];
float val = data[threadIdx.x];

// Conflict - all threads access bank 0
float val = data[0];
```

### 4. Synchronization

**`__syncthreads()`:**
- Barrier synchronization within a thread block
- All threads in the block must reach the barrier
- Cannot be used across different blocks
- Must not be inside divergent code paths

**Example:**
```cuda
__global__ void example() {
    __shared__ float sdata[256];

    // Load data
    sdata[threadIdx.x] = /* ... */;

    // Wait for all threads to load
    __syncthreads();

    // Now all threads can safely read any sdata element
    float val = sdata[255 - threadIdx.x];
}
```

**Common Pitfalls:**
```cuda
// WRONG - deadlock in divergent code
if (threadIdx.x < 128) {
    __syncthreads();  // Only some threads reach here
}

// CORRECT
if (threadIdx.x < 128) {
    // do work
}
__syncthreads();  // All threads reach here
```

### 5. Coalesced Memory Access

For optimal performance, threads in a warp should access consecutive memory locations.

**Coalesced (Good):**
```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float val = data[idx];  // Sequential access
```

**Non-coalesced (Bad):**
```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float val = data[idx * stride];  // Strided access
```

### 6. Atomic Operations

Atomic operations ensure thread-safe updates but can be slow:

```cuda
atomicAdd(&counter, 1);
atomicMax(&max_val, value);
atomicCAS(&lock, 0, 1);  // Compare-and-swap
```

**Use Cases:**
- Histograms
- Reductions (when shared memory not feasible)
- Thread-safe counters

## Reading Materials

### Essential Reading

1. **CUDA C Programming Guide - Memory Hierarchy**
   - [https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy)
   - Official documentation on CUDA memory types

2. **CUDA C Best Practices Guide - Memory Optimization**
   - [https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)
   - Best practices for memory usage

3. **Shared Memory and Synchronization**
   - [https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
   - Details on shared memory and bank conflicts

### Supplementary Reading

4. **Mark Harris - "Using Shared Memory in CUDA C/C++"**
   - [https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
   - Practical examples of shared memory usage

5. **CUDA Refresher - Memory and Caches**
   - [https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/)
   - Modern CUDA memory model

6. **Optimizing Parallel Reduction in CUDA**
   - [https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
   - Classic NVIDIA presentation on reduction optimization

### Papers

7. **Volkov & Demmel - "Benchmarking GPUs to Tune Dense Linear Algebra"**
   - Understanding memory bandwidth and optimization
   - SC'08 Paper

## Performance Considerations

### 1. Memory Bandwidth

Modern GPUs are often memory-bound. Key strategies:

- **Maximize coalesced access**: Ensure consecutive threads access consecutive memory
- **Reuse data**: Load from global memory once, reuse from shared/registers
- **Minimize transfers**: Reduce CPU-GPU communication
- **Use appropriate memory**: Shared > L1 > L2 > Global

### 2. Occupancy

Occupancy is the ratio of active warps to maximum warps per SM.

**Factors affecting occupancy:**
- Block size (must be multiple of 32)
- Register usage per thread
- Shared memory usage per block

**Guidelines:**
- Use 128-256 threads per block typically
- Check with `--ptxas-options=-v` compiler flag
- Use CUDA Occupancy Calculator

### 3. Shared Memory Optimization

**Tiling:**
Break large computations into tiles that fit in shared memory.

```cuda
// Load tile into shared memory
for (int i = 0; i < TILE_SIZE; i++) {
    tile[threadIdx.y][threadIdx.x] = A[...];
    __syncthreads();
    // Compute using tile
    __syncthreads();
}
```

**Padding to avoid bank conflicts:**
```cuda
// Without padding - potential conflicts
__shared__ float tile[32][32];

// With padding - no conflicts
__shared__ float tile[32][33];
```

### 4. Reduction Patterns

Efficient parallel reduction requires careful optimization:

1. Sequential addressing (avoid divergence)
2. First add during global load
3. Unroll last warp (no sync needed)
4. Multiple elements per thread

### 5. Common Pitfalls

1. **Uncoalesced memory access**: Kills performance
2. **Bank conflicts**: Reduces shared memory throughput
3. **Synchronization in divergent code**: Causes deadlock
4. **Too much shared memory**: Reduces occupancy
5. **Excessive register usage**: Spills to local memory
6. **Atomic operation overuse**: Serializes execution

## Examples

This chapter includes four detailed examples:

1. **2D Indexing** (`examples/01_2d_indexing/`)
   - Matrix element access patterns
   - Proper thread-to-data mapping

2. **Shared Memory** (`examples/02_shared_memory/`)
   - Matrix transpose optimization
   - Bank conflict avoidance

3. **Reduction** (`examples/03_reduction/`)
   - Parallel sum reduction
   - Multiple optimization stages

4. **Histogram** (`examples/04_histogram/`)
   - Atomic operations
   - Per-block histograms

## Exercises

Practice your skills with two exercises:

1. **Matrix Transpose** (`exercises/01_matrix_transpose/`)
   - Implement naive and optimized versions
   - Measure performance improvement

2. **Dot Product** (`exercises/02_dot_product/`)
   - Parallel vector dot product
   - Reduction within blocks

## Building Examples

```bash
cd /home/uvxiao/cuda-kernel-tutorial/chapters/02_cuda_basics
mkdir build && cd build
cmake ..
make -j
```

## Running Examples

```bash
# 2D Indexing
./examples/01_2d_indexing/indexing_2d

# Shared Memory
./examples/02_shared_memory/shared_mem

# Reduction
./examples/03_reduction/reduction

# Histogram
./examples/04_histogram/histogram
```

## Next Steps

After completing this chapter:
- Move to Chapter 03: Advanced Memory Patterns
- Study memory coalescing in detail
- Learn about constant and texture memory
- Explore warp-level primitives

## Additional Resources

- NVIDIA CUDA Samples: `/usr/local/cuda/samples`
- CUDA Toolkit Documentation: [https://docs.nvidia.com/cuda/](https://docs.nvidia.com/cuda/)
- CUDA Zone: [https://developer.nvidia.com/cuda-zone](https://developer.nvidia.com/cuda-zone)
- Parallel Forall Blog: [https://developer.nvidia.com/blog/](https://developer.nvidia.com/blog/)

## Summary

Key takeaways:
- Thread indexing maps computational work to GPU threads
- Memory hierarchy determines performance characteristics
- Shared memory enables fast data sharing within blocks
- Synchronization coordinates thread execution
- Coalesced access and proper memory usage are critical for performance

Master these fundamentals before moving to advanced topics!
