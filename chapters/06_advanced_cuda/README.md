# Chapter 06: Advanced CUDA Features

This chapter covers advanced CUDA programming features that enable efficient parallel algorithms and optimizations beyond basic thread-level programming.

## Learning Goals

By the end of this chapter, you will be able to:

1. **Use Warp Primitives** to perform efficient intra-warp communication
   - Understand shuffle operations for data exchange between threads
   - Apply vote operations for warp-level decision making
   - Leverage warp-level primitives for lock-free algorithms

2. **Implement Warp-Level Reductions** for high-performance collective operations
   - Build reductions using shuffle instructions
   - Compare performance with shared memory approaches
   - Understand when to use warp primitives vs. shared memory

3. **Work with Cooperative Groups** for flexible thread hierarchy
   - Use the cooperative groups API for cleaner code
   - Perform grid-wide synchronization
   - Create custom thread groups for algorithms

4. **Optimize with CUDA Graphs** to reduce kernel launch overhead
   - Create and execute CUDA graphs manually
   - Use stream capture for automatic graph creation
   - Measure performance improvements from graphs

5. **Design Parallel Scan Algorithms** (prefix sum)
   - Implement work-efficient scan algorithms
   - Understand inclusive vs. exclusive scans
   - Build block-level and device-level scans

## Key Concepts

### Warp Primitives

**Warp**: A group of 32 threads that execute in SIMT (Single Instruction, Multiple Thread) fashion. Threads within a warp are synchronized and can communicate efficiently.

#### Shuffle Operations
- `__shfl_sync(mask, var, srcLane)`: Read variable from a specific lane
- `__shfl_up_sync(mask, var, delta)`: Read from lane with lower ID
- `__shfl_down_sync(mask, delta)`: Read from lane with higher ID
- `__shfl_xor_sync(mask, var, laneMask)`: Read from lane with XOR'd ID

**Use Cases**:
- Warp-level reductions (sum, max, min)
- Broadcasting values within a warp
- Transpose operations
- Tree-based algorithms

#### Vote Operations
- `__all_sync(mask, predicate)`: Returns true if predicate is true for all active threads
- `__any_sync(mask, predicate)`: Returns true if predicate is true for any active thread
- `__ballot_sync(mask, predicate)`: Returns bitmask of predicate results

**Use Cases**:
- Early termination conditions
- Conflict detection
- Compact operations
- Dynamic parallelism decisions

### Cooperative Groups

A programming model that provides explicit thread group semantics:
- `thread_block`: Represents a CUDA block
- `thread_block_tile<Size>`: Subset of threads (e.g., warp)
- `grid_group`: Entire kernel grid

**Benefits**:
- More expressive code
- Compile-time optimization opportunities
- Easier-to-understand synchronization
- Support for grid-wide operations

### CUDA Graphs

A way to define GPU work as a graph of operations rather than a sequence of kernel launches.

**Advantages**:
- Reduced CPU overhead (single launch for entire graph)
- Kernel-level optimizations
- Better for recurring workloads
- Simplified scheduling

**Creation Methods**:
1. **Manual**: Explicitly define nodes and dependencies
2. **Stream Capture**: Record operations from a stream

**Performance**: 10-30% improvement for workloads with many small kernels

### Parallel Scan (Prefix Sum)

Given input array `[x₀, x₁, x₂, ..., xₙ]`:
- **Exclusive scan**: `[0, x₀, x₀+x₁, ..., Σxᵢ for i<n]`
- **Inclusive scan**: `[x₀, x₀+x₁, x₀+x₁+x₂, ..., Σxᵢ for i≤n]`

**Applications**:
- Stream compaction
- Radix sort
- Sparse matrix operations
- Resource allocation

## Directory Structure

```
06_advanced_cuda/
├── README.md (this file)
├── CMakeLists.txt
├── examples/
│   ├── 01_warp_primitives/
│   │   ├── warp_shuffle.cu
│   │   ├── warp_vote.cu
│   │   ├── CMakeLists.txt
│   │   └── README.md
│   ├── 02_warp_reduction/
│   │   ├── warp_reduce.cu
│   │   ├── CMakeLists.txt
│   │   └── README.md
│   ├── 03_cooperative_groups/
│   │   ├── cg_basics.cu
│   │   ├── grid_sync.cu
│   │   ├── CMakeLists.txt
│   │   └── README.md
│   ├── 04_cuda_graphs/
│   │   ├── graph_basics.cu
│   │   ├── graph_capture.cu
│   │   ├── CMakeLists.txt
│   │   └── README.md
│   └── 05_scan/
│       ├── prefix_scan.cu
│       ├── CMakeLists.txt
│       └── README.md
└── exercises/
    ├── 01_warp_reduce/
    │   ├── problem.md
    │   ├── starter.cu
    │   ├── solution.cu
    │   └── test.py
    └── 02_block_scan/
        ├── problem.md
        ├── starter.cu
        ├── solution.cu
        └── test.py
```

## Reading Materials

### Official NVIDIA Documentation

1. **[CUDA C++ Programming Guide - Warp Shuffle Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions)**
   - Complete reference for shuffle operations
   - Performance characteristics and constraints

2. **[CUDA C++ Programming Guide - Warp Vote Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-vote-functions)**
   - Vote operation semantics
   - Examples and use cases

3. **[Cooperative Groups Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)**
   - Complete cooperative groups API
   - Grid synchronization requirements

4. **[CUDA Graphs Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)**
   - Graph creation and execution
   - Stream capture API
   - Performance best practices

### Technical Papers and Blogs

5. **[Faster Parallel Reductions on Kepler](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/)**
   - Using shuffle instructions for reductions
   - Performance comparison with shared memory

6. **[Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)**
   - Comprehensive guide to warp primitives
   - Real-world examples and patterns

7. **[Cooperative Groups: Flexible CUDA Thread Programming](https://developer.nvidia.com/blog/cooperative-groups/)**
   - Introduction to cooperative groups
   - Migration from traditional CUDA code

8. **[Getting Started with CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/)**
   - Practical guide to CUDA graphs
   - Performance benchmarks

9. **[GPU Gems 3 - Chapter 39: Parallel Prefix Sum (Scan)](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)**
   - Work-efficient scan algorithm
   - Theory and implementation

### Research Papers

10. **"Work-Efficient Parallel GPU Methods for Single-Source Shortest Paths"** (Davidson et al., 2014)
    - Advanced use of warp primitives
    - High-performance graph algorithms

11. **"Single-pass Parallel Prefix Scan with Decoupled Look-back"** (Merrill & Garland, 2016)
    - State-of-the-art scan algorithm
    - Handles arbitrarily large inputs

## Prerequisites

- Completion of Chapter 03 (Memory Hierarchy) and Chapter 04 (Optimization)
- Understanding of CUDA execution model (warps, blocks, grids)
- Familiarity with parallel reduction algorithms
- CUDA Compute Capability 3.0 or higher (for shuffle operations)
- CUDA Compute Capability 6.0 or higher (for cooperative groups)
- CUDA 10.0 or higher (for CUDA graphs)

## Performance Characteristics

### Warp Primitives
- **Latency**: Shuffle operations are very fast (typically 1-2 cycles)
- **Bandwidth**: No shared memory or global memory access required
- **Occupancy**: No register pressure from synchronization
- **Best for**: Small reductions, intra-warp communication

### Cooperative Groups
- **Overhead**: Minimal (often compiled away)
- **Flexibility**: Better code organization and maintainability
- **Grid sync**: Requires kernel launch with cooperative launch API

### CUDA Graphs
- **Launch overhead**: ~90% reduction compared to separate launches
- **Optimization**: Driver can optimize graph execution
- **Limitation**: Graph structure must be known ahead of time
- **Best for**: Recurring workloads with fixed structure

## Common Pitfalls

1. **Incorrect warp mask**: Always use `0xffffffff` for full warp or proper mask
2. **Assuming warp size**: Never hardcode 32, use `warpSize` constant
3. **Shuffle across blocks**: Shuffle only works within a warp
4. **Grid sync without cooperative launch**: Must use `cudaLaunchCooperativeKernel`
5. **Modifying graphs**: Graphs are immutable; create new ones for changes
6. **Scan bank conflicts**: Careful indexing needed in shared memory scans

## Next Steps

After completing this chapter:
- **Chapter 07**: Multi-GPU Programming
- **Chapter 08**: Dynamic Parallelism
- **Chapter 09**: CUDA Streams and Concurrency
- **Advanced Projects**: Implement sorting, graph algorithms, or compression

## Building the Examples

From this directory:
```bash
mkdir build && cd build
cmake ..
make
```

Run individual examples:
```bash
./examples/01_warp_primitives/warp_shuffle
./examples/02_warp_reduction/warp_reduce
./examples/03_cooperative_groups/cg_basics
./examples/04_cuda_graphs/graph_basics
./examples/05_scan/prefix_scan
```

## Testing Your Understanding

Work through the exercises in order:
1. `exercises/01_warp_reduce/` - Implement warp-level max reduction
2. `exercises/02_block_scan/` - Implement block-level prefix scan

Each exercise includes:
- Problem description with requirements
- Starter code with TODOs
- Complete solution
- Python test script for validation
