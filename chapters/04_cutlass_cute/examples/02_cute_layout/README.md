# Example 02: CuTe Layout Algebra

## Overview

This example explores CuTe's powerful layout algebra, which is the foundation for efficient memory access patterns. Understanding layouts is critical for writing high-performance CUDA kernels.

## Learning Objectives

- Master CuTe Layout creation and manipulation
- Understand compile-time vs runtime layouts
- Apply layout transformations (composition, complement, logical_divide)
- Visualize memory access patterns
- Optimize data layouts for hardware characteristics

## What is a Layout?

A **Layout** in CuTe is a mathematical function that maps logical coordinates to physical memory offsets:

```
offset = Layout(coord)
```

**Components:**
- **Shape**: Logical dimensions (e.g., 4x8 matrix has shape (4, 8))
- **Stride**: Memory offset per dimension (e.g., row-major has stride (8, 1))

## Key Concepts

### 1. Row-Major vs Column-Major

```cpp
// Row-major (C/CUDA default): consecutive columns in memory
auto row_major = make_layout(make_shape(4, 8),
                             make_stride(8, 1));
// Memory: [row0_col0, row0_col1, ..., row0_col7, row1_col0, ...]

// Column-major (Fortran/BLAS default): consecutive rows in memory
auto col_major = make_layout(make_shape(4, 8),
                             make_stride(1, 4));
// Memory: [row0_col0, row1_col0, ..., row3_col0, row0_col1, ...]
```

### 2. Compile-Time Layouts

Using `Int<N>` enables compile-time optimizations:

```cpp
// Compile-time: all dimensions known at compile time
auto static_layout = make_layout(make_shape(Int<4>{}, Int<8>{}),
                                 make_stride(Int<8>{}, Int<1>{}));

// Runtime: dimensions determined at runtime
auto dynamic_layout = make_layout(make_shape(M, N),
                                  make_stride(N, 1));
```

**Benefits of compile-time:**
- No runtime overhead for bounds checking
- Better register allocation
- Enables loop unrolling
- Smaller code size

### 3. Layout Algebra Operations

#### Composition

Combines two layouts (function composition):

```cpp
auto layout1 = make_layout(make_shape(8, 4), make_stride(4, 1));  // 8x4
auto layout2 = make_layout(make_shape(4, 2), make_stride(2, 1));  // 4x2
auto composed = composition(layout1, layout2);  // 8x4x2
```

#### Complement

Finds orthogonal dimensions:

```cpp
auto layout = make_layout(make_shape(4, 8), make_stride(8, 1));
auto comp = complement(layout, make_shape(32));  // Fills remaining space
```

#### Logical Divide

Splits dimensions into tiles:

```cpp
auto layout = make_layout(make_shape(16, 16), make_stride(16, 1));
auto tiled = logical_divide(layout, make_shape(4, 4));  // 4x4 tiles
// Result: (tile_m, tile_n) : (thread_m, thread_n)
```

### 4. Hierarchical Layouts

CuTe supports nested shapes for hierarchical tiling:

```cpp
// 3-level hierarchy: block -> warp -> thread
auto layout = make_layout(
    make_shape(make_shape(4, 8), make_shape(2, 4), make_shape(1, 2)),
    make_stride(make_stride(32, 1), make_stride(16, 8), make_stride(0, 4))
);
```

### 5. Coalesced Memory Access

For optimal performance, threads in a warp should access consecutive memory:

```cpp
// Good: stride-1 in innermost dimension
auto coalesced = make_layout(make_shape(32, 4), make_stride(4, 1));

// Bad: stride-1 not in innermost dimension
auto strided = make_layout(make_shape(32, 4), make_stride(1, 32));
```

## Files in This Example

### layout_basics.cu

Demonstrates:
- Creating various layout types
- Visualizing memory patterns
- Compile-time vs runtime layouts
- Coordinate-to-offset mapping

**Key functions:**
- `print_layout_pattern()`: Visualizes how data is arranged in memory
- `compare_layouts()`: Shows difference between row/column-major
- `test_hierarchical_layout()`: Demonstrates multi-level tiling

### layout_operations.cu

Demonstrates:
- Layout composition
- Layout complement
- Logical divide for tiling
- Block/thread decomposition
- Swizzling for bank conflict avoidance

**Key functions:**
- `test_composition()`: Combines layouts
- `test_logical_divide()`: Creates tiled layouts
- `test_swizzle()`: Demonstrates XOR-based swizzling

## Building and Running

```bash
mkdir build && cd build
cmake .. -DCUTLASS_DIR=$CUTLASS_DIR
make -j$(nproc)

# Run layout basics
./layout_basics

# Run layout operations
./layout_operations
```

## Expected Output

### layout_basics

```
=== Layout Basics ===

1. Row-major layout (4x8):
   Logical view:
     [0.0  1.0  2.0  3.0  4.0  5.0  6.0  7.0]
     [8.0  9.0 10.0 11.0 12.0 13.0 14.0 15.0]
     ...

2. Column-major layout (4x8):
   Physical memory: [0, 8, 16, 24, 1, 9, 17, ...]

3. Compile-time optimizations:
   Static layout size: 32 (known at compile time)
   Dynamic layout size: 32 (computed at runtime)
```

### layout_operations

```
=== Layout Operations ===

1. Composition:
   Layout1: (8, 4) with stride (4, 1)
   Layout2: (4, 2) with stride (2, 1)
   Composed: (8, 4, 2)

2. Logical Divide:
   Original: (16, 16)
   Tiled (4x4 blocks): ((4, 4), (4, 4))

3. Swizzle for shared memory:
   Base layout: (128, 8)
   Swizzled layout: (128, 8) with XOR pattern
```

## Performance Considerations

### Coalescing

**Rule:** Threads 0-31 in a warp should access bytes `[0-3], [4-7], ..., [124-127]`

```cpp
// Thread index to data mapping
int tid = threadIdx.x;
float* ptr = base_ptr + layout(tid, ...);  // Should be consecutive for tid 0-31
```

### Bank Conflicts

**Rule:** Avoid multiple threads accessing same bank in shared memory

```cpp
// 32 banks, 4-byte words
// Conflict: All threads access column 0
__shared__ float smem[32][32];
float val = smem[threadIdx.x][0];  // BAD: 32-way conflict

// No conflict: Swizzle or transpose access
float val = smem[0][threadIdx.x];  // GOOD: All different banks
```

### Alignment

**Rule:** 16-byte alignment for vectorized loads (float4, uint4)

```cpp
// Ensure layout stride is multiple of 4 for float4 loads
auto layout = make_layout(make_shape(M, N), make_stride(N, 1));
// N should be multiple of 4 for optimal vectorization
```

## Common Patterns

### Pattern 1: Block-Thread Decomposition

```cpp
// Decompose MxN matrix into blocks and threads
auto gmem_layout = make_layout(make_shape(M, N), make_stride(N, 1));

// Each block handles BM x BN tile
auto block_layout = logical_divide(gmem_layout, make_shape(BM, BN));

// Each thread handles TM x TN elements within block
auto thread_layout = logical_divide(block_layout, make_shape(TM, TN));
```

### Pattern 2: Tiled Copy

```cpp
// Source and destination with different layouts
auto src_layout = make_layout(make_shape(M, N), make_stride(N, 1));     // Row-major
auto dst_layout = make_layout(make_shape(M, N), make_stride(1, M));     // Column-major

// Copy with automatic layout conversion
copy_tensor(src_tensor, dst_tensor);
```

### Pattern 3: Swizzled Shared Memory

```cpp
// Avoid bank conflicts with XOR swizzle
constexpr int kSwizzle = 3;  // XOR with bits [kSwizzle:0]
auto swizzled = composition(base_layout, Swizzle<kSwizzle, 0, 3>{});
```

## Exercises

After understanding the examples, try these:

1. **Blocked Layout**: Create a 16x16 matrix divided into 4x4 tiles
2. **Strided Access**: Measure coalescing efficiency for different strides
3. **Custom Swizzle**: Implement bank conflict avoidance for 64-element shared memory
4. **3D Tensor**: Create layout for 3D tensor with optimal access pattern

## Debugging Tips

### Visualize Layouts

```cpp
// Print layout structure
print(layout);

// Print coordinate mappings
for (int i = 0; i < size<0>(layout); ++i) {
    for (int j = 0; j < size<1>(layout); ++j) {
        printf("(%d,%d) -> %d\n", i, j, layout(i, j));
    }
}
```

### Check Coalescing

Use NSight Compute:
```bash
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum ./layout_basics
```

Look for:
- **Global Load Efficiency**: Should be >80% for coalesced access
- **Shared Memory Bank Conflicts**: Should be close to 0

## Next Steps

After mastering layouts:

1. **Proceed to Example 03**: Apply layouts to implement GEMM
2. **Study CUTLASS examples**: See production-quality layout usage
3. **Profile your layouts**: Verify coalescing and bank conflict avoidance

## References

- [CuTe Layout Documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/01_layout.md)
- [CuTe Layout Algebra](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/02_layout_algebra.md)
- [CUDA Memory Coalescing](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)
