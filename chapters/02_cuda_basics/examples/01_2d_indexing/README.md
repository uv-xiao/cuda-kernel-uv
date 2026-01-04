# Example 01: 2D Thread and Block Indexing

## Overview

This example demonstrates how to map CUDA threads to 2D data structures (matrices). Understanding thread indexing is fundamental to CUDA programming, as it determines how computational work is distributed across the GPU.

## Concepts Covered

1. **1D Indexing**: Flattened thread indexing for linear data access
2. **2D Indexing**: Row/column-based indexing for matrix operations
3. **Thread Organization**: How blocks and threads map to data
4. **Boundary Checking**: Handling non-multiple dimensions

## Thread Indexing Formulas

### 1D Indexing

For a linear array processed by 1D blocks:

```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;

if (idx < total_elements) {
    // Process element at idx
}
```

**Example**: 1000 elements, 256 threads/block
- Grid size: ceil(1000/256) = 4 blocks
- Block 0: threads 0-255 → elements 0-255
- Block 1: threads 0-255 → elements 256-511
- Block 2: threads 0-255 → elements 512-767
- Block 3: threads 0-255 → elements 768-1023 (some threads idle)

### 2D Indexing

For matrices processed by 2D blocks:

```cuda
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
int idx = row * width + col;  // Row-major order

if (row < height && col < width) {
    // Process element at (row, col)
}
```

**Example**: 1024×1024 matrix, 16×16 threads/block
- Grid size: (64, 64) blocks
- Each block processes a 16×16 tile
- Total threads: 64 × 64 × 16 × 16 = 1,048,576

## Memory Layout

CUDA uses **row-major** order for 2D arrays in memory:

```
Matrix:        Memory Layout:
[0 1 2]        [0, 1, 2, 3, 4, 5, 6, 7, 8]
[3 4 5]
[6 7 8]

Element (row, col) → Index = row * width + col
Element (1, 2) → Index = 1 * 3 + 2 = 5
```

## Block Size Selection

Choosing the right block size is important:

**1D Blocks:**
- Common sizes: 128, 256, 512 threads
- Must be multiple of 32 (warp size)
- 256 threads is a good default

**2D Blocks:**
- Common sizes: (16,16), (32,8), (8,32)
- Total threads = x × y (typically 128-512)
- Square blocks (16,16) work well for square matrices

**Factors to Consider:**
- Hardware limits: Max 1024 threads/block
- Occupancy: More threads/block → higher occupancy
- Shared memory: Larger blocks use more shared memory
- Warp efficiency: Avoid very small blocks

## Examples in This Program

### Example 1: 1D Indexing

```cuda
matrixAdd1D<<<gridSize1D, blockSize1D>>>(d_A, d_B, d_C, width, height);
```

- Treats 2D matrix as 1D array
- Simple but less intuitive for matrix operations
- Good for element-wise operations

### Example 2: 2D Indexing

```cuda
matrixAdd2D<<<gridSize2D, blockSize2D>>>(d_A, d_B, d_C, width, height);
```

- Uses 2D blocks and grids
- More intuitive for matrix operations
- Maps naturally to row/column structure

### Example 3: Thread Organization Visualization

Shows how threads are organized:
```
Thread (0,0) in Block (0,0): Global position (0,0)
Thread (1,0) in Block (0,0): Global position (1,0)
Thread (0,1) in Block (0,0): Global position (0,1)
...
```

### Example 4: Matrix Transpose

Demonstrates different access patterns:
- Read: row-major (coalesced)
- Write: column-major (non-coalesced)

## Building and Running

```bash
# From the example directory
mkdir build && cd build
cmake ..
make

# Run the example
./indexing_2d
```

## Expected Output

```
=== CUDA 2D Indexing Examples ===

Matrix size: 1024 x 1024 (1048576 elements, 4.00 MB)

Example 1: 1D Indexing
------------------------
Configuration:
  Block size: 256 threads
  Grid size:  4096 blocks
  Total threads: 1048576

1D Indexing: PASSED

Example 2: 2D Indexing
------------------------
Configuration:
  Block size: (16, 16) = 256 threads
  Grid size:  (64, 64) = 4096 blocks
  Total threads: 1048576

2D Indexing: PASSED

...
```

## Common Pitfalls

### 1. Missing Boundary Checks

```cuda
// WRONG - may access out of bounds
int idx = blockIdx.x * blockDim.x + threadIdx.x;
C[idx] = A[idx] + B[idx];

// CORRECT
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < size) {
    C[idx] = A[idx] + B[idx];
}
```

### 2. Incorrect Index Calculation

```cuda
// WRONG - column-major instead of row-major
int idx = col * height + row;

// CORRECT - row-major
int idx = row * width + col;
```

### 3. Integer Division for Grid Size

```cuda
// WRONG - truncates, may leave elements unprocessed
int grid_size = size / block_size;

// CORRECT - rounds up
int grid_size = (size + block_size - 1) / block_size;
```

### 4. Block Size Not Multiple of 32

```cuda
// POOR - wastes warp resources
dim3 block(30, 30);  // 900 threads = 29 warps (last warp has 4 threads)

// BETTER - full warps
dim3 block(32, 32);  // 1024 threads = 32 warps
```

## Performance Considerations

1. **Coalesced Memory Access**
   - 1D indexing with sequential access is coalesced
   - 2D indexing is coalesced if column access is sequential
   - Transpose has non-coalesced writes (Chapter 02, Example 2 fixes this)

2. **Occupancy**
   - 256 threads/block gives good occupancy on most GPUs
   - Use `--ptxas-options=-v` to see register usage

3. **Grid/Block Size**
   - Make grid large enough to fill all SMs
   - Modern GPUs have 80-132 SMs
   - Aim for 1000+ blocks for good load balancing

## Next Steps

- See Example 02 (Shared Memory) for optimized matrix transpose
- Learn about memory coalescing in detail
- Experiment with different block sizes and measure performance

## References

- CUDA C Programming Guide: Thread Hierarchy
- CUDA Best Practices: Execution Configuration
