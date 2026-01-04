# Exercise 01: Efficient Matrix Transpose

## Objective

Implement an efficient matrix transpose using shared memory optimization techniques.

## Problem Description

Given an input matrix of size M×N, produce its transpose of size N×M.

```
Input (4×3):        Output (3×4):
[1  2  3]           [1  4  7]
[4  5  6]           [2  5  8]
[7  8  9]           [3  6  9]
[10 11 12]          [10 11 12]
```

## Your Tasks

### Part 1: Naive Implementation

Implement a naive transpose kernel that reads from the input matrix and writes to the transposed position in the output matrix.

```cuda
__global__ void transposeNaive(float *input, float *output, int width, int height);
```

**Requirements:**
- Use 2D thread blocks (16×16 recommended)
- Handle boundary conditions correctly
- Ensure correct indexing for row-major layout

### Part 2: Shared Memory Optimization

Implement an optimized transpose using shared memory to enable coalesced memory access.

```cuda
__global__ void transposeShared(float *input, float *output, int width, int height);
```

**Requirements:**
- Use shared memory tiles
- Ensure coalesced reads from input
- Ensure coalesced writes to output
- Add proper synchronization

### Part 3: Bank Conflict Avoidance

Further optimize by avoiding shared memory bank conflicts.

```cuda
__global__ void transposeOptimized(float *input, float *output, int width, int height);
```

**Requirements:**
- Add padding to shared memory declaration
- Maintain correctness while avoiding conflicts

## Performance Goals

For a 4096×4096 matrix on a modern GPU:

| Implementation | Target Bandwidth |
|----------------|------------------|
| Naive | 100-150 GB/s |
| Shared Memory | 300-400 GB/s |
| No Bank Conflicts | 400-550 GB/s |

## Starter Code Structure

The `starter.cu` file provides:
- Matrix allocation and initialization
- Timing infrastructure
- Verification function
- Main driver code

You need to implement:
1. `transposeNaive()` kernel
2. `transposeShared()` kernel
3. `transposeOptimized()` kernel

## Hints

### Hint 1: 2D Indexing

```cuda
int col = blockIdx.x * TILE_DIM + threadIdx.x;
int row = blockIdx.y * TILE_DIM + threadIdx.y;
```

### Hint 2: Shared Memory Tile

```cuda
__shared__ float tile[TILE_DIM][TILE_DIM];

// Load tile (coalesced read)
tile[threadIdx.y][threadIdx.x] = input[row * width + col];
__syncthreads();

// Compute transposed indices
int new_col = blockIdx.y * TILE_DIM + threadIdx.x;
int new_row = blockIdx.x * TILE_DIM + threadIdx.y;

// Write transposed tile (coalesced write)
output[new_row * height + new_col] = tile[threadIdx.x][threadIdx.y];
```

### Hint 3: Bank Conflict Padding

```cuda
// Without padding (potential conflicts)
__shared__ float tile[TILE_DIM][TILE_DIM];

// With padding (no conflicts)
__shared__ float tile[TILE_DIM][TILE_DIM + 1];
```

### Hint 4: Boundary Handling

```cuda
if (row < height && col < width) {
    // Safe to access input[row * width + col]
}
```

## Testing

The test script (`test.py`) will:
1. Compile your code
2. Run all three kernels
3. Verify correctness
4. Measure and compare performance
5. Check if performance targets are met

Run tests:
```bash
python test.py
```

## Verification

Your implementation is correct if:
1. Output matrix is the transpose of input
2. All elements match exactly
3. Works for non-square matrices
4. Handles boundary cases correctly

## Evaluation Criteria

- **Correctness** (50%): All kernels produce correct results
- **Performance** (30%): Meets bandwidth targets
- **Code Quality** (20%): Clean, well-commented code

## Common Mistakes to Avoid

1. **Incorrect Index Calculation**
   - Remember: input is row-major, output is also row-major
   - input[row][col] → input[row * width + col]
   - output[col][row] → output[col * height + row]

2. **Missing Synchronization**
   - Always `__syncthreads()` after loading into shared memory
   - Always `__syncthreads()` before writing from shared memory

3. **Wrong Block Indexing**
   - Block indices swap during transpose
   - blockIdx.x for input → blockIdx.y for output position

4. **Boundary Conditions**
   - Non-square matrices need careful handling
   - Matrix size may not be multiple of tile size

## Bonus Challenges

1. **Dynamic Tile Size**: Make tile size a template parameter
2. **In-Place Transpose**: Transpose square matrices in-place
3. **Rectangular Tiles**: Use non-square tiles (e.g., 32×8)
4. **Multi-GPU**: Partition work across multiple GPUs

## Learning Objectives

After completing this exercise, you should understand:
- How memory access patterns affect performance
- The importance of coalesced memory access
- How shared memory enables optimization
- Bank conflict detection and avoidance
- Performance measurement and analysis

## Resources

- CUDA Programming Guide: Shared Memory
- Mark Harris: "Efficient Matrix Transpose in CUDA"
- Example 02 in this chapter (shared_mem.cu)

Good luck!
