# Exercise: Optimize Matrix Transpose

## Objective

Apply the optimization techniques you've learned to improve matrix transpose performance. You'll identify bottlenecks, fix them, and achieve >80% of peak memory bandwidth.

## Problem Description

Matrix transpose converts rows to columns: `B[j][i] = A[i][j]`

This operation is **memory-bound** and challenges your understanding of:
- Memory coalescing
- Shared memory bank conflicts
- Memory access patterns

## Your Task

1. **Analyze the naive implementation** (`starter.cu`)
   - Profile with Nsight Compute
   - Identify performance bottlenecks
   - Measure baseline bandwidth utilization

2. **Optimize the kernel** to achieve:
   - Memory bandwidth utilization: >80%
   - Shared memory bank conflicts: 0
   - Performance: >500 GB/s on A100 (or >80% of your GPU's peak)

3. **Verify correctness** using the provided test script

## Background: Why Transpose is Tricky

### The Problem

```cuda
// Reading A: Coalesced (consecutive threads read consecutive addresses)
A[row * N + col]  // row varies across threads

// Writing B: Uncoalesced (consecutive threads write to addresses N apart)
B[col * N + row]  // col varies, row same → stride-N access
```

One of the memory operations is always uncoalesced!

### The Solution

Use **shared memory** as a staging area:
1. Read from A coalesced → shared memory
2. Transpose in shared memory
3. Write to B coalesced from shared memory

**Challenge**: Avoid shared memory bank conflicts!

## Starter Code

The naive implementation in `starter.cu` has two kernels:

1. **transpose_naive**: Simple but slow (uncoalesced writes)
2. **transpose_shared**: Uses shared memory but has bank conflicts

Your job is to fix the bank conflicts in `transpose_shared`.

## Expected Performance

### Naive Version

| Metric | Expected Value |
|--------|----------------|
| Bandwidth | 100-150 GB/s |
| Memory BW Utilization | ~15-20% |
| Bank Conflicts | 0 |
| Global Store Efficiency | ~12% |

### Optimized Version

| Metric | Expected Value |
|--------|----------------|
| Bandwidth | >500 GB/s (A100) |
| Memory BW Utilization | >80% |
| Bank Conflicts | 0 |
| Global Store Efficiency | ~100% |
| **Speedup** | **3-5×** |

## Hints

### Hint 1: Padding

Shared memory banks are organized such that addresses differing by 32 elements (for 4-byte values) map to the same bank.

When reading columns from a 32×32 shared memory array:
```cuda
__shared__ float tile[32][32];
// Reading column: tile[0][col], tile[1][col], ..., tile[31][col]
// All access tile[i][col] which maps to bank (col % 32)
// → All 32 threads hit the same bank!
```

**Fix**: Add padding to shift the bank mapping
```cuda
__shared__ float tile[32][33];  // +1 column
// Now tile[i][col] maps to bank ((i * 33 + col) % 32)
// → Different banks for different i!
```

### Hint 2: The Complete Pattern

```cuda
__global__ void transpose_optimized(float* out, float* in, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 is key!

    // Calculate input position (coalesced)
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Read from input, coalesced
    if (x < N && y < N) {
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    }

    __syncthreads();

    // Calculate output position (transposed)
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Write to output, coalesced (reading transposed from tile)
    if (x < N && y < N) {
        out[y * N + x] = tile[threadIdx.x][threadIdx.y];
    }
}
```

### Hint 3: Understanding the Block Swap

Notice that we swap `blockIdx.x` and `blockIdx.y` when calculating output position. This ensures:
- Input block (bx, by) reads from A[by*TILE:, bx*TILE:]
- Output block writes to B[bx*TILE:, by*TILE:]
- This effectively transposes the block layout!

## Testing

### Build and Run

```bash
# Build
make

# Run naive version
./transpose_naive

# Run with profiling
ncu --set full -o naive ./transpose_naive

# Run optimized version
./transpose_optimized

# Run with profiling
ncu --set full -o optimized ./transpose_optimized

# Compare
ncu --import naive.ncu-rep optimized.ncu-rep
```

### Automated Testing

```bash
# Python test script (checks correctness and performance)
python test.py

# Should output:
# ✓ Correctness: PASSED
# ✓ Performance: PASSED (bandwidth > threshold)
# ✓ Bank conflicts: PASSED (count == 0)
```

## Profiling Checklist

Use these commands to verify your optimization:

```bash
# 1. Check bandwidth utilization
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./transpose

# 2. Check for bank conflicts
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum ./transpose

# 3. Check memory access efficiency
ncu --section MemoryWorkloadAnalysis ./transpose

# 4. Compare versions
ncu --set full -o v1 ./transpose_naive
ncu --set full -o v2 ./transpose_optimized
ncu --import v1.ncu-rep v2.ncu-rep
```

## Bonus Challenges

Once you have the basic optimization working:

### Challenge 1: Rectangular Tiles

Modify the kernel to use rectangular tiles (e.g., 32×8) to increase parallelism:

```cuda
#define TILE_DIM 32
#define BLOCK_ROWS 8
```

Each thread processes multiple rows to increase work per thread.

### Challenge 2: Vectorized Loads

Use `float4` for vectorized memory access:

```cuda
float4 val = *((float4*)&in[index]);
```

This can improve bandwidth for large matrices.

### Challenge 3: Out-of-Place vs. In-Place

Implement an in-place transpose (transposing a matrix into itself). This is much harder!

## Learning Objectives

By completing this exercise, you will:

1. ✓ Understand memory coalescing deeply
2. ✓ Master shared memory bank conflict resolution
3. ✓ Use profiling tools to guide optimization
4. ✓ Achieve near-peak memory bandwidth
5. ✓ Apply the roofline model to memory-bound kernels

## Common Mistakes

### Mistake 1: Forgetting `__syncthreads()`

```cuda
tile[ty][tx] = in[...];
// Missing: __syncthreads();
out[...] = tile[tx][ty];  // ← Race condition!
```

### Mistake 2: Wrong Padding

```cuda
__shared__ float tile[32][32];    // Bank conflicts!
__shared__ float tile[33][33];    // Wastes memory
__shared__ float tile[32][33];    // Just right ✓
```

### Mistake 3: Not Transposing Block Indices

```cuda
// Wrong: Same block indices for input and output
x_in = blockIdx.x * TILE_DIM + threadIdx.x;
x_out = blockIdx.x * TILE_DIM + threadIdx.x;  // ← Should be blockIdx.y!
```

## Resources

- [Efficient Matrix Transpose in CUDA](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)
- [Shared Memory Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- [Bank Conflicts Explained](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)

## Submission

When complete, your implementation should:
- [x] Pass correctness tests
- [x] Achieve >80% memory bandwidth utilization
- [x] Have zero shared memory bank conflicts
- [x] Be faster than naive by at least 3×

Good luck!
