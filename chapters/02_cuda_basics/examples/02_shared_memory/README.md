# Example 02: Shared Memory Optimization

## Overview

This example demonstrates the power of shared memory for performance optimization. Shared memory is on-chip memory that is much faster than global memory and is shared by all threads in a block.

## Why Shared Memory?

### Memory Hierarchy Performance

| Memory Type | Latency | Bandwidth |
|-------------|---------|-----------|
| Registers | 1 cycle | ~10 TB/s |
| Shared Memory | ~20-30 cycles | ~1-2 TB/s |
| L1 Cache | ~80 cycles | ~1 TB/s |
| L2 Cache | ~200 cycles | ~500 GB/s |
| Global Memory | ~400-800 cycles | ~200-900 GB/s |

**Key Insight**: Shared memory is ~10-20x faster than global memory!

## Matrix Transpose Problem

Matrix transpose demonstrates a classic memory access challenge:

```
Input (row-major):     Output (transposed):
[0  1  2  3]          [0  4  8  12]
[4  5  6  7]          [1  5  9  13]
[8  9  10 11]         [2  6  10 14]
[12 13 14 15]         [3  7  11 15]
```

### Naive Approach Problem

```cuda
// Read is coalesced (sequential)
float val = input[row * width + col];

// Write is non-coalesced (strided by height)
output[col * height + row] = val;
```

**Problem**: Non-coalesced writes reduce bandwidth by ~10x!

### Shared Memory Solution

1. Load tile into shared memory (coalesced read)
2. Transpose within shared memory
3. Write transposed tile (coalesced write)

Both reads AND writes are now coalesced!

## Kernels in This Example

### 1. Naive Transpose

```cuda
__global__ void transposeNaive(float *input, float *output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        output[col * height + row] = input[row * width + col];
    }
}
```

**Characteristics:**
- Coalesced reads from input
- Non-coalesced writes to output
- No shared memory usage
- Lowest performance

### 2. Shared Memory Transpose

```cuda
__global__ void transposeShared(float *input, float *output, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM];

    // Load tile (coalesced)
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    tile[threadIdx.y][threadIdx.x] = input[row * width + col];

    __syncthreads();  // Wait for all loads

    // Write transposed tile (coalesced)
    col = blockIdx.y * TILE_DIM + threadIdx.x;
    row = blockIdx.x * TILE_DIM + threadIdx.y;
    output[row * height + col] = tile[threadIdx.x][threadIdx.y];
}
```

**Characteristics:**
- Coalesced reads AND writes
- Uses shared memory for transposition
- ~2-5x faster than naive
- May have bank conflicts

### 3. Bank Conflict Optimization

```cuda
__shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 padding
```

**Why padding helps:**
- Shared memory has 32 banks
- Without padding: `tile[i][j]` and `tile[i][j+32]` → same bank
- With padding: Bank assignments shift, avoiding conflicts

## Shared Memory Concepts

### Declaration

```cuda
// Static allocation (size known at compile time)
__shared__ float data[256];

// Dynamic allocation (size specified at kernel launch)
extern __shared__ float data[];

kernel<<<grid, block, shared_bytes>>>(args);
```

### Bank Conflicts

Shared memory is divided into 32 banks (4-byte words).

**No Conflict:**
```cuda
__shared__ float data[32];
float val = data[threadIdx.x];  // All threads access different banks
```

**Conflict:**
```cuda
__shared__ float data[32][32];
float val = data[threadIdx.x][0];  // All threads access bank 0
```

**Resolution:**
```cuda
__shared__ float data[32][33];  // Padding shifts bank mapping
float val = data[threadIdx.x][0];  // Now distributed across banks
```

### Synchronization

`__syncthreads()` is a barrier that:
- Waits for all threads in the block
- Ensures memory consistency
- Must be reached by all threads (no divergent paths!)

```cuda
// CORRECT
__shared__ float data[256];
data[threadIdx.x] = /* load */;
__syncthreads();
float val = data[255 - threadIdx.x];

// WRONG - deadlock!
if (threadIdx.x < 128) {
    __syncthreads();  // Only some threads reach this
}
```

## Matrix Multiplication Example

Shared memory is essential for efficient matrix multiplication:

```cuda
// Without shared memory: N global memory reads per element
// With shared memory: 2*sqrt(N) global memory reads per element
```

**Tiling Strategy:**
1. Load tile of A and B into shared memory
2. Compute partial product using shared data
3. Accumulate results
4. Move to next tile

**Benefits:**
- Each element loaded once, reused TILE_DIM times
- ~10-20x reduction in global memory traffic
- Approaches theoretical peak performance

## Expected Performance

On a typical modern GPU (e.g., RTX 3080):

| Kernel | Bandwidth | Speedup |
|--------|-----------|---------|
| Naive | ~150 GB/s | 1.0x |
| Shared Memory | ~450 GB/s | 3.0x |
| No Bank Conflicts | ~550 GB/s | 3.7x |

**Note**: Results vary by GPU architecture and memory clock.

## Building and Running

```bash
mkdir build && cd build
cmake ..
make

./shared_mem
```

## Expected Output

```
=== CUDA Shared Memory Examples ===

Matrix size: 2048 x 2048 (16.00 MB)
Tile size: 32 x 32

1. Naive Transpose (no optimization)
-------------------------------------
  Correctness: PASSED
  Time: 1.234 ms
  Bandwidth: 195.45 GB/s

2. Shared Memory Transpose
---------------------------
  Correctness: PASSED
  Time: 0.567 ms
  Bandwidth: 425.67 GB/s
  Speedup: 2.18x

3. Shared Memory + No Bank Conflicts
-------------------------------------
  Correctness: PASSED
  Time: 0.456 ms
  Bandwidth: 529.82 GB/s
  Speedup over naive: 2.71x
  Speedup over shared: 1.24x
```

## Common Pitfalls

### 1. Missing Synchronization

```cuda
// WRONG
__shared__ float data[256];
data[threadIdx.x] = input[idx];
// Missing __syncthreads()
float val = data[otherIdx];  // May read uninitialized data!
```

### 2. Synchronization in Divergent Code

```cuda
// WRONG - deadlock!
if (threadIdx.x < 128) {
    __syncthreads();
}

// CORRECT
if (threadIdx.x < 128) {
    /* work */
}
__syncthreads();
```

### 3. Shared Memory Size Limits

```cuda
// May fail on GPUs with < 64KB shared memory
__shared__ float huge[16384];  // 64KB
```

Check limits:
```bash
nvidia-smi --query-gpu=compute_cap,memory.total --format=csv
```

### 4. Bank Conflicts Not Considered

Always profile to detect bank conflicts:
```bash
nvprof --metrics shared_load_transactions_per_request ./shared_mem
```

Ideal value: 1.0 (no conflicts)

## Performance Tips

1. **Tile Size**
   - 32x32 is common (1024 threads)
   - 16x16 for compute-heavy kernels
   - Must fit in shared memory

2. **Padding**
   - Add +1 column for square tiles
   - Prevents bank conflicts during transpose

3. **Occupancy**
   - Shared memory usage affects occupancy
   - Use `--ptxas-options=-v` to check
   - Balance shared memory vs. occupancy

4. **Data Reuse**
   - Maximize reuse of loaded data
   - Each load should be used multiple times
   - Tiling enables reuse

## Advanced Concepts

### Dynamic Shared Memory

```cuda
extern __shared__ float sdata[];

// Launch with specified shared memory size
kernel<<<grid, block, bytes>>>(args);
```

### Shared Memory Configuration

```cuda
// Prefer L1 cache
cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);

// Prefer shared memory
cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferShared);
```

### Warp-Level Access Patterns

- Threads in a warp access shared memory in lockstep
- Consider warp-level patterns for best performance
- Use `__shfl_*` intrinsics when appropriate

## Next Steps

- Study reduction patterns (Example 03)
- Learn about atomic operations (Example 04)
- Implement tiled matrix multiplication
- Profile with Nsight Compute for detailed analysis

## References

- Mark Harris: "Using Shared Memory in CUDA C/C++"
- NVIDIA: "CUDA C Programming Guide - Shared Memory"
- "Optimizing Parallel Reduction in CUDA" (reduction.pdf)
