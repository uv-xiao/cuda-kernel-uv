# Example 02: Matrix Multiplication with Coalesced Access

## Overview

This version improves upon the naive implementation by ensuring all global memory accesses are coalesced. We achieve this by pre-transposing matrix B, so both A and B^T can be accessed row-wise.

## Key Optimization: Memory Coalescing

### The Problem with Naive Implementation

In the naive version:
- **Matrix A**: Row-wise access → Coalesced ✓
- **Matrix B**: Column-wise access → Strided, uncoalesced ✗

Column-wise access means threads in a warp read addresses N elements apart, causing up to 32 separate memory transactions instead of 1.

### The Solution

Pre-transpose B so we compute `C = A × B^T^T = A × B`:

- **Matrix A**: Row-wise access → Coalesced ✓
- **Matrix B^T**: Row-wise access → Coalesced ✓

Now all accesses are coalesced, dramatically improving memory bandwidth utilization.

## Algorithm

```
1. Transpose B → B^T on GPU
2. Compute C = A × (B^T)^T using coalesced accesses
```

For element `C[i,j]`:
```
C[i,j] = Σ(k=0 to N-1) A[i,k] * B[k,j]
       = Σ(k=0 to N-1) A[i,k] * B^T[j,k]  // Both are row accesses!
```

## Implementation Details

### Efficient Matrix Transpose

We use shared memory for efficient, bank-conflict-free transposition:

```cuda
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose(const float* input, float* output, int N) {
    // Shared memory with padding to avoid bank conflicts
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // Read from input (coalesced)
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < N && (y + j) < N) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * N + x];
        }
    }

    __syncthreads();

    // Write to output, transposed (coalesced)
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < N && (y + j) < N) {
            output[(y + j) * N + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}
```

**Key techniques**:
1. **Tiled approach**: Process 32×32 tiles in shared memory
2. **Padding**: `[TILE_DIM][TILE_DIM + 1]` avoids bank conflicts
3. **Coalesced I/O**: Both reads and writes are coalesced

### Improved MatMul Kernel

```cuda
__global__ void matmul_coalesced(const float* A, const float* B_T, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            // Both accesses are now coalesced!
            sum += A[row * N + k] * B_T[col * N + k];
        }
        C[row * N + col] = sum;
    }
}
```

## Building and Running

```bash
# Build
mkdir -p build && cd build
cmake ..
make

# Run
./matmul_coalesced

# With custom size
./matmul_coalesced 4096
```

## Expected Performance

### Speedup

- **vs. Naive**: ~3-4× faster
- **Memory bandwidth**: 40-50% utilization (vs. 15-25% naive)
- **Absolute performance**: 1,000-1,500 GFLOPS on A100

### Sample Output

```
=== Coalesced Matrix Multiplication ===
Matrix size: 2048 x 2048

Transposing matrix B...

=== Performance Results ===
Average execution time: X.XXX ms
Performance: 1000-1500 GFLOPS
Bandwidth utilization: ~40-50%
```

## Profiling

### Compare with Naive Version

```bash
# Profile naive version
ncu --set full -o naive ../01_matmul_naive/matmul_naive

# Profile coalesced version
ncu --set full -o coalesced ./matmul_coalesced

# Compare side-by-side
ncu --import naive.ncu-rep coalesced.ncu-rep
```

### Key Metrics to Check

```bash
ncu --metrics \
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed \
./matmul_coalesced
```

### Expected Improvements

| Metric | Naive | Coalesced | Improvement |
|--------|-------|-----------|-------------|
| **Global Load Efficiency** | ~50% | ~90-100% | 2× better |
| **Memory Throughput** | 20-35% | 40-50% | ~2× better |
| **Memory Replay Overhead** | High | Low | Much better |
| **L1 Cache Hit Rate** | <5% | <5% | (Still low) |
| **Overall Performance** | 300-500 GFLOPS | 1000-1500 GFLOPS | 3-4× faster |

### Detailed Memory Analysis

```bash
# See memory access pattern details
ncu --section MemoryWorkloadAnalysis ./matmul_coalesced

# Check coalescing specifically
ncu --section MemoryWorkloadAnalysis_Chart ./matmul_coalesced
```

Look for:
- **Global Load/Store Transactions**: Should be minimized
- **Sector Utilization**: Should be near 100%
- **L2 Cache Hit Rate**: Still low (no reuse yet)

## Analysis

### Why This is Better

**Before (Naive)**:
```
Warp reads B[0,j], B[1,j], ..., B[31,j]
Memory addresses: j, N+j, 2N+j, ..., 31N+j
Stride: N elements (128 bytes for float on typical N)
Result: 32 separate 32-byte transactions = 1024 bytes
Efficiency: 128 bytes needed / 1024 bytes loaded = 12.5%
```

**After (Coalesced)**:
```
Warp reads B_T[j,0], B_T[j,1], ..., B_T[j,31]
Memory addresses: j*N, j*N+1, j*N+2, ..., j*N+31
Stride: 1 element (4 bytes)
Result: 1 coalesced 128-byte transaction
Efficiency: 128 bytes needed / 128 bytes loaded = 100%
```

### Remaining Bottlenecks

Despite the improvement, we're still far from optimal:

1. **No Data Reuse**:
   - Each element of A is read N times (once per column)
   - Each element of B^T is read N times (once per row)
   - All reads come from slow global memory

2. **Low Arithmetic Intensity**:
   - 2 FLOPs per 8 bytes loaded
   - Ratio: 0.25 FLOP/byte
   - GPU can do ~100+ FLOP/byte if we cache data

3. **Cache is Ineffective**:
   - L1/L2 cache can't hold entire rows/columns
   - No temporal locality at cache level

### Why Not Better Performance?

You might wonder: "We fixed coalescing, why aren't we at 100% bandwidth?"

**Answer**: We're loading the same data repeatedly!

For an N×N matrix multiply:
- **Minimum data movement**: `3N²` elements (read A, B once, write C once)
- **Actual data movement**: `2N³` elements (each of A and B read N times)
- **Overhead**: N× more data than necessary

## What's Next?

The next optimization (example 03) uses **shared memory tiling** to:

1. Load tiles of A and B into fast shared memory
2. Reuse those tiles for multiple computations
3. Reduce global memory traffic by factor of tile size

Expected improvement: **3-4× additional speedup** → 4,000-6,000 GFLOPS

## Exercise: Analyze Memory Traffic

1. **Calculate theoretical memory traffic**:

   For N=2048 with this implementation:
   - Reads from A: `2048² × 2048 = 8.6 billion` floats
   - Reads from B^T: `2048² × 2048 = 8.6 billion` floats
   - Writes to C: `2048²` floats
   - Total: ~69 GB of data

2. **Measure actual bandwidth**:

   ```bash
   ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum ./matmul_coalesced 2048
   ```

3. **Compare with peak**:

   If GPU has 900 GB/s peak bandwidth:
   - Time to transfer 69 GB: 69/900 = 77 ms
   - Actual kernel time: ~150 ms
   - Efficiency: 77/150 = 51% (matches expectations!)

## Key Takeaways

1. **Coalescing is critical**: 3-4× performance improvement just from access pattern
2. **All accesses should be coalesced**: Both reads and writes
3. **Transpose can help**: Sometimes pre-processing data is worthwhile
4. **Coalescing ≠ optimal**: We're still memory-bound, just less so
5. **Data reuse is the next step**: Shared memory tiling is the solution

## References

- [Memory Coalescing Guide](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)
- [Efficient Matrix Transpose](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)
- [CUDA Memory Model](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory)
