# Example 03: Matrix Multiplication with Shared Memory Tiling

## Overview

This version introduces **shared memory tiling**, the most important optimization for matrix multiplication. By loading tiles of data into fast shared memory and reusing them for multiple computations, we dramatically reduce global memory traffic and achieve a major performance boost.

## Key Optimization: Data Reuse via Shared Memory

### The Memory Hierarchy

CUDA GPUs have multiple levels of memory with vastly different performance:

| Memory Type | Latency | Bandwidth | Size |
|-------------|---------|-----------|------|
| Registers | 1 cycle | ~20 TB/s | ~256 KB per SM |
| Shared Memory | ~20 cycles | ~15 TB/s | 48-164 KB per SM |
| L1 Cache | ~25 cycles | ~15 TB/s | 128 KB per SM |
| L2 Cache | ~200 cycles | ~3 TB/s | 6-40 MB |
| Global Memory | ~300 cycles | ~1 TB/s | GBs |

**Key insight**: If we can load data into shared memory and reuse it, we can:
- Reduce global memory accesses by a factor of `TILE_SIZE`
- Increase arithmetic intensity from 0.25 to 8 FLOP/byte (for TILE_SIZE=32)
- Achieve 3-4× speedup over coalesced version

## Algorithm: 1D Tiling

### High-Level Strategy

Divide the computation into tiles:

```
C[i:i+T, j:j+T] = Σ(k) A[i:i+T, k:k+T] × B[k:k+T, j:j+T]
```

Where `T = TILE_SIZE` (typically 16, 32, or 64).

### Step-by-Step Execution

For each output tile `C[blockRow, blockCol]`:

1. **Loop over K dimension** in tiles of size `TILE_SIZE`
2. **Load tile of A** into shared memory `As[][]`
3. **Load tile of B** into shared memory `Bs[][]`
4. **Synchronize threads** to ensure all data is loaded
5. **Compute partial dot product** using data from shared memory
6. **Synchronize again** before loading next tile
7. **Accumulate** results across all K tiles
8. **Write final result** to global memory

### Visual Representation

```
For C[0:32, 0:32]:

Iteration 1:         Iteration 2:         Iteration 3:
A[0:32, 0:32]       A[0:32, 32:64]       A[0:32, 64:96]
×                   ×                    ×
B[0:32, 0:32]       B[32:64, 0:32]       B[64:96, 0:32]
= Partial sum 1     + Partial sum 2      + Partial sum 3
```

Each tile is loaded **once** from global memory but used **TILE_SIZE** times in the computation.

## Implementation Details

### Kernel Structure

```cuda
#define TILE_SIZE 32

__global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
    // Shared memory for tiles (2 * 32 * 32 * 4 bytes = 8 KB)
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        // Load tiles into shared memory (coalesced access)
        As[row][col] = A[(blockRow * TILE_SIZE + row) * N + (t * TILE_SIZE + col)];
        Bs[row][col] = B[(t * TILE_SIZE + row) * N + (blockCol * TILE_SIZE + col)];

        __syncthreads();  // Wait for all threads to load data

        // Compute using shared memory (very fast!)
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[row][k] * Bs[k][col];
        }

        __syncthreads();  // Wait before loading next tile
    }

    // Write result
    C[(blockRow * TILE_SIZE + row) * N + (blockCol * TILE_SIZE + col)] = sum;
}
```

### Why This Works

**Data reuse factor**: Each element loaded into shared memory is used `TILE_SIZE` times:
- `As[row][k]` is used by all threads with the same `row` (32 threads)
- `Bs[k][col]` is used by all threads with the same `col` (32 threads)

**Memory traffic reduction**:
- **Before**: Each thread loads `2N` elements from global memory
- **After**: Each thread loads `2N / TILE_SIZE` elements from global memory
- **Reduction**: `TILE_SIZE` times less traffic!

## Building and Running

```bash
# Build
mkdir -p build && cd build
cmake ..
make

# Run with default size (2048x2048)
./matmul_tiled

# Try different sizes
./matmul_tiled 1024
./matmul_tiled 4096
```

## Expected Performance

### Performance Targets

| GPU | Peak TFLOPS | Expected GFLOPS | % of cuBLAS |
|-----|-------------|-----------------|-------------|
| A100 | 19.5 | 4,000-6,000 | 20-30% |
| V100 | 15.7 | 3,000-5,000 | 20-30% |
| RTX 3090 | 35.6 | 7,000-10,000 | 20-30% |
| RTX 2080 Ti | 13.4 | 2,500-4,000 | 20-30% |

### Sample Output

```
=== Tiled Matrix Multiplication (Shared Memory) ===
Matrix size: 2048 x 2048
Tile size: 32 x 32
Shared memory per block: 8.00 KB

=== Performance Results ===
Average execution time: X.XXX ms
Performance: 4000-6000 GFLOPS (A100)
Bandwidth utilization: ~60-70%
Compute utilization: ~20-30%
```

## Profiling

### Compare All Versions

```bash
# Profile all three versions
ncu --set full -o naive ../01_matmul_naive/matmul_naive
ncu --set full -o coalesced ../02_matmul_coalesced/matmul_coalesced
ncu --set full -o tiled ./matmul_tiled

# Compare
ncu --import naive.ncu-rep coalesced.ncu-rep tiled.ncu-rep
```

### Key Metrics

```bash
ncu --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,\
smsp__average_warps_issue_stalled_barrier.pct \
./matmul_tiled
```

### Expected Improvements

| Metric | Naive | Coalesced | Tiled | Notes |
|--------|-------|-----------|-------|-------|
| **Performance** | 300 GFLOPS | 1,200 GFLOPS | 5,000 GFLOPS | 4× jump! |
| **Memory Throughput** | 25% | 45% | 65% | Better utilization |
| **SM Throughput** | 5% | 8% | 25% | More compute-bound |
| **Shared Mem Traffic** | 0 | 0 | High | Using shared mem |
| **L2 Hit Rate** | <5% | <5% | 15-20% | Some cache benefit |
| **Warp Stalls (Barrier)** | Low | Low | ~10% | `__syncthreads()` overhead |

### Analyze Shared Memory Usage

```bash
# Check shared memory details
ncu --section SharedMemoryTables ./matmul_tiled

# Look for bank conflicts (should be minimal)
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum ./matmul_tiled
```

Expected: 0 or very few bank conflicts (our access pattern is conflict-free).

## Analysis

### Arithmetic Intensity Improvement

**Before (Coalesced)**:
- FLOPs per thread: `2N`
- Bytes loaded per thread: `2N × 4 = 8N` bytes
- Arithmetic intensity: `2N / 8N = 0.25 FLOP/byte`

**After (Tiled)**:
- FLOPs per thread: `2N` (same)
- Bytes loaded per thread from global memory: `2N / TILE_SIZE × 4 = 8N / 32` bytes
- Bytes loaded from shared memory: `2 × TILE_SIZE × 4 = 256` bytes (very fast!)
- Arithmetic intensity (global): `2N / (8N/32) = 8 FLOP/byte` (32× better!)

### Why Shared Memory is Fast

1. **On-chip**: Located on the SM, no trip to DRAM
2. **Low latency**: ~20 cycles vs. ~300 for global memory
3. **High bandwidth**: ~15 TB/s vs. ~1 TB/s for global memory
4. **Explicitly managed**: We control what gets cached

### Bottleneck Analysis

We're still not at peak performance. Why?

1. **Instruction Mix**:
   - Lots of address calculations and loop overhead
   - Not enough parallel arithmetic operations

2. **Single Tile Per Thread**:
   - Each thread only works on one element at a time
   - Can't utilize vectorization or instruction-level parallelism (ILP)

3. **Synchronization Overhead**:
   - `__syncthreads()` forces all warps to wait
   - Can cause warp stalls

4. **Memory Bank Conflicts** (if present):
   - Multiple threads accessing same bank simultaneously
   - Serializes accesses

## Tile Size Analysis

### Optimal Tile Size

The tile size affects several factors:

| Tile Size | Shared Memory | Occupancy | Reuse Factor | Best For |
|-----------|---------------|-----------|--------------|----------|
| 8 | 0.5 KB | High | 8× | Small GPUs |
| 16 | 2 KB | High | 16× | Balanced |
| 32 | 8 KB | Medium-High | 32× | Modern GPUs |
| 64 | 32 KB | Medium-Low | 64× | Large shared mem |

### Experiment: Try Different Tile Sizes

Modify `TILE_SIZE` in the source and measure:

```cpp
// Try these values:
#define TILE_SIZE 8   // Should be faster on old GPUs
#define TILE_SIZE 16  // Good balance
#define TILE_SIZE 32  // Optimal for most modern GPUs
#define TILE_SIZE 64  // May reduce occupancy
```

**Expected results**:
- Tile size 16-32 typically optimal
- Too small: Low reuse, more iterations
- Too large: Reduced occupancy, register pressure

### Profiling Different Tile Sizes

```bash
# Build with different tile sizes and compare
ncu --set full -o tile_16 ./matmul_tiled_16
ncu --set full -o tile_32 ./matmul_tiled_32
ncu --set full -o tile_64 ./matmul_tiled_64

# Check occupancy
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
./matmul_tiled_16 \
./matmul_tiled_32 \
./matmul_tiled_64
```

## What's Next?

The final optimization (example 04) combines multiple advanced techniques:

1. **2D Thread Block Tiling**: Each thread computes a small 2D tile (e.g., 8×8 elements)
2. **Vectorized Loads**: Use `float4` to load 4 elements at once
3. **Loop Unrolling**: Reduce loop overhead
4. **Register Blocking**: Keep intermediate results in registers

Expected improvement: **3-4× additional speedup** → 15,000-18,000 GFLOPS (~90% of cuBLAS)

## Key Takeaways

1. **Shared memory is crucial**: Enables data reuse, the key to high performance
2. **Tiling reduces memory traffic**: By a factor of tile size (32× improvement!)
3. **Synchronization is necessary**: But has overhead
4. **Tile size matters**: 16-32 is typically optimal
5. **Still room for improvement**: We're at ~25% of peak, can reach ~90%

## Exercise: Bank Conflict Analysis

Shared memory is divided into 32 banks. If multiple threads in a warp access the same bank (but different addresses), they serialize.

1. **Analyze our access pattern**:
   ```cuda
   As[row][k]  // row is 0-31, k varies
   Bs[k][col]  // k varies, col is 0-31
   ```

2. **Check for conflicts**:
   ```bash
   ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum ./matmul_tiled
   ```

3. **Understand why we're conflict-free**:
   - Consecutive threads access consecutive columns
   - Columns map to different banks (stride-1 access)

## Common Issues

**Q: Performance is lower than expected**
- Check your GPU's compute capability and shared memory size
- Verify tile size is appropriate for your hardware
- Ensure optimization flags are enabled

**Q: Bank conflicts detected**
- Our implementation should have none
- If you modified the code, check array indexing patterns

**Q: Occupancy is low**
- Large tile sizes use more shared memory
- Try reducing TILE_SIZE to 16
- Check `ncu --section Occupancy`

## References

- [Shared Memory Programming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- [Matrix Multiplication Case Study](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-matrix-multiplication)
- [Bank Conflicts Explained](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
- [Optimizing Matrix Multiplication](https://siboehm.com/articles/22/CUDA-MMM)
