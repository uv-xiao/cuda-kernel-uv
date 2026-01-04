# Example 01: Naive Matrix Multiplication

## Overview

This is the baseline implementation of matrix multiplication. Each thread computes one element of the output matrix by computing the dot product of a row from matrix A and a column from matrix B.

## Algorithm

For matrices `C = A × B` where all are `N × N`:

```
C[i,j] = Σ(k=0 to N-1) A[i,k] * B[k,j]
```

**Thread mapping**: Thread `(i,j)` computes `C[i,j]`

## Implementation Details

### Kernel Structure

```cuda
__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

### Memory Access Pattern

**Matrix A (row-wise access)**:
- Thread `(i, j)` reads `A[i, 0]`, `A[i, 1]`, ..., `A[i, N-1]`
- Threads in the same warp access consecutive memory addresses
- **Result**: Coalesced access ✓

**Matrix B (column-wise access)**:
- Thread `(i, j)` reads `B[0, j]`, `B[1, j]`, ..., `B[N-1, j]`
- Threads in the same warp access addresses N elements apart
- **Result**: Non-coalesced, strided access ✗

### Launch Configuration

- **Block size**: 32 × 32 = 1,024 threads (maximum for most GPUs)
- **Grid size**: `⌈N/32⌉ × ⌈N/32⌉` blocks
- **Total threads**: Equal to number of output elements (`N²`)

## Building and Running

### Build

```bash
# From this directory
mkdir -p build && cd build
cmake ..
make

# Or from the chapter root
cd /home/uvxiao/cuda-kernel-tutorial/chapters/03_profiling
mkdir -p build && cd build
cmake ..
make
```

### Run

```bash
# Default size (2048x2048)
./matmul_naive

# Custom size
./matmul_naive 4096
```

### Expected Output

```
=== Naive Matrix Multiplication ===
Matrix size: 2048 x 2048
Total elements: 4194304
Memory per matrix: 16.00 MB

Kernel configuration:
  Block size: 32 x 32 = 1024 threads
  Grid size: 64 x 64 = 4096 blocks
  Total threads: 4194304

=== Performance Results ===
Average execution time: X.XXX ms
Performance: 300-500 GFLOPS
Effective bandwidth: XX.XX GB/s
Peak memory bandwidth: XXX.XX GB/s
Bandwidth utilization: ~15-25%
```

## Profiling

### Basic Profiling with Nsight Compute

```bash
# Quick profile (key metrics only)
ncu --set basic ./matmul_naive

# Full profile (all metrics)
ncu --set full ./matmul_naive

# Save results for later analysis
ncu --set full -o naive_profile ./matmul_naive

# Focus on memory metrics
ncu --section MemoryWorkloadAnalysis ./matmul_naive

# Compare with different matrix sizes
ncu --set full -o naive_2048 ./matmul_naive 2048
ncu --set full -o naive_4096 ./matmul_naive 4096
```

### Key Metrics to Examine

Run this command to see specific metrics:

```bash
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct \
./matmul_naive
```

### Expected Metrics

| Metric | Expected Value | Interpretation |
|--------|----------------|----------------|
| **SM Throughput** | 5-15% | Very low compute utilization |
| **Memory Throughput** | 20-35% | Poor memory bandwidth usage |
| **Global Load Efficiency** | ~50% | Half the loaded data is wasted |
| **Global Store Efficiency** | ~100% | Writes are coalesced |
| **L1 Cache Hit Rate** | <5% | Almost no cache reuse |
| **Warp Execution Efficiency** | ~100% | No divergence (good!) |
| **Occupancy** | 50-100% | Good, but doesn't help |

### Profiling with Nsight Systems

```bash
# System-level timeline
nsys profile --stats=true -o naive_timeline ./matmul_naive

# Focus on GPU activity
nsys profile --trace=cuda,nvtx --stats=true ./matmul_naive

# View the report (opens GUI)
nsys-ui naive_timeline.nsys-rep
```

## Performance Analysis

### Theoretical Performance

For `N = 2048`:
- **FLOPs**: `2 × 2048³ = 17.2 billion` operations
- **Memory**: Minimum `3 × 2048² × 4 bytes = 48 MB`
- **Arithmetic Intensity**: `17.2 GFLOP / 48 MB ≈ 358 FLOP/byte`

This is extremely compute-heavy, so we should be compute-bound. However...

### Actual Bottleneck: Memory Bandwidth

The naive implementation doesn't reuse data:
- Each element of A is loaded N times (once per column of B)
- Each element of B is loaded N times (once per row of A)
- **Actual memory traffic**: `N² × N = N³` loads (not `N²`!)
- **Actual arithmetic intensity**: `2 FLOPs / 8 bytes = 0.25 FLOP/byte`

This makes the kernel **severely memory-bound**.

### Why Performance is Poor

1. **Strided Access to B**:
   - Threads in a warp read from addresses N elements apart
   - This causes 32 separate memory transactions instead of 1
   - ~32× memory bandwidth loss

2. **No Data Reuse**:
   - Each value is fetched from global memory every time it's used
   - No use of shared memory or registers to cache frequently accessed data

3. **Low Cache Hit Rate**:
   - Matrix B access pattern defeats L1/L2 cache
   - Working set is too large to fit in cache

### Performance Comparison

Expected performance on different GPUs (N=4096):

| GPU | Peak TFLOPS | Expected GFLOPS | % of Peak |
|-----|-------------|-----------------|-----------|
| A100 | 19.5 | 300-500 | 1.5-2.5% |
| V100 | 15.7 | 250-400 | 1.6-2.5% |
| RTX 3090 | 35.6 | 400-700 | 1.1-2.0% |
| RTX 2080 Ti | 13.4 | 200-350 | 1.5-2.6% |

## What's Next?

The next example (`02_matmul_coalesced`) addresses the strided access pattern by:
1. Transposing matrix B before multiplication
2. Ensuring all accesses are coalesced

Expected improvement: **3-4× speedup** (20-30% memory bandwidth utilization)

## Key Takeaways

1. **Simple doesn't mean efficient**: The straightforward implementation is far from optimal
2. **Memory access patterns matter**: Strided access destroys performance
3. **Profile before optimizing**: Metrics confirm memory is the bottleneck
4. **Occupancy isn't everything**: High occupancy but still poor performance

## Exercise

1. **Profile the kernel** with different block sizes (16×16, 32×32, 64×64... wait, 64×64 = 4096 threads, which exceeds the 1024 limit!)

   Try: 16×16, 32×32, and 16×64

   ```bash
   # Modify blockDim in the source and rebuild, or add it as a parameter
   ncu --set full -o profile_16x16 ./matmul_naive
   ```

2. **Check metrics**:
   - Which block size gives best performance?
   - How does occupancy change?
   - Does memory bandwidth utilization improve?

3. **Analyze the memory access pattern**:
   - Use `ncu --section MemoryWorkloadAnalysis` to see detailed memory metrics
   - Look at "Global Memory Access Pattern" in the report
   - Identify which loads are uncoalesced

## Common Issues

**Q: My performance is even worse than expected**
- Check if GPU is in compute mode: `nvidia-smi`
- Ensure no other processes are using GPU
- Verify optimizations are enabled in CMakeLists.txt

**Q: Verification fails for large matrices**
- This is expected due to floating-point precision
- Use smaller epsilon or skip verification for N > 512

**Q: Out of memory error**
- Reduce matrix size: `./matmul_naive 1024`
- Check available memory: `nvidia-smi`

## References

- [CUDA C++ Programming Guide - Memory Access](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)
- [Memory Coalescing Blog Post](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)
- [Nsight Compute Metrics Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-guide)
