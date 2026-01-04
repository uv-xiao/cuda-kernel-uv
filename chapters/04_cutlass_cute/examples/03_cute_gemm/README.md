# Example 03: GEMM with CuTe

## Overview

This example implements General Matrix Multiplication (GEMM) using CuTe primitives. You'll build GEMM kernels from scratch, starting with a naive implementation and progressing to a tiled, shared-memory version that achieves competitive performance.

## Learning Objectives

- Implement GEMM using CuTe tensors and layouts
- Apply tiling strategies for cache efficiency
- Use shared memory with CuTe abstractions
- Optimize memory access patterns for coalescing
- Achieve 60-80% of cuBLAS performance (FP32)
- Understand performance bottlenecks via profiling

## GEMM Background

**Operation**: `C = α·A·B + β·C`

Where:
- `A`: M x K matrix
- `B`: K x N matrix
- `C`: M x N matrix
- `α, β`: scalar coefficients

**Computational complexity**: O(2·M·N·K) FLOPs

**Key performance considerations:**
1. **Memory bandwidth**: For small matrices, memory-bound
2. **Compute**: For large matrices, compute-bound
3. **Cache utilization**: Tile to fit in L1/shared memory
4. **Thread cooperation**: Use shared memory for data reuse

## Files in This Example

### gemm_simple.cu

A straightforward GEMM implementation using CuTe:
- Per-thread computation without shared memory
- Demonstrates basic CuTe tensor usage in kernels
- Useful for understanding concepts before optimization
- Performance: ~10-20% of cuBLAS (expected)

### gemm_tiled.cu

Optimized GEMM with tiling and shared memory:
- Block-level tiling for shared memory
- Thread-level tiling for registers
- Coalesced global memory access
- Bank conflict avoidance
- Performance: 60-80% of cuBLAS (target)

## Building and Running

```bash
mkdir build && cd build
cmake .. -DCUTLASS_DIR=$CUTLASS_DIR
make -j$(nproc)

# Run simple GEMM
./gemm_simple

# Run tiled GEMM
./gemm_tiled

# Run with NSight Compute profiling
ncu --set full -o gemm_profile ./gemm_tiled
```

## Tiling Strategy

### Hierarchical Decomposition

```
Global Matrix (M x N)
    ↓
Block Tiles (BM x BN)
    ↓
Warp Tiles (WM x WN)
    ↓
Thread Tiles (TM x TN)
```

### Example Configuration

For M=N=K=1024:

```cpp
constexpr int BM = 128;  // Block tile M
constexpr int BN = 128;  // Block tile N
constexpr int BK = 8;    // Block tile K
constexpr int TM = 8;    // Thread tile M
constexpr int TN = 8;    // Thread tile N
```

**Threads per block**: (BM/TM) × (BN/TN) = 16 × 16 = 256

### Memory Layout

```
Global Memory (DRAM)
    ↓ Coalesced loads
Shared Memory (On-chip)
    ↓ Low-latency access
Registers (Per-thread)
    ↓ FMA operations
Accumulator (Registers)
```

## Performance Analysis

### Theoretical Peak

For NVIDIA A100 (FP32):
- Peak FP32: 19.5 TFLOPS
- Memory bandwidth: 1555 GB/s

**Arithmetic intensity** (FLOPs per byte):
```
AI = (2·M·N·K) / (4·(M·K + K·N + M·N))
```

For M=N=K=1024:
```
AI = 2·1024³ / (4·(1024² + 1024² + 1024²))
   = 2,147,483,648 / 12,582,912
   = 170.67 FLOPs/byte
```

**Compute-bound** (AI > 78 for A100)

### Expected Performance

| Implementation | TFLOPS | % of Peak | Notes |
|----------------|--------|-----------|-------|
| gemm_simple | 1-2 | 5-10% | No tiling, poor cache use |
| gemm_tiled | 12-15 | 60-80% | Tiled, shared memory |
| cuBLAS | 18-19 | 95%+ | Highly optimized |

### Profiling Metrics

Use NSight Compute to measure:

```bash
ncu --metrics sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct \
./gemm_tiled
```

**Target metrics:**
- **FMA utilization**: >80%
- **Global load efficiency**: >80% (coalescing)
- **Shared memory bank conflicts**: <5%
- **Occupancy**: >50%

## Algorithm Walkthrough

### Simple GEMM (gemm_simple.cu)

```cpp
// Each thread computes one element of C
C(i, j) = 0
for k in range(K):
    C(i, j) += A(i, k) * B(k, j)
```

**Problems:**
- No data reuse (each element of A and B loaded multiple times)
- Poor cache locality
- Low arithmetic intensity

### Tiled GEMM (gemm_tiled.cu)

```cpp
// Each block computes BM x BN tile of C
for each K tile:
    // Cooperatively load A and B tiles to shared memory
    load A_tile[BM x BK] from global to shared
    load B_tile[BK x BN] from global to shared
    __syncthreads()

    // Each thread computes TM x TN sub-tile
    for each element in thread tile:
        accumulate using shared memory
    __syncthreads()

// Write accumulated results to C
```

**Benefits:**
- A and B tiles loaded once, reused by all threads in block
- Shared memory provides low-latency access
- Coalesced global memory access
- Higher arithmetic intensity

## CuTe-Specific Patterns

### 1. Tensor Partitioning

```cpp
// Partition global tensor among threads
auto gA = make_tensor(A_ptr, make_layout(make_shape(M, K), make_stride(K, 1)));
auto tiled_gA = local_partition(gA, thread_layout, thread_idx);
```

### 2. Shared Memory Tensors

```cpp
// Declare shared memory with CuTe layout
__shared__ float smem_A[BM * BK];
auto sA = make_tensor(make_smem_ptr(smem_A),
                      make_layout(make_shape(BM, BK), make_stride(BK, 1)));
```

### 3. Cooperative Copy

```cpp
// All threads in block cooperatively copy tile
auto copy_op = make_tiled_copy(Copy_Atom<DefaultCopy, float>{},
                                thread_layout,
                                val_layout);
copy(copy_op, src_tensor, dst_tensor);
```

### 4. Thread-Local Accumulation

```cpp
// Accumulate in registers
Tensor acc = make_fragment_like(partition_C);
clear(acc);

// FMA loop
for (int k = 0; k < BK; ++k) {
    gemm(tiled_mma, acc, sA_partition, sB_partition);
}
```

## Optimization Checklist

- [ ] Use shared memory for A and B tiles
- [ ] Ensure coalesced global memory access (stride-1 innermost)
- [ ] Avoid shared memory bank conflicts (swizzle if needed)
- [ ] Maximize register reuse (thread-level tiling)
- [ ] Overlap computation and memory transfer (double buffering)
- [ ] Tune block dimensions for occupancy
- [ ] Use appropriate tile sizes (BM, BN, BK)
- [ ] Validate correctness against cuBLAS

## Common Pitfalls

1. **Missing synchronization**: Forgetting `__syncthreads()` after shared memory loads
2. **Bank conflicts**: All threads accessing same shared memory column
3. **Poor coalescing**: Non-contiguous global memory access
4. **Register spilling**: Too many registers per thread (reduce TM/TN)
5. **Low occupancy**: Block too large or too much shared memory

## Debugging Tips

### Verify Correctness

```cpp
// Compare with cuBLAS
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K, &alpha, B, N, A, K, &beta, C_ref, N);

// Check max error
float max_error = 0.0f;
for (int i = 0; i < M * N; ++i) {
    max_error = max(max_error, abs(C[i] - C_ref[i]));
}
```

### Profile with NSight Compute

```bash
# Launch count
ncu --metrics launch__registers_per_thread ./gemm_tiled

# Memory metrics
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum ./gemm_tiled

# Compute metrics
ncu --metrics sm__sass_thread_inst_executed_op_ffma_pred_on.sum ./gemm_tiled
```

## Performance Tuning

### Tile Size Selection

**Constraints:**
- Shared memory: `2 * BM * BK + 2 * BK * BN < 48KB` (typical)
- Registers: `TM * TN * 2 < 64` (per thread, approximate)
- Threads per block: `(BM/TM) * (BN/TN) < 1024`

**Heuristics:**
- Start with BM=BN=128, BK=8
- Start with TM=TN=8
- Adjust based on profiling (occupancy, shared memory usage)

### Occupancy Optimization

```bash
# Check occupancy
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./gemm_tiled
```

Target: >50% occupancy

If low:
- Reduce shared memory usage (decrease BM, BN, BK)
- Reduce register usage (decrease TM, TN)
- Increase threads per block

## Next Steps

After completing this example:

1. **Profile your implementation**: Compare with cuBLAS
2. **Experiment with tile sizes**: Find optimal configuration for your GPU
3. **Proceed to Example 04**: Learn Tensor Cores for 10x speedup on FP16
4. **Read CUTLASS GEMM**: Study production-quality implementation

## References

- [CUTLASS GEMM Documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md)
- [How to Optimize GEMM](https://siboehm.com/articles/22/CUDA-MMM)
- [NSight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
