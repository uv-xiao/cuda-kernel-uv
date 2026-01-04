# Exercise 02: FP16 GEMM with Tensor Cores

## Problem Statement

Implement a high-performance FP16 GEMM kernel using WMMA (Warp Matrix Multiply-Accumulate) Tensor Cores. Target: achieve >90% of cuBLAS FP16 performance.

## Learning Objectives

- Master WMMA API for Tensor Cores
- Understand FP16/FP32 mixed precision
- Optimize for Tensor Core utilization
- Achieve production-level performance

## Specifications

### Input

- **A**: Matrix of shape `(M, K)`, dtype `half` (FP16)
- **B**: Matrix of shape `(K, N)`, dtype `half` (FP16)
- **C**: Matrix of shape `(M, N)`, dtype `half` (FP16), output

### Requirements

1. **Use WMMA API**: Required for Tensor Cores
2. **Tile size**: 16x16x16 (WMMA fragment size)
3. **Mixed precision**: FP16 input, FP32 accumulation, FP16 output
4. **Performance**: >90% of cuBLAS FP16 on matrices ≥1024x1024
5. **Correctness**: Max error < 0.1 (FP16 has lower precision)

## WMMA Basics

### Fragment Declaration

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

// Matrix A fragment (16x16x16, FP16, row-major)
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;

// Matrix B fragment (16x16x16, FP16, row-major)
fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;

// Accumulator fragment (16x16x16, FP32 for precision)
fragment<accumulator, 16, 16, 16, float> acc_frag;
```

### WMMA Operations

```cpp
// Initialize accumulator to zero
fill_fragment(acc_frag, 0.0f);

// Load from shared memory (or global)
load_matrix_sync(a_frag, smem_A + offset, leading_dim);
load_matrix_sync(b_frag, smem_B + offset, leading_dim);

// Matrix multiply-accumulate
mma_sync(acc_frag, a_frag, b_frag, acc_frag);

// Store result (convert FP32 -> FP16)
fragment<accumulator, 16, 16, 16, half> c_frag;
for (int i = 0; i < acc_frag.num_elements; ++i) {
    c_frag.x[i] = __float2half(acc_frag.x[i]);
}
store_matrix_sync(smem_C + offset, c_frag, leading_dim, mem_row_major);
```

## Implementation Strategy

### 1. Block and Warp Organization

```cpp
// Block tile size (must be multiple of 16)
constexpr int BM = 128;  // Block M
constexpr int BN = 128;  // Block N
constexpr int BK = 32;   // Block K

// WMMA tile size (fixed)
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Warps per block
constexpr int WARPS_M = BM / WMMA_M;  // 8 warps
constexpr int WARPS_N = BN / WMMA_N;  // 8 warps
constexpr int WARPS_PER_BLOCK = WARPS_M * WARPS_N;  // 64 warps

// Threads per block
constexpr int THREADS = WARPS_PER_BLOCK * 32;  // 2048 threads
```

### 2. Shared Memory Layout

```cpp
__shared__ half smem_A[BM * BK];
__shared__ half smem_B[BK * BN];

// Consider swizzling to avoid bank conflicts
```

### 3. Warp Positioning

```cpp
int warp_id = threadIdx.x / 32;
int warp_m = warp_id / WARPS_N;  // Warp row
int warp_n = warp_id % WARPS_N;  // Warp column

// Global output position for this warp
int warp_m_offset = blockIdx.x * BM + warp_m * WMMA_M;
int warp_n_offset = blockIdx.y * BN + warp_n * WMMA_N;
```

### 4. Main Loop

```cpp
// Initialize accumulator
fill_fragment(acc_frag, 0.0f);

for (int k_tile = 0; k_tile < K; k_tile += BK) {
    // 1. Cooperatively load A and B tiles to shared memory
    //    (all threads in block participate)

    __syncthreads();

    // 2. Each warp computes its 16x16 output tile
    for (int k = 0; k < BK; k += WMMA_K) {
        // Load WMMA fragments from shared memory
        load_matrix_sync(a_frag, smem_A + ..., BK);
        load_matrix_sync(b_frag, smem_B + ..., BN);

        // Accumulate
        mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    __syncthreads();
}

// 3. Store accumulated result to global memory
```

## Optimization Checklist

- [ ] Use FP32 accumulation (higher precision)
- [ ] Ensure 16-byte alignment for all pointers
- [ ] Swizzle shared memory to avoid bank conflicts
- [ ] Cooperative loading of tiles (all threads participate)
- [ ] Verify warp-level synchronization (no explicit __syncwarp() needed)
- [ ] Maximize Tensor Core utilization (large tiles)

## Test Cases

Test with various matrix sizes:

1. **1024x1024x1024**: Baseline performance
2. **2048x2048x2048**: Larger working set
3. **4096x4096x4096**: Maximum performance
4. **Non-square**: M=1024, N=2048, K=512

## Performance Targets

On NVIDIA A100 GPU:

| Matrix Size | Target TFLOPS | % of cuBLAS |
|-------------|---------------|-------------|
| 1024³ | 220+ | 90%+ |
| 2048³ | 250+ | 92%+ |
| 4096³ | 270+ | 95%+ |

On NVIDIA V100 GPU:

| Matrix Size | Target TFLOPS | % of cuBLAS |
|-------------|---------------|-------------|
| 1024³ | 90+ | 85%+ |
| 2048³ | 105+ | 90%+ |
| 4096³ | 115+ | 92%+ |

## Common Pitfalls

1. **Alignment**: WMMA requires 16-byte aligned addresses
   ```cpp
   assert((uintptr_t)ptr % 16 == 0);
   ```

2. **Layout mismatch**: Fragment layout must match memory layout
   ```cpp
   // If memory is row-major:
   fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
   ```

3. **Leading dimension**: Must match actual stride
   ```cpp
   load_matrix_sync(a_frag, smem_A, BK);  // BK is leading dimension
   ```

4. **Warp divergence**: All threads in warp must execute WMMA ops together

5. **FP16 overflow**: Use FP32 accumulation to avoid overflow

## Profiling

Check Tensor Core utilization:

```bash
ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active \
    ./fp16_gemm
```

Target: >80% utilization

Check memory efficiency:

```bash
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct \
./fp16_gemm
```

## Hints

1. **Start from gemm_tiled.cu**: Adapt FP32 tiled GEMM
2. **Replace inner loop**: Use WMMA instead of scalar FMAs
3. **Handle warps**: Each warp is independent
4. **Test incrementally**: Verify correctness before optimizing

## Bonus Challenges

1. **Async copy**: Use `cp.async` for data loading (Ampere+)
2. **Double buffering**: Overlap compute and memory transfer
3. **BF16 support**: Support bfloat16 (Ampere+)
4. **Dynamic shapes**: Handle non-multiple-of-16 dimensions

## Evaluation Criteria

- **Correctness** (30%): Passes all test cases
- **Performance** (50%): Meets TFLOPS targets
- **Code quality** (20%): Clean, well-documented code

## Resources

- [WMMA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
- [Example 04: Tensor Cores](../../examples/04_tensor_cores/)
- [CUTLASS WMMA GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples)
