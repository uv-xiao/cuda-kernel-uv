# Example 04: Tensor Cores

## Overview

Tensor Cores are specialized hardware units on NVIDIA GPUs (Volta+) that accelerate matrix multiply-accumulate operations. This example demonstrates how to use Tensor Cores via WMMA (Warp Matrix Multiply-Accumulate) API and PTX MMA instructions for 8-16x speedup over CUDA cores.

## Learning Objectives

- Understand Tensor Core architecture and capabilities
- Use WMMA API (Volta/Turing compatible)
- Use MMA PTX instructions (Ampere+ optimized)
- Achieve >90% cuBLAS performance with FP16
- Handle mixed-precision computation (FP16 input, FP32 accumulation)

## Tensor Core Capabilities by Architecture

| Architecture | SM | Supported Types | Tile Size | Peak (TFLOPS) |
|--------------|-----|-----------------|-----------|---------------|
| Volta | 70 | FP16 | 16x16x16 | 125 (V100) |
| Turing | 75 | FP16, INT8, INT4 | 16x16x16 | 130 (T4) |
| Ampere | 80 | FP16, BF16, TF32, INT8 | 16x8x16 | 312 (A100) |
| Hopper | 90 | FP8, FP16, BF16, TF32 | 16x8x16 | 989 (H100) |

## WMMA vs MMA

### WMMA (Warp Matrix Multiply-Accumulate)

- **API Level**: C++ template API
- **Availability**: SM70+ (Volta, Turing, Ampere, Hopper)
- **Pros**: Portable, easier to use
- **Cons**: Slightly lower performance than PTX
- **Use case**: General purpose, cross-architecture

### MMA (PTX Matrix Multiply-Accumulate)

- **API Level**: PTX assembly intrinsics
- **Availability**: Architecture-specific (best on SM80+)
- **Pros**: Maximum performance, finer control
- **Cons**: More complex, architecture-specific
- **Use case**: Squeezing last 5-10% performance

## Files

### wmma_gemm.cu

FP16 GEMM using WMMA API:
- Works on Volta, Turing, Ampere, Hopper
- 16x16x16 matrix fragments
- FP16 input/output, FP32 accumulation
- Target: 85-90% cuBLAS

### mma_gemm.cu

FP16 GEMM using Ampere MMA instructions:
- Optimized for SM80+ (Ampere, Hopper)
- 16x8x16 matrix tiles
- Better register efficiency
- Target: 90-95% cuBLAS

## WMMA API Basics

### Fragment Types

```cpp
using namespace nvcuda::wmma;

// Declare fragments
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> c_frag;
```

### Operations

```cpp
// Initialize accumulator to zero
fill_fragment(c_frag, 0.0f);

// Load from shared/global memory
load_matrix_sync(a_frag, smem_A, 16);
load_matrix_sync(b_frag, smem_B, 16);

// Matrix multiply-accumulate
mma_sync(c_frag, a_frag, b_frag, c_frag);

// Store result
store_matrix_sync(smem_C, c_frag, 16, mem_row_major);
```

## Requirements and Constraints

### Alignment

- **Memory addresses**: 16-byte aligned (128-bit)
- **Leading dimensions**: Multiple of 16 for FP16

### Layout

- **Matrix A**: Row-major or column-major
- **Matrix B**: Must be opposite of A for best performance
- **Matrix C**: Row-major or column-major (independent)

### Warp Synchronization

- All WMMA operations are warp-synchronous
- All 32 threads in warp must participate
- No `__syncwarp()` needed within fragment operations

## Performance Optimization

### 1. Maximize Tensor Core Utilization

```cpp
// Good: Large enough tiles to saturate Tensor Cores
constexpr int BM = 128, BN = 128, BK = 32;

// Bad: Too small, underutilizes hardware
constexpr int BM = 32, BN = 32, BK = 8;
```

### 2. Pipeline Global → Shared → Tensor Cores

```cpp
// Double buffering for overlap
__shared__ half smem_A[2][BM * BK];
__shared__ half smem_B[2][BK * BN];

int write_idx = 0, read_idx = 1;

for (int k_tile = 0; k_tile < K; k_tile += BK) {
    // Load next tile while computing current
    load_tile_async(smem_A[write_idx], ...);
    load_tile_async(smem_B[write_idx], ...);

    // Compute current tile
    wmma_compute(smem_A[read_idx], smem_B[read_idx]);

    swap(write_idx, read_idx);
}
```

### 3. Swizzle Shared Memory

Avoid bank conflicts when loading fragments:

```cpp
// Apply XOR swizzle
constexpr int kSwizzle = 3;
auto smem_layout = composition(
    base_layout,
    Swizzle<kSwizzle, 0, 3>{}
);
```

## Building and Running

```bash
mkdir build && cd build
cmake .. -DCUTLASS_DIR=$CUTLASS_DIR -DCMAKE_CUDA_ARCHITECTURES=80
make -j$(nproc)

# WMMA (works on SM70+)
./wmma_gemm

# MMA (requires SM80+)
./mma_gemm
```

## Expected Performance

### FP16 GEMM (M=N=K=4096)

| Implementation | A100 (TFLOPS) | % of cuBLAS |
|----------------|---------------|-------------|
| FP32 tiled | 12-15 | 60-75% |
| WMMA FP16 | 220-240 | 85-90% |
| MMA FP16 | 250-270 | 90-95% |
| cuBLAS FP16 | 280-300 | 100% |

## Common Issues

### Issue 1: "Illegal memory access"

**Cause**: Misaligned memory addresses

**Solution**: Ensure 16-byte alignment
```cpp
cudaMalloc(&ptr, size);  // Already 256-byte aligned
// Or explicitly:
assert(reinterpret_cast<uintptr_t>(ptr) % 16 == 0);
```

### Issue 2: Wrong results

**Cause**: Layout mismatch (row vs column major)

**Solution**: Match fragment layout to memory layout
```cpp
// If memory is row-major:
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;

// If memory is column-major:
fragment<matrix_a, 16, 16, 16, half, col_major> a_frag;
```

### Issue 3: Low performance

**Causes**:
- Insufficient occupancy (too much shared memory)
- Poor Tensor Core utilization (tiles too small)
- Memory bandwidth bottleneck

**Solutions**:
- Reduce shared memory usage
- Increase tile sizes (BM, BN, BK)
- Use async copies and double buffering

## Profiling

### NSight Compute Metrics

```bash
ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active \
    ./wmma_gemm
```

**Target**: >80% Tensor Core utilization

### Check Instruction Mix

```bash
ncu --metrics smsp__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active \
    ./wmma_gemm
```

### Memory Bandwidth

```bash
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
    ./wmma_gemm
```

## Advanced Topics

### Mixed Precision

```cpp
// FP16 inputs, FP32 accumulation, FP16 output
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> acc_frag;  // FP32 for precision

// After accumulation, convert to FP16 for output
```

### TF32 Mode (Ampere+)

```cpp
// Enable TF32 for FP32 inputs with Tensor Cores
// Happens automatically on Ampere+ for FP32 GEMM
// Maintains FP32 range with reduced precision (19 bits)
```

## Next Steps

After mastering Tensor Cores:

1. **Proceed to Example 05**: Explore CuteDSL Python interface
2. **Complete Exercise 02**: Implement FP16 GEMM from scratch
3. **Study CUTLASS**: Learn production-quality Tensor Core kernels

## References

- [CUDA Programming Guide - WMMA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
- [PTX ISA - MMA Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions)
- [Tensor Cores Programming](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)
- [CUTLASS GEMM with Tensor Cores](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md)
