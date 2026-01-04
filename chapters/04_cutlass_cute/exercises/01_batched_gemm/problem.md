# Exercise 01: Batched GEMM

## Problem Statement

Implement a batched GEMM kernel using CuTe that computes multiple independent matrix multiplications in parallel:

```
for i in range(batch_size):
    C[i] = A[i] @ B[i]
```

## Learning Objectives

- Understand batched matrix operations
- Apply CuTe layouts to 3D tensors
- Optimize for multiple small/medium matrices
- Compare performance with cuBLAS batched GEMM

## Specifications

### Input

- **A**: Array of matrices, shape `(batch_size, M, K)`, dtype `float32`
- **B**: Array of matrices, shape `(batch_size, K, N)`, dtype `float32`
- **C**: Array of matrices, shape `(batch_size, M, N)`, dtype `float32` (output)

### Requirements

1. **Correctness**: Results must match cuBLAS within 1e-3 tolerance
2. **Performance**: Achieve >70% of cuBLAS batched GEMM
3. **Memory**: Use shared memory efficiently
4. **Scalability**: Support batch sizes from 1 to 1024+

## Batched GEMM Strategies

### Strategy 1: One Block per Matrix (Small Matrices)

- Best for: M, N, K ≤ 64
- Each thread block handles one entire matrix multiplication
- Excellent occupancy, minimal overhead

### Strategy 2: Multiple Blocks per Matrix (Large Matrices)

- Best for: M, N, K > 256
- Treat batch dimension as part of grid
- Each matrix uses multiple blocks (standard tiling)

### Strategy 3: Hybrid

- Combine strategies based on matrix size
- Use stream-based parallelism for very large batches

## Implementation Steps

1. **Create 3D Tensor Layout**
   ```cpp
   auto layout = make_layout(
       make_shape(batch_size, M, K),
       make_stride(M * K, K, 1)
   );
   ```

2. **Partition by Batch**
   ```cpp
   int batch_idx = blockIdx.z;  // or compute from linear blockIdx
   auto A_batch = A_global(batch_idx, _, _);  // Slice for this batch
   ```

3. **Apply Standard GEMM**
   - Use your tiled GEMM kernel from Example 03
   - Each batch element processed independently

4. **Optimize Grid Configuration**
   ```cpp
   dim3 grid(
       (M + BM - 1) / BM,
       (N + BN - 1) / BN,
       batch_size
   );
   ```

## Test Cases

Test your implementation with:

1. **Small matrices**: batch=100, M=N=K=64
2. **Medium matrices**: batch=50, M=N=K=256
3. **Large matrices**: batch=10, M=N=K=1024
4. **Mixed**: batch=32, M=128, N=256, K=512

## Performance Targets

| Matrix Size | Batch | Target vs cuBLAS |
|-------------|-------|------------------|
| 64x64x64 | 100 | 70%+ |
| 256x256x256 | 50 | 75%+ |
| 1024x1024x1024 | 10 | 80%+ |

## cuBLAS Reference

```cpp
// cuBLAS batched GEMM
cublasGemmStridedBatchedEx(
    handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    N, M, K,
    &alpha,
    B, CUDA_R_32F, N, N * K,  // stride
    A, CUDA_R_32F, K, M * K,  // stride
    &beta,
    C, CUDA_R_32F, N, M * N,  // stride
    batch_size,
    CUBLAS_COMPUTE_32F,
    CUBLAS_GEMM_DEFAULT
);
```

## Hints

1. **Memory layout**: Ensure contiguous access within each batch
2. **Stream parallelism**: For large batches, consider multiple streams
3. **Persistent kernels**: Reuse thread blocks across batches
4. **Profile**: Use NSight Compute to check occupancy per batch

## Bonus Challenges

1. **Variable sizes**: Support different (M, N, K) per batch element
2. **FP16**: Implement with half precision and Tensor Cores
3. **Grouped GEMM**: Different A, B, C shapes per group
4. **Fused operations**: Add element-wise operations (bias, ReLU)

## Evaluation Criteria

- **Correctness** (40%): Passes all test cases
- **Performance** (40%): Meets targets across sizes
- **Code quality** (20%): Clean, well-documented code

## Resources

- [cuBLAS Batched GEMM](https://docs.nvidia.com/cuda/cublas/index.html#cublas-GemmStridedBatchedEx)
- [Batched Operations Best Practices](https://developer.nvidia.com/blog/optimizing-batched-linear-algebra/)
- [CUTLASS Batched GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/13_two_tensor_op_fusion)
