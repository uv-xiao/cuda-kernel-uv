/*
 * CUDA C++ GEMM Implementation
 *
 * Implement: C = alpha * A @ B + beta * C
 *
 * TODO:
 * 1. Implement shared memory tiling
 * 2. Add register blocking
 * 3. Optimize memory access patterns
 * 4. Handle arbitrary matrix sizes
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>

#define TILE_SIZE 32  // TODO: Tune this
#define BLOCK_SIZE 16 // TODO: Tune this

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


/*
 * Naive GEMM kernel (baseline)
 */
__global__ void gemm_naive_kernel(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    float alpha, float beta
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}


/*
 * Tiled GEMM kernel with shared memory
 * TODO: Implement this kernel
 */
__global__ void gemm_tiled_kernel(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    float alpha, float beta
) {
    // TODO: Implement shared memory tiling
    // 1. Declare shared memory for tiles of A and B
    // 2. Load tiles into shared memory
    // 3. Compute partial products
    // 4. Accumulate results
    // 5. Write final result to C

    // Hints:
    // - Use __shared__ memory
    // - Synchronize threads after loading tiles
    // - Handle boundary conditions

    // Placeholder - replace with your implementation
    gemm_naive_kernel<<<1, 1>>>(A, B, C, M, N, K, alpha, beta);
}


/*
 * Optimized GEMM kernel with register blocking
 * TODO: Implement this kernel for extra performance
 */
__global__ void gemm_optimized_kernel(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    float alpha, float beta
) {
    // TODO: Implement optimizations
    // 1. Register blocking (e.g., 8x8 per thread)
    // 2. Vector loads
    // 3. Prefetching
    // 4. Avoid bank conflicts

    // Placeholder
    gemm_tiled_kernel<<<1, 1>>>(A, B, C, M, N, K, alpha, beta);
}


/*
 * Host function to launch GEMM kernel
 */
extern "C" void gemm_cuda(
    const float* h_A,
    const float* h_B,
    float* h_C,
    int M, int N, int K,
    float alpha, float beta
) {
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // TODO: Choose which kernel to launch
    // gemm_naive_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    gemm_tiled_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    // gemm_optimized_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}


/*
 * Simple test
 */
int main() {
    const int M = 256, N = 256, K = 256;
    float alpha = 1.0f, beta = 0.0f;

    // Allocate and initialize host matrices
    float *A = new float[M * K];
    float *B = new float[K * N];
    float *C = new float[M * N];

    for (int i = 0; i < M * K; i++) A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) B[i] = 1.0f;
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;

    // Run GEMM
    gemm_cuda(A, B, C, M, N, K, alpha, beta);

    // Verify (result should be K for all elements)
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        if (fabs(C[i] - K) > 1e-3) {
            printf("Error at %d: expected %f, got %f\n", i, (float)K, C[i]);
            correct = false;
            break;
        }
    }

    if (correct) {
        printf("CUDA GEMM test PASSED\n");
    } else {
        printf("CUDA GEMM test FAILED\n");
    }

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
