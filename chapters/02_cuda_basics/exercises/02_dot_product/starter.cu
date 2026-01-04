#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK_CUDA(call)                                                     \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    }

#define BLOCK_SIZE 256

/**
 * TODO: Implement naive dot product using atomic operations
 *
 * Each thread computes one A[i] * B[i] and atomically adds to result
 */
__global__ void dotProductNaive(float *A, float *B, float *result, int n) {
    // YOUR CODE HERE

}

/**
 * TODO: Implement shared memory dot product
 *
 * 1. Load A[i] * B[i] into shared memory
 * 2. Perform reduction in shared memory
 * 3. Write partial result for this block
 */
__global__ void dotProductShared(float *A, float *B, float *partial, int n) {
    // YOUR CODE HERE

}

/**
 * TODO: Implement optimized dot product with unrolled warp
 *
 * Same as shared memory version but unroll last warp
 */
__global__ void dotProductOptimized(float *A, float *B, float *partial, int n) {
    // YOUR CODE HERE

}

// CPU reference implementation
float dotProductCPU(float *A, float *B, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += A[i] * B[i];
    }
    return sum;
}

// Reduce partial results on host
float reducePartials(float *partials, int count) {
    float sum = 0.0f;
    for (int i = 0; i < count; i++) {
        sum += partials[i];
    }
    return sum;
}

int main(int argc, char **argv) {
    printf("=== Dot Product Exercise ===\n\n");

    // Vector size
    int n = 64 * 1024 * 1024;  // 64M elements
    size_t bytes = n * sizeof(float);

    printf("Vector size: %d elements (%.2f MB)\n", n, bytes / (1024.0 * 1024.0));
    printf("Block size: %d threads\n\n", BLOCK_SIZE);

    // Allocate and initialize host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);

    for (int i = 0; i < n; i++) {
        h_A[i] = (float)(rand() % 100) / 100.0f;
        h_B[i] = (float)(rand() % 100) / 100.0f;
    }

    // Compute CPU reference
    printf("Computing CPU reference...\n");
    float cpu_result = dotProductCPU(h_A, h_B, n);
    printf("CPU result: %.6f\n\n", cpu_result);

    // Allocate device memory
    float *d_A, *d_B;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Setup grid dimensions
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t partial_bytes = gridSize * sizeof(float);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    float ms;

    // Test 1: Naive (atomic)
    printf("Test 1: Naive (Atomic Operations)\n");
    printf("----------------------------------\n");

    float *d_result;
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(float)));
    CHECK_CUDA(cudaMemset(d_result, 0, sizeof(float)));

    CHECK_CUDA(cudaEventRecord(start));
    dotProductNaive<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_result, n);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    float gpu_result;
    CHECK_CUDA(cudaMemcpy(&gpu_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    float error = fabs(gpu_result - cpu_result) / cpu_result * 100.0f;
    printf("GPU result: %.6f\n", gpu_result);
    printf("Error: %.6f%%\n", error);
    printf("Time: %.3f ms\n", ms);
    printf("Status: %s\n\n", (error < 0.1f) ? "PASSED" : "FAILED");

    cudaFree(d_result);

    // Test 2: Shared memory
    printf("Test 2: Shared Memory Reduction\n");
    printf("--------------------------------\n");

    float *d_partial, *h_partial;
    CHECK_CUDA(cudaMalloc(&d_partial, partial_bytes));
    h_partial = (float *)malloc(partial_bytes);

    CHECK_CUDA(cudaEventRecord(start));
    dotProductShared<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_partial, n);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaMemcpy(h_partial, d_partial, partial_bytes, cudaMemcpyDeviceToHost));
    gpu_result = reducePartials(h_partial, gridSize);

    error = fabs(gpu_result - cpu_result) / cpu_result * 100.0f;
    printf("GPU result: %.6f\n", gpu_result);
    printf("Error: %.6f%%\n", error);
    printf("Time: %.3f ms\n", ms);
    printf("Status: %s\n\n", (error < 0.1f) ? "PASSED" : "FAILED");

    // Test 3: Optimized
    printf("Test 3: Optimized (Unrolled Warp)\n");
    printf("----------------------------------\n");

    CHECK_CUDA(cudaEventRecord(start));
    dotProductOptimized<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_partial, n);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaMemcpy(h_partial, d_partial, partial_bytes, cudaMemcpyDeviceToHost));
    gpu_result = reducePartials(h_partial, gridSize);

    error = fabs(gpu_result - cpu_result) / cpu_result * 100.0f;
    printf("GPU result: %.6f\n", gpu_result);
    printf("Error: %.6f%%\n", error);
    printf("Time: %.3f ms\n", ms);
    printf("Status: %s\n\n", (error < 0.1f) ? "PASSED" : "FAILED");

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_partial);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_partial);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
