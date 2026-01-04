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
 * Naive dot product using atomic operations
 * Simple but slow due to atomic contention
 */
__global__ void dotProductNaive(float *A, float *B, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float product = A[idx] * B[idx];
        atomicAdd(result, product);
    }
}

/**
 * Shared memory dot product with block-level reduction
 * Much faster than atomic approach
 */
__global__ void dotProductShared(float *A, float *B, float *partial, int n) {
    // Shared memory for reduction
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data and multiply
    sdata[tid] = (idx < n) ? A[idx] * B[idx] : 0.0f;
    __syncthreads();

    // Reduction in shared memory (sequential addressing)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

/**
 * Warp reduction helper (no synchronization needed)
 */
__device__ void warpReduce(volatile float *sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

/**
 * Optimized dot product with unrolled last warp
 */
__global__ void dotProductOptimized(float *A, float *B, float *partial, int n) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // First add during global load (process 2 elements per thread)
    float sum = 0.0f;
    if (idx < n) sum = A[idx] * B[idx];
    if (idx + blockDim.x < n) sum += A[idx + blockDim.x] * B[idx + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    // Reduction with sequential addressing
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Unroll last warp (no __syncthreads needed)
    if (tid < 32) warpReduce(sdata, tid);

    // Write result
    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
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
    printf("=== Dot Product Exercise - SOLUTION ===\n\n");

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

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    float ms;

    // Test 1: Naive (atomic)
    printf("Test 1: Naive (Atomic Operations)\n");
    printf("----------------------------------\n");

    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
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
    float naive_time = ms;

    printf("GPU result: %.6f\n", gpu_result);
    printf("Error: %.6f%%\n", error);
    printf("Time: %.3f ms\n", ms);
    printf("Status: %s\n\n", (error < 0.1f) ? "PASSED" : "FAILED");

    cudaFree(d_result);

    // Test 2: Shared memory
    printf("Test 2: Shared Memory Reduction\n");
    printf("--------------------------------\n");

    size_t partial_bytes = gridSize * sizeof(float);
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
    printf("Speedup vs naive: %.2fx\n", naive_time / ms);
    printf("Status: %s\n\n", (error < 0.1f) ? "PASSED" : "FAILED");

    float shared_time = ms;

    // Test 3: Optimized
    printf("Test 3: Optimized (Unrolled Warp)\n");
    printf("----------------------------------\n");

    // For optimized version, use half the blocks (2 elements per thread)
    int gridSize_opt = (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
    size_t partial_bytes_opt = gridSize_opt * sizeof(float);

    float *d_partial_opt;
    CHECK_CUDA(cudaMalloc(&d_partial_opt, partial_bytes_opt));
    free(h_partial);
    h_partial = (float *)malloc(partial_bytes_opt);

    CHECK_CUDA(cudaEventRecord(start));
    dotProductOptimized<<<gridSize_opt, BLOCK_SIZE>>>(d_A, d_B, d_partial_opt, n);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaMemcpy(h_partial, d_partial_opt, partial_bytes_opt, cudaMemcpyDeviceToHost));
    gpu_result = reducePartials(h_partial, gridSize_opt);

    error = fabs(gpu_result - cpu_result) / cpu_result * 100.0f;
    printf("GPU result: %.6f\n", gpu_result);
    printf("Error: %.6f%%\n", error);
    printf("Time: %.3f ms\n", ms);
    printf("Speedup vs shared: %.2fx\n", shared_time / ms);
    printf("Status: %s\n\n", (error < 0.1f) ? "PASSED" : "FAILED");

    // Summary
    printf("=== Summary ===\n");
    printf("Optimization Progression:\n");
    printf("1. Naive (atomic):     %.3f ms\n", naive_time);
    printf("2. Shared memory:      %.3f ms (%.2fx)\n", shared_time, naive_time / shared_time);
    printf("3. Unrolled warp:      %.3f ms (%.2fx)\n", ms, shared_time / ms);
    printf("\nKey Learnings:\n");
    printf("- Atomic operations cause severe contention\n");
    printf("- Shared memory reduction is much faster\n");
    printf("- Unrolling last warp eliminates synchronization\n");
    printf("- Processing multiple elements per thread improves efficiency\n");

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_partial);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_partial);
    cudaFree(d_partial_opt);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
