/**
 * @file solution.cu
 * @brief Complete solution for Vector Subtraction exercise
 *
 * This is the reference solution. Try to solve it yourself first!
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * @brief CPU implementation of vector subtraction (reference)
 */
void vectorSubtractCPU(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] - b[i];
    }
}

/**
 * @brief GPU kernel for vector subtraction
 */
__global__ void vectorSubtractKernel(const float* a, const float* b,
                                     float* c, int n) {
    // Calculate global thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check and compute subtraction
    if (i < n) {
        c[i] = a[i] - b[i];
    }
}

/**
 * @brief Initialize vector with random values
 */
void initVector(float* vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

/**
 * @brief Verify results match within tolerance
 */
bool verifyResults(const float* cpu, const float* gpu, int n) {
    const float epsilon = 1e-5f;
    for (int i = 0; i < n; i++) {
        if (fabsf(cpu[i] - gpu[i]) > epsilon) {
            fprintf(stderr, "Mismatch at index %d: CPU=%f, GPU=%f\n",
                    i, cpu[i], gpu[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    printf("=== CUDA Vector Subtraction - Solution ===\n\n");

    // Vector size
    int n = 1000000; // Default: 1 million elements
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    printf("Vector size: %d elements\n\n", n);

    size_t bytes = n * sizeof(float);

    // ========================================================================
    // Step 1: Allocate host memory
    // ========================================================================
    printf("Allocating memory...\n");

    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c_cpu = (float*)malloc(bytes);
    float* h_c_gpu = (float*)malloc(bytes);

    if (!h_a || !h_b || !h_c_cpu || !h_c_gpu) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }

    // ========================================================================
    // Step 2: Initialize input data
    // ========================================================================
    printf("Initializing data...\n");

    srand(12345); // Fixed seed for reproducibility
    initVector(h_a, n);
    initVector(h_b, n);

    // ========================================================================
    // Step 3: Allocate device memory
    // ========================================================================
    printf("Allocating device memory...\n");

    float *d_a, *d_b, *d_c;

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // ========================================================================
    // Step 4: Transfer data to device
    // ========================================================================
    printf("Transferring data to device...\n");

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // ========================================================================
    // Step 5: Launch kernel
    // ========================================================================
    printf("Launching kernel...\n");

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    printf("  Grid: %d blocks\n", blocksPerGrid);
    printf("  Block: %d threads\n", threadsPerBlock);

    vectorSubtractKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ========================================================================
    // Step 6: Copy results back to host
    // ========================================================================
    printf("Transferring results back...\n");

    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost));

    // ========================================================================
    // Step 7: Verify results
    // ========================================================================
    printf("Computing CPU reference...\n");

    vectorSubtractCPU(h_a, h_b, h_c_cpu, n);

    printf("Verifying results...\n");

    bool correct = verifyResults(h_c_cpu, h_c_gpu, n);

    if (correct) {
        printf("\nSUCCESS: GPU results match CPU!\n");
    } else {
        printf("\nFAILURE: GPU results do not match CPU!\n");
    }

    // Print sample results
    printf("\nSample results (first 5 elements):\n");
    for (int i = 0; i < 5 && i < n; i++) {
        printf("  A[%d] - B[%d] = C[%d]: %.4f - %.4f = %.4f (CPU: %.4f)\n",
               i, i, i, h_a[i], h_b[i], h_c_gpu[i], h_c_cpu[i]);
    }

    // ========================================================================
    // Step 8: Cleanup
    // ========================================================================
    printf("\nCleaning up...\n");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);

    printf("Done!\n");

    return correct ? 0 : 1;
}

/**
 * ============================================================================
 * KEY DIFFERENCES FROM VECTOR ADDITION
 * ============================================================================
 *
 * The only change from vector addition is the operation in the kernel:
 *   Addition:    c[i] = a[i] + b[i];
 *   Subtraction: c[i] = a[i] - b[i];
 *
 * Everything else (memory management, data transfer, verification) is identical.
 * This demonstrates the modularity of CUDA programming - the same structure
 * applies to many different operations.
 *
 * ============================================================================
 * LEARNING POINTS
 * ============================================================================
 *
 * 1. Simple operations only require changing the kernel computation
 * 2. The workflow (allocate, transfer, execute, retrieve) stays the same
 * 3. Error checking and verification patterns are reusable
 * 4. Block and grid configuration doesn't change for same-size problems
 *
 * ============================================================================
 */
