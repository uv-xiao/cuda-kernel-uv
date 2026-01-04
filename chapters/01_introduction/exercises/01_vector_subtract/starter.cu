/**
 * @file starter.cu
 * @brief Starter code for Vector Subtraction exercise
 *
 * TODO: Complete the implementation by filling in the sections marked with TODO
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
 *
 * TODO: Implement this function
 * Compute c[i] = a[i] - b[i] for all i
 */
void vectorSubtractCPU(const float* a, const float* b, float* c, int n) {
    // TODO: Implement CPU version for verification
}

/**
 * @brief GPU kernel for vector subtraction
 *
 * TODO: Implement this kernel
 * Each thread should compute one element: c[i] = a[i] - b[i]
 * Don't forget bounds checking!
 */
__global__ void vectorSubtractKernel(const float* a, const float* b,
                                     float* c, int n) {
    // TODO: Calculate global thread index
    // int i = ???

    // TODO: Bounds check and compute subtraction
    // if (i < n) {
    //     c[i] = ???
    // }
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
    printf("=== CUDA Vector Subtraction Exercise ===\n\n");

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

    // TODO: Allocate host memory for a, b, c_cpu, c_gpu
    float* h_a = /* TODO */;
    float* h_b = /* TODO */;
    float* h_c_cpu = /* TODO */;
    float* h_c_gpu = /* TODO */;

    // TODO: Check if allocations succeeded
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

    // TODO: Declare device pointers
    float *d_a, *d_b, *d_c;

    // TODO: Allocate device memory using cudaMalloc
    // CUDA_CHECK(cudaMalloc(...));

    // ========================================================================
    // Step 4: Transfer data to device
    // ========================================================================
    printf("Transferring data to device...\n");

    // TODO: Copy h_a and h_b to d_a and d_b
    // CUDA_CHECK(cudaMemcpy(..., cudaMemcpyHostToDevice));

    // ========================================================================
    // Step 5: Launch kernel
    // ========================================================================
    printf("Launching kernel...\n");

    // TODO: Configure kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = /* TODO: Calculate based on n and threadsPerBlock */;

    printf("  Grid: %d blocks\n", blocksPerGrid);
    printf("  Block: %d threads\n", threadsPerBlock);

    // TODO: Launch the kernel
    // vectorSubtractKernel<<<blocksPerGrid, threadsPerBlock>>>(...);

    // TODO: Check for kernel launch errors
    // CUDA_CHECK(cudaGetLastError());

    // ========================================================================
    // Step 6: Copy results back to host
    // ========================================================================
    printf("Transferring results back...\n");

    // TODO: Copy d_c to h_c_gpu
    // CUDA_CHECK(cudaMemcpy(..., cudaMemcpyDeviceToHost));

    // ========================================================================
    // Step 7: Verify results
    // ========================================================================
    printf("Computing CPU reference...\n");

    // TODO: Compute reference result on CPU
    // vectorSubtractCPU(...);

    printf("Verifying results...\n");

    // TODO: Verify GPU results match CPU results
    // bool correct = verifyResults(...);

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

    // TODO: Free device memory
    // cudaFree(...);

    // TODO: Free host memory
    // free(...);

    printf("Done!\n");

    return correct ? 0 : 1;
}

/**
 * ============================================================================
 * INSTRUCTIONS
 * ============================================================================
 *
 * 1. Search for "TODO" comments in this file
 * 2. Implement each section following the patterns from vector_add example
 * 3. Compile: nvcc -o vector_subtract starter.cu
 * 4. Run: ./vector_subtract
 * 5. Verify output shows "SUCCESS"
 * 6. Test with different sizes: ./vector_subtract 100000
 *
 * Key points to remember:
 * - Global thread ID: blockIdx.x * blockDim.x + threadIdx.x
 * - Bounds checking: if (i < n)
 * - Grid size: (n + blockSize - 1) / blockSize
 * - Operation: c[i] = a[i] - b[i]
 *
 * ============================================================================
 */
