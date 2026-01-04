/**
 * @file solution.cu
 * @brief Complete solution for SAXPY exercise
 *
 * SAXPY: Y = α·X + Y (Scalar A times X Plus Y)
 *
 * This is the reference solution. Try to solve it yourself first!
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <string.h>

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
 * @brief CPU implementation of SAXPY (reference)
 */
void saxpyCPU(float alpha, const float* x, float* y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = alpha * x[i] + y[i];
    }
}

/**
 * @brief GPU kernel for SAXPY
 */
__global__ void saxpyKernel(float alpha, const float* x, float* y, int n) {
    // Calculate global thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check and compute SAXPY
    if (i < n) {
        y[i] = alpha * x[i] + y[i];
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
    const float epsilon = 1e-4f; // Slightly larger due to more operations
    for (int i = 0; i < n; i++) {
        if (fabsf(cpu[i] - gpu[i]) > epsilon) {
            fprintf(stderr, "Mismatch at index %d: CPU=%f, GPU=%f, diff=%e\n",
                    i, cpu[i], gpu[i], fabsf(cpu[i] - gpu[i]));
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    printf("=== CUDA SAXPY - Solution (Y = α·X + Y) ===\n\n");

    // Parse command line arguments
    int n = 1000000; // Default: 1 million elements
    float alpha = 2.0f; // Default: alpha = 2.0

    if (argc > 1) {
        n = atoi(argv[1]);
    }
    if (argc > 2) {
        alpha = atof(argv[2]);
    }

    printf("Vector size: %d elements\n", n);
    printf("Alpha (α): %f\n\n", alpha);

    size_t bytes = n * sizeof(float);

    // ========================================================================
    // Step 1: Allocate host memory
    // ========================================================================
    printf("Allocating memory...\n");

    float* h_x = (float*)malloc(bytes);
    float* h_y = (float*)malloc(bytes);
    float* h_y_cpu = (float*)malloc(bytes); // For CPU reference result

    if (!h_x || !h_y || !h_y_cpu) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }

    // ========================================================================
    // Step 2: Initialize input data
    // ========================================================================
    printf("Initializing data...\n");

    srand(12345); // Fixed seed for reproducibility
    initVector(h_x, n);
    initVector(h_y, n);

    // Save a copy of original Y for CPU computation
    // This is important because Y will be modified by the GPU
    memcpy(h_y_cpu, h_y, bytes);

    // ========================================================================
    // Step 3: Allocate device memory
    // ========================================================================
    printf("Allocating device memory...\n");

    float *d_x, *d_y;

    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));

    // ========================================================================
    // Step 4: Transfer data to device
    // ========================================================================
    printf("Transferring data to device...\n");

    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    // ========================================================================
    // Step 5: Launch kernel
    // ========================================================================
    printf("Launching kernel...\n");

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    printf("  Grid: %d blocks\n", blocksPerGrid);
    printf("  Block: %d threads\n", threadsPerBlock);
    printf("  Computing: Y = %.2f * X + Y\n\n", alpha);

    saxpyKernel<<<blocksPerGrid, threadsPerBlock>>>(alpha, d_x, d_y, n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ========================================================================
    // Step 6: Copy results back to host
    // ========================================================================
    printf("Transferring results back...\n");

    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

    // ========================================================================
    // Step 7: Verify results
    // ========================================================================
    printf("Computing CPU reference...\n");

    // Compute reference result on CPU using original Y values
    saxpyCPU(alpha, h_x, h_y_cpu, n);

    printf("Verifying results...\n");

    bool correct = verifyResults(h_y_cpu, h_y, n);

    if (correct) {
        printf("\nSUCCESS: GPU results match CPU!\n");
    } else {
        printf("\nFAILURE: GPU results do not match CPU!\n");
    }

    // Print sample results
    printf("\nSample results (first 5 elements):\n");
    printf("  Formula: Y_new = %.2f * X + Y_old\n", alpha);
    // Re-initialize for display (we've already verified correctness)
    srand(12345);
    float* temp_x = (float*)malloc(5 * sizeof(float));
    float* temp_y = (float*)malloc(5 * sizeof(float));
    for (int i = 0; i < 5 && i < n; i++) {
        temp_x[i] = (float)rand() / RAND_MAX;
        temp_y[i] = (float)rand() / RAND_MAX;
        printf("  [%d] Y = %.2f * %.4f + %.4f = %.4f\n",
               i, alpha, temp_x[i], temp_y[i],
               alpha * temp_x[i] + temp_y[i]);
    }
    free(temp_x);
    free(temp_y);

    // ========================================================================
    // Step 8: Cleanup
    // ========================================================================
    printf("\nCleaning up...\n");

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    free(h_x);
    free(h_y);
    free(h_y_cpu);

    printf("Done!\n");

    return correct ? 0 : 1;
}

/**
 * ============================================================================
 * KEY IMPLEMENTATION DETAILS
 * ============================================================================
 *
 * 1. Scalar Parameter:
 *    - Alpha is passed by value to the kernel
 *    - Each thread gets its own copy (in registers)
 *    - Very efficient - no memory access needed
 *
 * 2. In-Place Operation:
 *    - Y is both read and written: y[i] = alpha * x[i] + y[i]
 *    - Must read original Y value before writing
 *    - This is safe because each thread accesses different elements
 *
 * 3. Memory Pattern:
 *    - Read X: 1 global memory read
 *    - Read Y: 1 global memory read
 *    - Write Y: 1 global memory write
 *    - Total: 2 reads + 1 write = 3 memory ops per element
 *
 * 4. Arithmetic:
 *    - 1 multiply (alpha * x[i])
 *    - 1 add (result + y[i])
 *    - Total: 2 FLOPs per element
 *    - Arithmetic intensity: 2 FLOPs / 12 bytes = 0.17 FLOPs/byte
 *
 * 5. Verification Strategy:
 *    - Save copy of Y before GPU modifies it
 *    - Run CPU version on original Y
 *    - Compare CPU result with GPU result
 *
 * ============================================================================
 * PERFORMANCE CHARACTERISTICS
 * ============================================================================
 *
 * SAXPY is strongly memory-bound:
 * - Very low arithmetic intensity (0.17 ops/byte)
 * - Performance limited by memory bandwidth
 * - Optimization focus should be on memory access patterns
 *
 * For 1M elements (4MB per vector):
 * - Total memory: 3 * 4MB = 12MB
 * - Compute: 2M FLOPs
 * - On RTX 3080 (760 GB/s bandwidth):
 *   - Time: ~0.016 ms (memory-limited)
 *   - Achieved bandwidth: ~750 GB/s (near peak)
 *
 * ============================================================================
 * APPLICATIONS
 * ============================================================================
 *
 * SAXPY appears in many numerical algorithms:
 *
 * 1. Gradient Descent:
 *    weights = weights - learning_rate * gradients
 *    (alpha = -learning_rate, X = gradients, Y = weights)
 *
 * 2. Conjugate Gradient Method:
 *    p = r + beta * p
 *    (alpha = beta, X = p, Y = r)
 *
 * 3. Time Integration:
 *    position = position + dt * velocity
 *    (alpha = dt, X = velocity, Y = position)
 *
 * 4. Linear Combinations:
 *    result = a * vector1 + vector2
 *
 * ============================================================================
 */
