/**
 * @file starter.cu
 * @brief Starter code for SAXPY exercise
 *
 * SAXPY: Y = α·X + Y (Scalar A times X Plus Y)
 *
 * TODO: Complete the implementation by filling in the sections marked with TODO
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
 *
 * TODO: Implement this function
 * Compute y[i] = alpha * x[i] + y[i] for all i
 *
 * Note: This modifies y in-place!
 */
void saxpyCPU(float alpha, const float* x, float* y, int n) {
    // TODO: Implement CPU version for verification
}

/**
 * @brief GPU kernel for SAXPY
 *
 * TODO: Implement this kernel
 * Each thread should compute: y[i] = alpha * x[i] + y[i]
 *
 * Parameters:
 * - alpha: Scalar multiplier
 * - x: Input vector (read-only)
 * - y: Input/output vector (modified in-place)
 * - n: Vector size
 */
__global__ void saxpyKernel(float alpha, const float* x, float* y, int n) {
    // TODO: Calculate global thread index
    // int i = ???

    // TODO: Bounds check and compute SAXPY
    // if (i < n) {
    //     y[i] = ???
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
    printf("=== CUDA SAXPY Exercise (Y = α·X + Y) ===\n\n");

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

    // TODO: Allocate host memory for x, y, and y_cpu (for verification)
    float* h_x = /* TODO */;
    float* h_y = /* TODO */;
    float* h_y_cpu = /* TODO */; // For CPU reference result

    // TODO: Check if allocations succeeded

    // ========================================================================
    // Step 2: Initialize input data
    // ========================================================================
    printf("Initializing data...\n");

    srand(12345); // Fixed seed for reproducibility
    initVector(h_x, n);
    initVector(h_y, n);

    // TODO: Save a copy of original Y for CPU computation
    // We need this because Y will be modified by the GPU
    // memcpy(...);

    // ========================================================================
    // Step 3: Allocate device memory
    // ========================================================================
    printf("Allocating device memory...\n");

    // TODO: Declare device pointers
    float *d_x, *d_y;

    // TODO: Allocate device memory
    // CUDA_CHECK(cudaMalloc(...));

    // ========================================================================
    // Step 4: Transfer data to device
    // ========================================================================
    printf("Transferring data to device...\n");

    // TODO: Copy h_x and h_y to d_x and d_y
    // CUDA_CHECK(cudaMemcpy(..., cudaMemcpyHostToDevice));

    // ========================================================================
    // Step 5: Launch kernel
    // ========================================================================
    printf("Launching kernel...\n");

    // TODO: Configure kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = /* TODO: Calculate */;

    printf("  Grid: %d blocks\n", blocksPerGrid);
    printf("  Block: %d threads\n", threadsPerBlock);
    printf("  Computing: Y = %.2f * X + Y\n\n", alpha);

    // TODO: Launch the kernel
    // saxpyKernel<<<blocksPerGrid, threadsPerBlock>>>(...);

    // TODO: Check for kernel launch errors
    // CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());

    // ========================================================================
    // Step 6: Copy results back to host
    // ========================================================================
    printf("Transferring results back...\n");

    // TODO: Copy d_y back to h_y (h_y now contains the result)
    // CUDA_CHECK(cudaMemcpy(..., cudaMemcpyDeviceToHost));

    // ========================================================================
    // Step 7: Verify results
    // ========================================================================
    printf("Computing CPU reference...\n");

    // TODO: Compute reference result on CPU using original Y values
    // saxpyCPU(alpha, h_x, h_y_cpu, n);

    printf("Verifying results...\n");

    // TODO: Verify GPU results match CPU results
    // bool correct = verifyResults(h_y_cpu, h_y, n);

    if (correct) {
        printf("\nSUCCESS: GPU results match CPU!\n");
    } else {
        printf("\nFAILURE: GPU results do not match CPU!\n");
    }

    // Print sample results
    printf("\nSample results (first 5 elements):\n");
    printf("  i     X[i]      Y_orig    Y_result  Y_cpu\n");
    // Note: We can't show Y_orig here since h_y was modified
    // In a real implementation, you might save original values
    for (int i = 0; i < 5 && i < n; i++) {
        printf("  %d   %.4f    (orig)    %.4f    %.4f\n",
               i, h_x[i], h_y[i], h_y_cpu[i]);
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
 * 2. Implement each section
 * 3. Key points:
 *    - Kernel computes: y[i] = alpha * x[i] + y[i]
 *    - Y is modified in-place (input and output)
 *    - Save original Y before GPU modifies it for CPU verification
 *    - Only need to copy Y back (X doesn't change)
 *
 * 4. Compile: nvcc -o saxpy starter.cu
 * 5. Run: ./saxpy [size] [alpha]
 *    Examples:
 *      ./saxpy               # 1M elements, alpha=2.0
 *      ./saxpy 100000 3.5    # 100K elements, alpha=3.5
 *      ./saxpy 50000 -1.0    # 50K elements, alpha=-1.0
 *
 * 6. Test edge cases:
 *    ./saxpy 1000 0.0       # alpha=0, Y should be unchanged
 *    ./saxpy 1000 1.0       # alpha=1, Y = X + Y
 *
 * ============================================================================
 * KEY FORMULA
 * ============================================================================
 *
 * SAXPY: Y = α·X + Y
 *
 * For each element i:
 *   Y[i] = alpha * X[i] + Y[i]
 *         └─────┬──────┘   └─┬─┘
 *           multiply     original Y
 *
 * Example with alpha=2.0:
 *   X = [1, 2, 3]
 *   Y = [4, 5, 6]
 *   Result: Y = [2*1+4, 2*2+5, 2*3+6] = [6, 9, 12]
 *
 * ============================================================================
 */
