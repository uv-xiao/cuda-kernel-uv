/**
 * @file vector_add.cu
 * @brief Complete example of vector addition on GPU
 *
 * This example demonstrates a complete CUDA workflow:
 * - Host (CPU) and Device (GPU) memory allocation
 * - Data transfer between host and device
 * - Kernel implementation for data-parallel operation
 * - Error checking and validation
 * - Performance measurement and comparison
 *
 * Vector addition: C[i] = A[i] + B[i] for all i
 * This is a fundamental data-parallel operation where each element
 * is computed independently - perfect for GPU acceleration.
 *
 * Learning Objectives:
 * 1. Manage device memory with cudaMalloc/cudaFree
 * 2. Transfer data with cudaMemcpy
 * 3. Implement bounds checking in kernels
 * 4. Validate GPU results against CPU
 * 5. Measure and compare performance
 *
 * CUDA Programming Guide Reference:
 * - Section 3.2: CUDA Runtime API
 * - Section 3.2.1: Initialization
 * - Section 3.2.2: Device Memory
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

/**
 * @brief CUDA error checking macro
 *
 * This macro wraps CUDA calls and checks for errors.
 * Usage: CUDA_CHECK(cudaMalloc(&ptr, size));
 */
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
 * @brief CPU implementation of vector addition
 *
 * This serves as the reference implementation for validation.
 * Sequential execution on a single CPU core.
 *
 * Time complexity: O(n)
 * Space complexity: O(1) auxiliary space
 *
 * @param a First input vector
 * @param b Second input vector
 * @param c Output vector (a + b)
 * @param n Number of elements
 */
void vectorAddCPU(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

/**
 * @brief GPU kernel for vector addition
 *
 * Each thread computes one element of the output vector.
 * This is the CUDA way of expressing data parallelism.
 *
 * Thread Organization:
 * - 1D grid of 1D blocks (most common for 1D data)
 * - Each thread calculates its global index
 * - Bounds checking ensures we don't access invalid memory
 *
 * Key Pattern:
 *   int i = blockIdx.x * blockDim.x + threadIdx.x;
 *   if (i < n) { ... }
 *
 * Why bounds checking?
 * - Grid size is often rounded up to block size multiple
 * - Extra threads must be prevented from accessing invalid memory
 *
 * @param a Device pointer to first input vector
 * @param b Device pointer to second input vector
 * @param c Device pointer to output vector
 * @param n Number of elements
 */
__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
    // Calculate global thread index
    // This is the standard pattern for 1D data-parallel operations
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check: ensure thread index is within array bounds
    // Critical for correctness when n is not a multiple of block size
    if (i < n) {
        c[i] = a[i] + b[i];
    }

    // Note: No return value - kernel modifies device memory in place
    // Results will be copied back to host after kernel completes
}

/**
 * @brief Initialize vector with sequential values
 * @param vec Vector to initialize
 * @param n Number of elements
 */
void initVector(float* vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)i;
    }
}

/**
 * @brief Initialize vector with random values
 * @param vec Vector to initialize
 * @param n Number of elements
 */
void initVectorRandom(float* vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

/**
 * @brief Verify GPU results against CPU reference
 *
 * @param cpu CPU-computed results
 * @param gpu GPU-computed results
 * @param n Number of elements
 * @param epsilon Tolerance for floating-point comparison
 * @return true if results match within epsilon
 */
bool verifyResults(const float* cpu, const float* gpu, int n, float epsilon = 1e-5) {
    for (int i = 0; i < n; i++) {
        if (fabs(cpu[i] - gpu[i]) > epsilon) {
            fprintf(stderr, "Mismatch at index %d: CPU=%f, GPU=%f\n",
                    i, cpu[i], gpu[i]);
            return false;
        }
    }
    return true;
}

/**
 * @brief Get elapsed time in milliseconds
 * @param start Start time
 * @param end End time
 * @return Elapsed milliseconds
 */
double getElapsedTime(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 +
           (end.tv_nsec - start.tv_nsec) / 1000000.0;
}

int main(int argc, char** argv) {
    printf("=== CUDA Vector Addition Example ===\n\n");

    // ========================================================================
    // Step 1: Setup and Configuration
    // ========================================================================

    // Vector size (can be passed as command-line argument)
    int n = 1000000; // 1 million elements (4 MB per vector)
    if (argc > 1) {
        n = atoi(argv[1]);
    }

    printf("Vector size: %d elements (%.2f MB per vector)\n",
           n, n * sizeof(float) / (1024.0 * 1024.0));

    size_t bytes = n * sizeof(float);

    // ========================================================================
    // Step 2: Host Memory Allocation
    // ========================================================================

    printf("\n--- Memory Allocation ---\n");

    // Allocate host (CPU) memory
    float* h_a = (float*)malloc(bytes);      // First input vector
    float* h_b = (float*)malloc(bytes);      // Second input vector
    float* h_c_gpu = (float*)malloc(bytes);  // GPU results
    float* h_c_cpu = (float*)malloc(bytes);  // CPU results (for verification)

    if (!h_a || !h_b || !h_c_gpu || !h_c_cpu) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }

    printf("Allocated host memory: %.2f MB\n", 4 * bytes / (1024.0 * 1024.0));

    // ========================================================================
    // Step 3: Initialize Input Data
    // ========================================================================

    printf("\n--- Initializing Data ---\n");

    srand(time(NULL));
    initVectorRandom(h_a, n);
    initVectorRandom(h_b, n);

    printf("Initialized input vectors with random values\n");

    // ========================================================================
    // Step 4: Device Memory Allocation
    // ========================================================================

    printf("\n--- Device Memory Allocation ---\n");

    float *d_a, *d_b, *d_c;

    // Allocate device (GPU) memory
    // cudaMalloc returns a pointer to device memory (cannot be dereferenced on host!)
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    printf("Allocated device memory: %.2f MB\n", 3 * bytes / (1024.0 * 1024.0));

    // Query device memory info
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    printf("GPU memory: %.2f MB used, %.2f MB free, %.2f MB total\n",
           (total_mem - free_mem) / (1024.0 * 1024.0),
           free_mem / (1024.0 * 1024.0),
           total_mem / (1024.0 * 1024.0));

    // ========================================================================
    // Step 5: Host to Device Data Transfer
    // ========================================================================

    printf("\n--- Host to Device Transfer ---\n");

    struct timespec h2d_start, h2d_end;
    clock_gettime(CLOCK_MONOTONIC, &h2d_start);

    // Copy input vectors from host to device
    // cudaMemcpy is synchronous - blocks until transfer completes
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    clock_gettime(CLOCK_MONOTONIC, &h2d_end);
    double h2d_time = getElapsedTime(h2d_start, h2d_end);

    printf("Transferred %.2f MB to device in %.2f ms (%.2f GB/s)\n",
           2 * bytes / (1024.0 * 1024.0),
           h2d_time,
           (2 * bytes / (1024.0 * 1024.0 * 1024.0)) / (h2d_time / 1000.0));

    // ========================================================================
    // Step 6: Configure and Launch Kernel
    // ========================================================================

    printf("\n--- Kernel Launch Configuration ---\n");

    // Choose block size (threads per block)
    // Best practice: Use multiple of warp size (32)
    // Common choices: 128, 256, 512
    int threadsPerBlock = 256;

    // Calculate grid size (number of blocks)
    // Formula ensures enough threads to cover all elements
    // (n + threadsPerBlock - 1) / threadsPerBlock is ceiling division
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    printf("Grid size: %d blocks\n", blocksPerGrid);
    printf("Block size: %d threads\n", threadsPerBlock);
    printf("Total threads: %d (covers %d elements)\n",
           blocksPerGrid * threadsPerBlock, n);
    printf("Extra threads: %d (will be bounds-checked)\n",
           blocksPerGrid * threadsPerBlock - n);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    printf("\n--- Launching Kernel ---\n");

    // Record start event
    CUDA_CHECK(cudaEventRecord(start));

    // Launch kernel
    // Kernel execution is asynchronous - control returns immediately
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Record stop event
    CUDA_CHECK(cudaEventRecord(stop));

    // Wait for kernel to complete
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float kernel_time;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time, start, stop));

    printf("Kernel executed in %.2f ms\n", kernel_time);

    // Calculate effective bandwidth
    // 3 memory operations per element: read A, read B, write C
    double bandwidth = (3.0 * bytes / (1024.0 * 1024.0 * 1024.0)) / (kernel_time / 1000.0);
    printf("Effective bandwidth: %.2f GB/s\n", bandwidth);

    // ========================================================================
    // Step 7: Device to Host Data Transfer
    // ========================================================================

    printf("\n--- Device to Host Transfer ---\n");

    struct timespec d2h_start, d2h_end;
    clock_gettime(CLOCK_MONOTONIC, &d2h_start);

    // Copy result from device to host
    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost));

    clock_gettime(CLOCK_MONOTONIC, &d2h_end);
    double d2h_time = getElapsedTime(d2h_start, d2h_end);

    printf("Transferred %.2f MB from device in %.2f ms (%.2f GB/s)\n",
           bytes / (1024.0 * 1024.0),
           d2h_time,
           (bytes / (1024.0 * 1024.0 * 1024.0)) / (d2h_time / 1000.0));

    // ========================================================================
    // Step 8: CPU Execution for Comparison
    // ========================================================================

    printf("\n--- CPU Execution ---\n");

    struct timespec cpu_start, cpu_end;
    clock_gettime(CLOCK_MONOTONIC, &cpu_start);

    vectorAddCPU(h_a, h_b, h_c_cpu, n);

    clock_gettime(CLOCK_MONOTONIC, &cpu_end);
    double cpu_time = getElapsedTime(cpu_start, cpu_end);

    printf("CPU execution time: %.2f ms\n", cpu_time);

    // ========================================================================
    // Step 9: Verify Results
    // ========================================================================

    printf("\n--- Verification ---\n");

    bool correct = verifyResults(h_c_cpu, h_c_gpu, n);

    if (correct) {
        printf("SUCCESS: GPU results match CPU results!\n");
    } else {
        printf("FAILURE: GPU results do not match CPU results!\n");
    }

    // Print first few results as sanity check
    printf("\nFirst 10 results:\n");
    printf("  i   A[i]      B[i]      CPU       GPU\n");
    for (int i = 0; i < 10; i++) {
        printf("%3d  %8.3f  %8.3f  %8.3f  %8.3f\n",
               i, h_a[i], h_b[i], h_c_cpu[i], h_c_gpu[i]);
    }

    // ========================================================================
    // Step 10: Performance Analysis
    // ========================================================================

    printf("\n--- Performance Summary ---\n");

    double total_gpu_time = h2d_time + kernel_time + d2h_time;
    double speedup = cpu_time / kernel_time;
    double speedup_with_transfer = cpu_time / total_gpu_time;

    printf("CPU time:              %8.2f ms\n", cpu_time);
    printf("GPU kernel time:       %8.2f ms\n", kernel_time);
    printf("GPU transfer time:     %8.2f ms (H2D: %.2f, D2H: %.2f)\n",
           h2d_time + d2h_time, h2d_time, d2h_time);
    printf("Total GPU time:        %8.2f ms\n", total_gpu_time);
    printf("\n");
    printf("Speedup (kernel only): %.2fx\n", speedup);
    printf("Speedup (with transfer): %.2fx\n", speedup_with_transfer);

    if (speedup_with_transfer < 1.0) {
        printf("\nNote: GPU is slower when including transfer overhead.\n");
        printf("This is common for small data sizes or simple operations.\n");
        printf("Try larger vector sizes to see GPU benefits.\n");
    }

    // ========================================================================
    // Step 11: Cleanup
    // ========================================================================

    printf("\n--- Cleanup ---\n");

    // Free device memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    // Destroy events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c_gpu);
    free(h_c_cpu);

    printf("Released all resources\n");

    printf("\n=== Program Complete ===\n");

    return 0;
}

/**
 * ============================================================================
 * COMPILATION AND EXECUTION
 * ============================================================================
 *
 * Compile:
 *   nvcc -o vector_add vector_add.cu
 *
 * Run with default size (1M elements):
 *   ./vector_add
 *
 * Run with custom size:
 *   ./vector_add 10000000  # 10M elements
 *
 * ============================================================================
 * KEY TAKEAWAYS
 * ============================================================================
 *
 * 1. CUDA Workflow:
 *    - Allocate host and device memory
 *    - Initialize data on host
 *    - Copy data to device (H2D)
 *    - Execute kernel
 *    - Copy results back (D2H)
 *    - Verify and cleanup
 *
 * 2. Memory Management:
 *    - cudaMalloc/cudaFree for device memory
 *    - cudaMemcpy for transfers
 *    - Host and device pointers are incompatible
 *
 * 3. Kernel Design:
 *    - Calculate global thread ID
 *    - Always include bounds checking
 *    - One thread per element is typical for simple operations
 *
 * 4. Performance:
 *    - Kernel execution is fast
 *    - Memory transfer can dominate for small data
 *    - GPU shines with larger datasets
 *
 * 5. Error Handling:
 *    - Check all CUDA API calls
 *    - Use cudaGetLastError after kernel launches
 *    - Synchronize before checking results
 *
 * ============================================================================
 */
