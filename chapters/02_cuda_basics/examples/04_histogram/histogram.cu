#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHECK_CUDA(call)                                                     \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    }

#define NUM_BINS 256

/**
 * Version 1: Naive global memory atomic operations
 * Each thread directly atomics to global histogram
 */
__global__ void histogram_v1_global_atomic(unsigned char *data, int *histogram, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        atomicAdd(&histogram[data[idx]], 1);
    }
}

/**
 * Version 2: Shared memory per-block histogram
 * Reduce contention by using shared memory, then merge to global
 */
__global__ void histogram_v2_shared(unsigned char *data, int *histogram, int n) {
    __shared__ int s_hist[NUM_BINS];

    // Initialize shared memory histogram
    if (threadIdx.x < NUM_BINS) {
        s_hist[threadIdx.x] = 0;
    }
    __syncthreads();

    // Compute histogram in shared memory
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&s_hist[data[idx]], 1);
    }
    __syncthreads();

    // Merge to global histogram
    if (threadIdx.x < NUM_BINS) {
        atomicAdd(&histogram[threadIdx.x], s_hist[threadIdx.x]);
    }
}

/**
 * Version 3: Per-block with multiple elements per thread
 * Process multiple elements to increase arithmetic intensity
 */
__global__ void histogram_v3_multi_elements(unsigned char *data, int *histogram,
                                             int n, int elements_per_thread) {
    __shared__ int s_hist[NUM_BINS];

    // Initialize shared memory
    if (threadIdx.x < NUM_BINS) {
        s_hist[threadIdx.x] = 0;
    }
    __syncthreads();

    // Process multiple elements per thread
    int base_idx = blockIdx.x * blockDim.x * elements_per_thread + threadIdx.x;

    for (int i = 0; i < elements_per_thread; i++) {
        int idx = base_idx + i * blockDim.x;
        if (idx < n) {
            atomicAdd(&s_hist[data[idx]], 1);
        }
    }
    __syncthreads();

    // Merge to global
    if (threadIdx.x < NUM_BINS) {
        atomicAdd(&histogram[threadIdx.x], s_hist[threadIdx.x]);
    }
}

/**
 * Version 4: Privatized shared memory histograms
 * Multiple histograms per block to reduce contention
 */
__global__ void histogram_v4_privatized(unsigned char *data, int *histogram, int n) {
    // Use 4 privatized histograms to reduce contention
    __shared__ int s_hist[4][NUM_BINS];

    int tid = threadIdx.x;
    int private_id = tid % 4;  // Which private histogram to use

    // Initialize all private histograms
    for (int i = tid; i < 4 * NUM_BINS; i += blockDim.x) {
        s_hist[i / NUM_BINS][i % NUM_BINS] = 0;
    }
    __syncthreads();

    // Each thread uses its assigned private histogram
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&s_hist[private_id][data[idx]], 1);
    }
    __syncthreads();

    // Merge private histograms
    for (int i = tid; i < NUM_BINS; i += blockDim.x) {
        int count = 0;
        for (int j = 0; j < 4; j++) {
            count += s_hist[j][i];
        }
        atomicAdd(&histogram[i], count);
    }
}

/**
 * Host histogram for verification
 */
void histogramHost(unsigned char *data, int *histogram, int n) {
    for (int i = 0; i < NUM_BINS; i++) {
        histogram[i] = 0;
    }

    for (int i = 0; i < n; i++) {
        histogram[data[i]]++;
    }
}

/**
 * Verify histogram results
 */
bool verifyHistogram(int *h_ref, int *h_gpu, int num_bins) {
    for (int i = 0; i < num_bins; i++) {
        if (h_ref[i] != h_gpu[i]) {
            printf("Mismatch at bin %d: expected %d, got %d\n",
                   i, h_ref[i], h_gpu[i]);
            return false;
        }
    }
    return true;
}

/**
 * Print histogram as a simple ASCII chart
 */
void printHistogram(const char *name, int *histogram, int num_bins) {
    printf("\n%s:\n", name);

    // Find maximum for scaling
    int max_count = 0;
    for (int i = 0; i < num_bins; i++) {
        if (histogram[i] > max_count) max_count = histogram[i];
    }

    // Print with scaling
    int scale = (max_count + 49) / 50;  // Scale to ~50 chars width
    if (scale == 0) scale = 1;

    printf("Bin  Count        Distribution\n");
    printf("---  ------------ ------------------------------------------------\n");

    for (int i = 0; i < num_bins; i += 16) {  // Show every 16th bin
        int bar_length = histogram[i] / scale;
        printf("%3d  %12d ", i, histogram[i]);
        for (int j = 0; j < bar_length && j < 50; j++) {
            printf("#");
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    printf("=== CUDA Histogram with Atomic Operations ===\n\n");

    srand(time(NULL));

    // Data size
    int n = 64 * 1024 * 1024;  // 64M elements
    size_t bytes = n * sizeof(unsigned char);
    size_t hist_bytes = NUM_BINS * sizeof(int);

    printf("Data size: %d elements (%.2f MB)\n", n, bytes / (1024.0 * 1024.0));
    printf("Histogram bins: %d\n\n", NUM_BINS);

    // Allocate host memory
    unsigned char *h_data = (unsigned char *)malloc(bytes);
    int *h_histogram_ref = (int *)malloc(hist_bytes);
    int *h_histogram_gpu = (int *)malloc(hist_bytes);

    // Generate random data with non-uniform distribution
    printf("Generating data (normal distribution)...\n");
    for (int i = 0; i < n; i++) {
        // Approximate normal distribution
        int sum = 0;
        for (int j = 0; j < 12; j++) {
            sum += rand() % 256;
        }
        h_data[i] = (unsigned char)((sum / 12) % 256);
    }

    // Compute reference histogram
    printf("Computing reference histogram on CPU...\n");
    clock_t cpu_start = clock();
    histogramHost(h_data, h_histogram_ref, n);
    clock_t cpu_end = clock();
    float cpu_time = 1000.0f * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    printf("CPU time: %.2f ms\n\n", cpu_time);

    // Allocate device memory
    unsigned char *d_data;
    int *d_histogram;
    CHECK_CUDA(cudaMalloc(&d_data, bytes));
    CHECK_CUDA(cudaMalloc(&d_histogram, hist_bytes));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    // Setup execution parameters
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    float gpu_time;

    // ===== Version 1: Global Atomic =====
    printf("Version 1: Global Memory Atomics\n");
    printf("---------------------------------\n");

    CHECK_CUDA(cudaMemset(d_histogram, 0, hist_bytes));
    CHECK_CUDA(cudaEventRecord(start));
    histogram_v1_global_atomic<<<gridSize, blockSize>>>(d_data, d_histogram, n);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&gpu_time, start, stop));

    CHECK_CUDA(cudaMemcpy(h_histogram_gpu, d_histogram, hist_bytes,
                          cudaMemcpyDeviceToHost));

    bool v1_correct = verifyHistogram(h_histogram_ref, h_histogram_gpu, NUM_BINS);
    printf("Correctness: %s\n", v1_correct ? "PASSED" : "FAILED");
    printf("Time: %.3f ms\n", gpu_time);
    printf("Speedup vs CPU: %.2fx\n\n", cpu_time / gpu_time);

    float v1_time = gpu_time;

    // ===== Version 2: Shared Memory =====
    printf("Version 2: Shared Memory Per-Block\n");
    printf("-----------------------------------\n");

    CHECK_CUDA(cudaMemset(d_histogram, 0, hist_bytes));
    CHECK_CUDA(cudaEventRecord(start));
    histogram_v2_shared<<<gridSize, blockSize>>>(d_data, d_histogram, n);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&gpu_time, start, stop));

    CHECK_CUDA(cudaMemcpy(h_histogram_gpu, d_histogram, hist_bytes,
                          cudaMemcpyDeviceToHost));

    bool v2_correct = verifyHistogram(h_histogram_ref, h_histogram_gpu, NUM_BINS);
    printf("Correctness: %s\n", v2_correct ? "PASSED" : "FAILED");
    printf("Time: %.3f ms\n", gpu_time);
    printf("Speedup vs V1: %.2fx\n\n", v1_time / gpu_time);

    float v2_time = gpu_time;

    // ===== Version 3: Multiple Elements =====
    printf("Version 3: Multiple Elements Per Thread\n");
    printf("----------------------------------------\n");

    int elements_per_thread = 4;
    int gridSize_v3 = (n + blockSize * elements_per_thread - 1) /
                      (blockSize * elements_per_thread);

    CHECK_CUDA(cudaMemset(d_histogram, 0, hist_bytes));
    CHECK_CUDA(cudaEventRecord(start));
    histogram_v3_multi_elements<<<gridSize_v3, blockSize>>>(
        d_data, d_histogram, n, elements_per_thread);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&gpu_time, start, stop));

    CHECK_CUDA(cudaMemcpy(h_histogram_gpu, d_histogram, hist_bytes,
                          cudaMemcpyDeviceToHost));

    bool v3_correct = verifyHistogram(h_histogram_ref, h_histogram_gpu, NUM_BINS);
    printf("Correctness: %s\n", v3_correct ? "PASSED" : "FAILED");
    printf("Time: %.3f ms\n", gpu_time);
    printf("Speedup vs V2: %.2fx\n\n", v2_time / gpu_time);

    float v3_time = gpu_time;

    // ===== Version 4: Privatized Histograms =====
    printf("Version 4: Privatized Shared Memory\n");
    printf("------------------------------------\n");

    CHECK_CUDA(cudaMemset(d_histogram, 0, hist_bytes));
    CHECK_CUDA(cudaEventRecord(start));
    histogram_v4_privatized<<<gridSize, blockSize>>>(d_data, d_histogram, n);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&gpu_time, start, stop));

    CHECK_CUDA(cudaMemcpy(h_histogram_gpu, d_histogram, hist_bytes,
                          cudaMemcpyDeviceToHost));

    bool v4_correct = verifyHistogram(h_histogram_ref, h_histogram_gpu, NUM_BINS);
    printf("Correctness: %s\n", v4_correct ? "PASSED" : "FAILED");
    printf("Time: %.3f ms\n", gpu_time);
    printf("Speedup vs V3: %.2fx\n\n", v3_time / gpu_time);

    // ===== Display Results =====
    printHistogram("Histogram Distribution", h_histogram_ref, NUM_BINS);

    // ===== Performance Summary =====
    printf("\n=== Performance Summary ===\n\n");
    printf("%-30s %10s %12s\n", "Version", "Time (ms)", "Speedup");
    printf("%-30s %10.3f %12s\n", "CPU Reference", cpu_time, "1.00x");
    printf("%-30s %10.3f %11.2fx\n", "V1: Global Atomic", v1_time,
           cpu_time / v1_time);
    printf("%-30s %10.3f %11.2fx\n", "V2: Shared Memory", v2_time,
           v1_time / v2_time);
    printf("%-30s %10.3f %11.2fx\n", "V3: Multi-Element", v3_time,
           v2_time / v3_time);
    printf("%-30s %10.3f %11.2fx\n", "V4: Privatized", gpu_time,
           v3_time / gpu_time);
    printf("\n");

    printf("Key Insights:\n");
    printf("1. Shared memory atomics are faster than global atomics\n");
    printf("2. Reducing atomic contention improves performance\n");
    printf("3. Privatization reduces contention on shared memory\n");
    printf("4. Multiple elements per thread increases efficiency\n\n");

    printf("Atomic Operations Characteristics:\n");
    printf("- Provide thread-safe updates without locks\n");
    printf("- Serialize access to the same memory location\n");
    printf("- Shared memory atomics faster than global\n");
    printf("- Consider alternatives (reduction) when possible\n");

    // Cleanup
    free(h_data);
    free(h_histogram_ref);
    free(h_histogram_gpu);

    cudaFree(d_data);
    cudaFree(d_histogram);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\n=== All tests completed successfully! ===\n");

    return 0;
}
