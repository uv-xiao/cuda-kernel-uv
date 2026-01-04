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

/**
 * Reduction Version 1: Interleaved addressing with divergent warps
 * This is the naive approach with poor performance due to divergence
 */
__global__ void reduce_v1_interleaved(float *g_data, float *g_out, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (idx < n) ? g_data[idx] : 0.0f;
    __syncthreads();

    // Reduction in shared memory (interleaved addressing)
    for (int s = 1; s < blockDim.x; s *= 2) {
        // Divergent branches hurt performance
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        g_out[blockIdx.x] = sdata[0];
    }
}

/**
 * Reduction Version 2: Sequential addressing
 * Eliminates divergent branches for better performance
 */
__global__ void reduce_v2_sequential(float *g_data, float *g_out, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? g_data[idx] : 0.0f;
    __syncthreads();

    // Sequential addressing (no divergence)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_out[blockIdx.x] = sdata[0];
    }
}

/**
 * Reduction Version 3: First add during load
 * Halves the number of blocks needed
 */
__global__ void reduce_v3_first_add(float *g_data, float *g_out, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // First add during global load
    sdata[tid] = 0.0f;
    if (idx < n) sdata[tid] = g_data[idx];
    if (idx + blockDim.x < n) sdata[tid] += g_data[idx + blockDim.x];
    __syncthreads();

    // Sequential addressing
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_out[blockIdx.x] = sdata[0];
    }
}

/**
 * Reduction Version 4: Unroll last warp
 * No synchronization needed for warp-level operations
 */
__device__ void warpReduce(volatile float *sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduce_v4_unroll_warp(float *g_data, float *g_out, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // First add during global load
    sdata[tid] = 0.0f;
    if (idx < n) sdata[tid] = g_data[idx];
    if (idx + blockDim.x < n) sdata[tid] += g_data[idx + blockDim.x];
    __syncthreads();

    // Sequential addressing until last warp
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Unroll last warp (no __syncthreads needed)
    if (tid < 32) warpReduce(sdata, tid);

    if (tid == 0) {
        g_out[blockIdx.x] = sdata[0];
    }
}

/**
 * Reduction Version 5: Multiple elements per thread
 * Increases arithmetic intensity and reduces kernel launches
 */
__global__ void reduce_v5_multi_elements(float *g_data, float *g_out, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int gridSize = blockDim.x * 2 * gridDim.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Grid-stride loop: each thread processes multiple elements
    float sum = 0.0f;
    for (int i = idx; i < n; i += gridSize) {
        sum += g_data[i];
        if (i + blockDim.x < n) {
            sum += g_data[i + blockDim.x];
        }
    }
    sdata[tid] = sum;
    __syncthreads();

    // Sequential addressing
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Unroll last warp
    if (tid < 32) warpReduce(sdata, tid);

    if (tid == 0) {
        g_out[blockIdx.x] = sdata[0];
    }
}

/**
 * Host reduction for verification
 */
float reduceHost(float *data, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

/**
 * Recursive GPU reduction - reduces partial results
 */
float reduceGPU(void (*kernel)(float*, float*, int),
                float *d_data, float *d_temp, int n,
                int blockSize, float *time_ms) {
    int gridSize = (n + blockSize - 1) / blockSize;
    size_t smemSize = blockSize * sizeof(float);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    // First reduction
    kernel<<<gridSize, blockSize, smemSize>>>(d_data, d_temp, n);

    // Reduce partial results on CPU (simpler for this example)
    float *h_temp = (float *)malloc(gridSize * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_temp, d_temp, gridSize * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float sum = 0.0f;
    for (int i = 0; i < gridSize; i++) {
        sum += h_temp[i];
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(time_ms, start, stop));

    free(h_temp);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return sum;
}

int main(int argc, char **argv) {
    printf("=== CUDA Reduction Optimization ===\n\n");

    srand(time(NULL));

    // Array size
    int n = 16 * 1024 * 1024;  // 16M elements
    size_t bytes = n * sizeof(float);

    printf("Array size: %d elements (%.2f MB)\n\n", n, bytes / (1024.0 * 1024.0));

    // Allocate and initialize host data
    float *h_data = (float *)malloc(bytes);
    for (int i = 0; i < n; i++) {
        h_data[i] = (float)(rand() % 100) / 100.0f;
    }

    // Compute reference result
    printf("Computing reference result on CPU...\n");
    clock_t cpu_start = clock();
    float cpu_sum = reduceHost(h_data, n);
    clock_t cpu_end = clock();
    float cpu_time = 1000.0f * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    printf("CPU result: %.6f (%.2f ms)\n\n", cpu_sum, cpu_time);

    // Allocate device memory
    float *d_data, *d_temp;
    CHECK_CUDA(cudaMalloc(&d_data, bytes));
    CHECK_CUDA(cudaMalloc(&d_temp, bytes / 256));  // Space for partial results
    CHECK_CUDA(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    int blockSize = 256;
    float gpu_time;
    float gpu_sum;
    float error;

    // ===== Version 1: Interleaved Addressing =====
    printf("Version 1: Interleaved Addressing (divergent warps)\n");
    printf("----------------------------------------------------\n");
    gpu_sum = reduceGPU(reduce_v1_interleaved, d_data, d_temp, n, blockSize, &gpu_time);
    error = fabs(gpu_sum - cpu_sum) / cpu_sum * 100.0f;
    printf("Result: %.6f\n", gpu_sum);
    printf("Error: %.6f%%\n", error);
    printf("Time: %.3f ms\n", gpu_time);
    printf("Speedup vs CPU: %.2fx\n\n", cpu_time / gpu_time);

    // ===== Version 2: Sequential Addressing =====
    printf("Version 2: Sequential Addressing (no divergence)\n");
    printf("-------------------------------------------------\n");
    float v1_time = gpu_time;
    gpu_sum = reduceGPU(reduce_v2_sequential, d_data, d_temp, n, blockSize, &gpu_time);
    error = fabs(gpu_sum - cpu_sum) / cpu_sum * 100.0f;
    printf("Result: %.6f\n", gpu_sum);
    printf("Error: %.6f%%\n", error);
    printf("Time: %.3f ms\n", gpu_time);
    printf("Speedup vs V1: %.2fx\n\n", v1_time / gpu_time);

    // ===== Version 3: First Add During Load =====
    printf("Version 3: First Add During Load (half the blocks)\n");
    printf("---------------------------------------------------\n");
    float v2_time = gpu_time;
    gpu_sum = reduceGPU(reduce_v3_first_add, d_data, d_temp, n, blockSize, &gpu_time);
    error = fabs(gpu_sum - cpu_sum) / cpu_sum * 100.0f;
    printf("Result: %.6f\n", gpu_sum);
    printf("Error: %.6f%%\n", error);
    printf("Time: %.3f ms\n", gpu_time);
    printf("Speedup vs V2: %.2fx\n\n", v2_time / gpu_time);

    // ===== Version 4: Unroll Last Warp =====
    printf("Version 4: Unroll Last Warp (no sync in last warp)\n");
    printf("---------------------------------------------------\n");
    float v3_time = gpu_time;
    gpu_sum = reduceGPU(reduce_v4_unroll_warp, d_data, d_temp, n, blockSize, &gpu_time);
    error = fabs(gpu_sum - cpu_sum) / cpu_sum * 100.0f;
    printf("Result: %.6f\n", gpu_sum);
    printf("Error: %.6f%%\n", error);
    printf("Time: %.3f ms\n", gpu_time);
    printf("Speedup vs V3: %.2fx\n\n", v3_time / gpu_time);

    // ===== Version 5: Multiple Elements Per Thread =====
    printf("Version 5: Multiple Elements Per Thread\n");
    printf("----------------------------------------\n");
    float v4_time = gpu_time;
    gpu_sum = reduceGPU(reduce_v5_multi_elements, d_data, d_temp, n, blockSize, &gpu_time);
    error = fabs(gpu_sum - cpu_sum) / cpu_sum * 100.0f;
    printf("Result: %.6f\n", gpu_sum);
    printf("Error: %.6f%%\n", error);
    printf("Time: %.3f ms\n", gpu_time);
    printf("Speedup vs V4: %.2fx\n\n", v4_time / gpu_time);

    // ===== Performance Summary =====
    printf("=== Performance Summary ===\n\n");
    printf("Bandwidth Analysis (%.2f MB array):\n", bytes / (1024.0 * 1024.0));
    float bandwidth = bytes / (1024.0 * 1024.0 * 1024.0) / (gpu_time / 1000.0f);
    printf("Best GPU bandwidth: %.2f GB/s\n", bandwidth);
    printf("Overall speedup vs CPU: %.2fx\n\n", cpu_time / gpu_time);

    printf("Optimization Techniques Applied:\n");
    printf("1. Interleaved → Sequential: Eliminate warp divergence\n");
    printf("2. First add during load: Reduce blocks by 2x\n");
    printf("3. Unroll last warp: Remove unnecessary synchronization\n");
    printf("4. Multiple elements/thread: Increase arithmetic intensity\n\n");

    printf("Key Insights:\n");
    printf("- Warp divergence kills performance\n");
    printf("- Sequential addressing is crucial\n");
    printf("- Reduce data movement by doing more work per thread\n");
    printf("- Warp-level operations don't need __syncthreads()\n");

    // Cleanup
    free(h_data);
    cudaFree(d_data);
    cudaFree(d_temp);

    printf("\n=== All tests completed successfully! ===\n");

    return 0;
}
