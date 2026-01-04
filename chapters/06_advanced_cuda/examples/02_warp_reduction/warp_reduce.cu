#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Error checking macro
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Traditional shared memory reduction (for comparison)
__global__ void reduce_shared_memory(int *output, const int *input, int n) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load from global memory to shared memory
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Warp-level reduction using shuffle (SUM)
__device__ int warp_reduce_sum(int value) {
    // Use XOR-based butterfly reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffff, value, offset);
    }
    return value;
}

// Warp-level reduction using shuffle (MAX)
__device__ int warp_reduce_max(int value) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        int other = __shfl_xor_sync(0xffffffff, value, offset);
        value = max(value, other);
    }
    return value;
}

// Warp-level reduction using shuffle (MIN)
__device__ int warp_reduce_min(int value) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        int other = __shfl_xor_sync(0xffffffff, value, offset);
        value = min(value, other);
    }
    return value;
}

// Block reduction using warp primitives
// Each warp reduces independently, then warp leaders use shared memory
__global__ void reduce_warp_shuffle(int *output, const int *input, int n) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load value
    int value = (i < n) ? input[i] : 0;

    // Reduce within warp using shuffle
    value = warp_reduce_sum(value);

    // Shared memory for warp results
    __shared__ int warp_sums[32]; // Max 32 warps per block (1024 threads)

    int lane_id = tid % warpSize;
    int warp_id = tid / warpSize;

    // First thread in each warp writes warp result to shared memory
    if (lane_id == 0) {
        warp_sums[warp_id] = value;
    }
    __syncthreads();

    // First warp reduces the warp results
    if (warp_id == 0) {
        int warp_count = (blockDim.x + warpSize - 1) / warpSize;
        value = (tid < warp_count) ? warp_sums[tid] : 0;
        value = warp_reduce_sum(value);

        if (tid == 0) {
            output[blockIdx.x] = value;
        }
    }
}

// Optimized version: reduce directly without intermediate shared memory
// Only works when blockDim.x <= warpSize
__global__ void reduce_single_warp(int *output, const int *input, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int value = (i < n) ? input[i] : 0;
    value = warp_reduce_sum(value);

    if (threadIdx.x == 0) {
        output[blockIdx.x] = value;
    }
}

// Multi-operation reduction: compute sum, max, and min simultaneously
__global__ void reduce_multiple_ops(int *sum_out, int *max_out, int *min_out,
                                    const int *input, int n) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int value = (i < n) ? input[i] : 0;

    // Perform all three reductions in parallel
    int sum_val = warp_reduce_sum(value);
    int max_val = warp_reduce_max(value);
    int min_val = warp_reduce_min(value);

    __shared__ int warp_sums[32], warp_maxs[32], warp_mins[32];

    int lane_id = tid % warpSize;
    int warp_id = tid / warpSize;

    if (lane_id == 0) {
        warp_sums[warp_id] = sum_val;
        warp_maxs[warp_id] = max_val;
        warp_mins[warp_id] = min_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        int warp_count = (blockDim.x + warpSize - 1) / warpSize;

        sum_val = (tid < warp_count) ? warp_sums[tid] : 0;
        max_val = (tid < warp_count) ? warp_maxs[tid] : INT_MIN;
        min_val = (tid < warp_count) ? warp_mins[tid] : INT_MAX;

        sum_val = warp_reduce_sum(sum_val);
        max_val = warp_reduce_max(max_val);
        min_val = warp_reduce_min(min_val);

        if (tid == 0) {
            sum_out[blockIdx.x] = sum_val;
            max_out[blockIdx.x] = max_val;
            min_out[blockIdx.x] = min_val;
        }
    }
}

// CPU reference implementation
int cpu_reduce_sum(const int *data, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

int cpu_reduce_max(const int *data, int n) {
    int max_val = INT_MIN;
    for (int i = 0; i < n; i++) {
        max_val = (data[i] > max_val) ? data[i] : max_val;
    }
    return max_val;
}

int cpu_reduce_min(const int *data, int n) {
    int min_val = INT_MAX;
    for (int i = 0; i < n; i++) {
        min_val = (data[i] < min_val) ? data[i] : min_val;
    }
    return min_val;
}

void benchmark_reductions(const int *d_input, int n, int num_iterations) {
    int blocks = (n + 255) / 256;
    int *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, blocks * sizeof(int)));

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Warmup
    reduce_shared_memory<<<blocks, 256, 256 * sizeof(int)>>>(d_output, d_input, n);
    reduce_warp_shuffle<<<blocks, 256>>>(d_output, d_input, n);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Benchmark shared memory version
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; i++) {
        reduce_shared_memory<<<blocks, 256, 256 * sizeof(int)>>>(d_output, d_input, n);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float shared_time;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&shared_time, start, stop));

    // Benchmark warp shuffle version
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; i++) {
        reduce_warp_shuffle<<<blocks, 256>>>(d_output, d_input, n);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float shuffle_time;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&shuffle_time, start, stop));

    printf("\nPerformance Comparison (%d iterations):\n", num_iterations);
    printf("  Shared Memory: %.3f ms (avg: %.3f us)\n",
           shared_time, shared_time * 1000.0f / num_iterations);
    printf("  Warp Shuffle:  %.3f ms (avg: %.3f us)\n",
           shuffle_time, shuffle_time * 1000.0f / num_iterations);
    printf("  Speedup:       %.2fx\n", shared_time / shuffle_time);

    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaFree(d_output));
}

int main() {
    printf("========== Warp Reduction Examples ==========\n\n");

    const int n = 1024;
    const int bytes = n * sizeof(int);

    // Allocate and initialize host memory
    int *h_input = (int*)malloc(bytes);
    srand(42);
    for (int i = 0; i < n; i++) {
        h_input[i] = rand() % 100;
    }

    // Compute reference on CPU
    int cpu_sum = cpu_reduce_sum(h_input, n);
    int cpu_max = cpu_reduce_max(h_input, n);
    int cpu_min = cpu_reduce_min(h_input, n);

    printf("Array size: %d elements\n", n);
    printf("CPU Reference: sum=%d, max=%d, min=%d\n\n",
           cpu_sum, cpu_max, cpu_min);

    // Allocate device memory
    int *d_input;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, bytes));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // Test 1: Shared memory reduction
    printf("1. Shared Memory Reduction:\n");
    {
        int blocks = (n + 255) / 256;
        int *d_output, *h_output;
        CHECK_CUDA_ERROR(cudaMalloc(&d_output, blocks * sizeof(int)));
        h_output = (int*)malloc(blocks * sizeof(int));

        reduce_shared_memory<<<blocks, 256, 256 * sizeof(int)>>>(d_output, d_input, n);
        CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, blocks * sizeof(int),
                                     cudaMemcpyDeviceToHost));

        int gpu_sum = cpu_reduce_sum(h_output, blocks);
        printf("   GPU sum: %d, CPU sum: %d, %s\n",
               gpu_sum, cpu_sum, (gpu_sum == cpu_sum) ? "PASS" : "FAIL");

        free(h_output);
        CHECK_CUDA_ERROR(cudaFree(d_output));
    }

    // Test 2: Warp shuffle reduction
    printf("\n2. Warp Shuffle Reduction:\n");
    {
        int blocks = (n + 255) / 256;
        int *d_output, *h_output;
        CHECK_CUDA_ERROR(cudaMalloc(&d_output, blocks * sizeof(int)));
        h_output = (int*)malloc(blocks * sizeof(int));

        reduce_warp_shuffle<<<blocks, 256>>>(d_output, d_input, n);
        CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, blocks * sizeof(int),
                                     cudaMemcpyDeviceToHost));

        int gpu_sum = cpu_reduce_sum(h_output, blocks);
        printf("   GPU sum: %d, CPU sum: %d, %s\n",
               gpu_sum, cpu_sum, (gpu_sum == cpu_sum) ? "PASS" : "FAIL");

        free(h_output);
        CHECK_CUDA_ERROR(cudaFree(d_output));
    }

    // Test 3: Single warp reduction (for small arrays)
    printf("\n3. Single Warp Reduction (32 elements):\n");
    {
        int small_n = 32;
        int blocks = (small_n + 31) / 32;
        int *d_output, *h_output;
        CHECK_CUDA_ERROR(cudaMalloc(&d_output, blocks * sizeof(int)));
        h_output = (int*)malloc(blocks * sizeof(int));

        reduce_single_warp<<<blocks, 32>>>(d_output, d_input, small_n);
        CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, blocks * sizeof(int),
                                     cudaMemcpyDeviceToHost));

        int gpu_sum = cpu_reduce_sum(h_output, blocks);
        int cpu_sum_small = cpu_reduce_sum(h_input, small_n);
        printf("   GPU sum: %d, CPU sum: %d, %s\n",
               gpu_sum, cpu_sum_small, (gpu_sum == cpu_sum_small) ? "PASS" : "FAIL");

        free(h_output);
        CHECK_CUDA_ERROR(cudaFree(d_output));
    }

    // Test 4: Multiple operations simultaneously
    printf("\n4. Multiple Reductions (sum, max, min):\n");
    {
        int blocks = (n + 255) / 256;
        int *d_sum, *d_max, *d_min;
        int *h_sum, *h_max, *h_min;

        CHECK_CUDA_ERROR(cudaMalloc(&d_sum, blocks * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_max, blocks * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_min, blocks * sizeof(int)));

        h_sum = (int*)malloc(blocks * sizeof(int));
        h_max = (int*)malloc(blocks * sizeof(int));
        h_min = (int*)malloc(blocks * sizeof(int));

        reduce_multiple_ops<<<blocks, 256>>>(d_sum, d_max, d_min, d_input, n);

        CHECK_CUDA_ERROR(cudaMemcpy(h_sum, d_sum, blocks * sizeof(int),
                                     cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(h_max, d_max, blocks * sizeof(int),
                                     cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(h_min, d_min, blocks * sizeof(int),
                                     cudaMemcpyDeviceToHost));

        int gpu_sum = cpu_reduce_sum(h_sum, blocks);
        int gpu_max = cpu_reduce_max(h_max, blocks);
        int gpu_min = cpu_reduce_min(h_min, blocks);

        printf("   Sum: GPU=%d, CPU=%d, %s\n",
               gpu_sum, cpu_sum, (gpu_sum == cpu_sum) ? "PASS" : "FAIL");
        printf("   Max: GPU=%d, CPU=%d, %s\n",
               gpu_max, cpu_max, (gpu_max == cpu_max) ? "PASS" : "FAIL");
        printf("   Min: GPU=%d, CPU=%d, %s\n",
               gpu_min, cpu_min, (gpu_min == cpu_min) ? "PASS" : "FAIL");

        free(h_sum);
        free(h_max);
        free(h_min);
        CHECK_CUDA_ERROR(cudaFree(d_sum));
        CHECK_CUDA_ERROR(cudaFree(d_max));
        CHECK_CUDA_ERROR(cudaFree(d_min));
    }

    // Benchmark
    benchmark_reductions(d_input, n, 1000);

    // Cleanup
    free(h_input);
    CHECK_CUDA_ERROR(cudaFree(d_input));

    printf("\n========== All Tests Completed ==========\n");

    return 0;
}
