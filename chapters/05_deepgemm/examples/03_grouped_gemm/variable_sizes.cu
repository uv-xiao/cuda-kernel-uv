#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Work-stealing grouped GEMM with dynamic task assignment
__global__ void grouped_gemm_work_stealing(
    const float* __restrict__ A_concat,
    const float* __restrict__ B_concat,
    float* __restrict__ C_concat,
    const int* __restrict__ group_offsets,
    const int* __restrict__ M_sizes,
    int* __restrict__ work_counter,
    int K, int N, int num_groups, int total_tiles) {

    const int TILE_SIZE = 16;
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    while (true) {
        // Atomically grab next tile
        int tile_id = atomicAdd(work_counter, 1);
        if (tile_id >= total_tiles) break;

        // Find which group this tile belongs to
        int group_id = 0;
        int tile_offset = tile_id;

        for (int g = 0; g < num_groups; g++) {
            int M = M_sizes[g];
            int tiles_per_group = ((M + TILE_SIZE - 1) / TILE_SIZE) *
                                  ((N + TILE_SIZE - 1) / TILE_SIZE);
            if (tile_offset < tiles_per_group) {
                group_id = g;
                break;
            }
            tile_offset -= tiles_per_group;
        }

        int M = M_sizes[group_id];
        int offset = group_offsets[group_id];

        int tiles_n = (N + TILE_SIZE - 1) / TILE_SIZE;
        int tile_row = tile_offset / tiles_n;
        int tile_col = tile_offset % tiles_n;

        int row = tile_row * TILE_SIZE + ty;
        int col = tile_col * TILE_SIZE + tx;

        if (row >= M) continue;

        const float* A = A_concat + offset * K;
        const float* B = B_concat + group_id * K * N;
        float* C = C_concat + offset * N;

        float sum = 0.0f;

        for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
            int k_idx = t * TILE_SIZE + tx;
            As[ty][tx] = (row < M && k_idx < K) ? A[row * K + k_idx] : 0.0f;

            k_idx = t * TILE_SIZE + ty;
            Bs[ty][tx] = (k_idx < K && col < N) ? B[k_idx * N + col] : 0.0f;

            __syncthreads();

            #pragma unroll
            for (int k = 0; k < TILE_SIZE; k++) {
                sum += As[ty][k] * Bs[k][tx];
            }
            __syncthreads();
        }

        if (row < M && col < N) {
            C[row * N + col] = sum;
        }
    }
}

// Benchmark different load balancing strategies
void benchmark_load_balancing() {
    printf("=== Variable-Size Grouped GEMM Load Balancing ===\n\n");

    const int num_experts = 8;
    const int K = 2048;
    const int N = 2048;

    // Highly imbalanced distribution
    std::vector<int> M_sizes = {500, 50, 480, 30, 450, 60, 420, 100};

    int total_tokens = 0;
    for (int m : M_sizes) total_tokens += m;

    printf("Configuration:\n");
    printf("  Experts: %d, K: %d, N: %d\n", num_experts, K, N);
    printf("  Token distribution: [");
    for (int i = 0; i < num_experts; i++) {
        printf("%d", M_sizes[i]);
        if (i < num_experts - 1) printf(", ");
    }
    printf("]\n");
    printf("  Total tokens: %d\n", total_tokens);

    // Show imbalance
    int max_tokens = *std::max_element(M_sizes.begin(), M_sizes.end());
    int min_tokens = *std::min_element(M_sizes.begin(), M_sizes.end());
    printf("  Load imbalance ratio: %.1fx\n\n", (float)max_tokens / min_tokens);

    // Setup data
    std::vector<int> group_offsets(num_experts + 1, 0);
    for (int i = 0; i < num_experts; i++) {
        group_offsets[i + 1] = group_offsets[i] + M_sizes[i];
    }

    float *d_A, *d_B, *d_C;
    int *d_M_sizes, *d_offsets, *d_work_counter;

    CHECK_CUDA(cudaMalloc(&d_A, total_tokens * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, num_experts * K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, total_tokens * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_M_sizes, num_experts * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_offsets, (num_experts + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_work_counter, sizeof(int)));

    cudaMemset(d_A, 0, total_tokens * K * sizeof(float));
    cudaMemset(d_B, 0, num_experts * K * N * sizeof(float));

    CHECK_CUDA(cudaMemcpy(d_M_sizes, M_sizes.data(),
                          num_experts * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_offsets, group_offsets.data(),
                          (num_experts + 1) * sizeof(int), cudaMemcpyHostToDevice));

    // Calculate total tiles
    const int TILE_SIZE = 16;
    int total_tiles = 0;
    for (int m : M_sizes) {
        int tiles_m = (m + TILE_SIZE - 1) / TILE_SIZE;
        int tiles_n = (N + TILE_SIZE - 1) / TILE_SIZE;
        total_tiles += tiles_m * tiles_n;
    }

    printf("Total tiles to process: %d\n\n", total_tiles);

    // Get SM count
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int num_sms = prop.multiProcessorCount;
    printf("Device: %s (%d SMs)\n\n", prop.name, num_sms);

    // Benchmark work-stealing approach
    dim3 blockDim(16, 16);
    int num_blocks = num_sms * 4;  // Multiple blocks per SM

    const int warmup = 5;
    const int iters = 50;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Work-Stealing Strategy (%d blocks):\n", num_blocks);

    for (int i = 0; i < warmup; i++) {
        cudaMemset(d_work_counter, 0, sizeof(int));
        grouped_gemm_work_stealing<<<num_blocks, blockDim>>>(
            d_A, d_B, d_C, d_offsets, d_M_sizes, d_work_counter,
            K, N, num_experts, total_tiles);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        cudaMemset(d_work_counter, 0, sizeof(int));
        grouped_gemm_work_stealing<<<num_blocks, blockDim>>>(
            d_A, d_B, d_C, d_offsets, d_M_sizes, d_work_counter,
            K, N, num_experts, total_tiles);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= iters;

    long long total_flops = 0;
    for (int m : M_sizes) {
        total_flops += 2LL * m * N * K;
    }
    double tflops = (total_flops / 1e12) / (ms / 1000.0);

    printf("  Time: %.3f ms\n", ms);
    printf("  Throughput: %.1f TFLOPS\n\n", tflops);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_M_sizes);
    cudaFree(d_offsets);
    cudaFree(d_work_counter);
}

int main() {
    benchmark_load_balancing();
    return 0;
}
