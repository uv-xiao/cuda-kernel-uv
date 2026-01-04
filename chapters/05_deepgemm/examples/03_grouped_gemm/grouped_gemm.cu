#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            printf("cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// Simple Grouped GEMM: Concatenate inputs and process sequentially
// ============================================================================

// Compute C = A @ B for multiple groups
// A[i]: M[i] x K, B[i]: K x N, C[i]: M[i] x N
__global__ void grouped_gemm_naive_kernel(
    const float* const* A_ptrs,  // Array of input pointers
    const float* const* B_ptrs,  // Array of weight pointers
    float* const* C_ptrs,        // Array of output pointers
    const int* M_sizes,          // Number of rows for each group
    int K, int N,                // Shared dimensions
    int num_groups) {

    int group_id = blockIdx.x;
    if (group_id >= num_groups) return;

    const float* A = A_ptrs[group_id];
    const float* B = B_ptrs[group_id];
    float* C = C_ptrs[group_id];
    int M = M_sizes[group_id];

    // Simple tile-based GEMM within each group
    int tile_size = 16;
    int row = (blockIdx.y * blockDim.y + threadIdx.y);
    int col = (blockIdx.z * blockDim.x + threadIdx.x);

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Optimized Grouped GEMM: Use shared memory tiling
// ============================================================================

template<int TILE_SIZE>
__global__ void grouped_gemm_tiled_kernel(
    const float* __restrict__ A_concat,  // Concatenated inputs
    const float* __restrict__ B_concat,  // Concatenated weights
    float* __restrict__ C_concat,        // Concatenated outputs
    const int* __restrict__ group_offsets,  // Cumulative offsets for each group
    const int* __restrict__ M_sizes,     // M dimension for each group
    int K, int N, int num_groups) {

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Find which group this block is processing
    int group_id = blockIdx.x % num_groups;
    int tile_id = blockIdx.x / num_groups;

    int M = M_sizes[group_id];
    int offset = group_offsets[group_id];

    // Calculate tile position
    int tiles_per_row = (N + TILE_SIZE - 1) / TILE_SIZE;
    int tile_row = tile_id / tiles_per_row;
    int tile_col = tile_id % tiles_per_row;

    int row = tile_row * TILE_SIZE + ty;
    int col = tile_col * TILE_SIZE + tx;

    if (row >= M) return;

    const float* A = A_concat + offset * K;
    const float* B = B_concat + group_id * K * N;
    float* C = C_concat + offset * N;

    float sum = 0.0f;

    // Tile across K dimension
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        int k_idx = t * TILE_SIZE + tx;
        if (row < M && k_idx < K) {
            As[ty][tx] = A[row * K + k_idx];
        } else {
            As[ty][tx] = 0.0f;
        }

        k_idx = t * TILE_SIZE + ty;
        if (k_idx < K && col < N) {
            Bs[ty][tx] = B[k_idx * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Wrapper for cuBLAS batched GEMM (with padding)
// ============================================================================

void cublas_batched_gemm(cublasHandle_t handle,
                        const std::vector<float*>& A_ptrs,
                        const std::vector<float*>& B_ptrs,
                        std::vector<float*>& C_ptrs,
                        const std::vector<int>& M_sizes,
                        int K, int N) {
    int num_groups = A_ptrs.size();

    // Find max M for padding
    int max_M = *std::max_element(M_sizes.begin(), M_sizes.end());

    // Use cuBLAS batched GEMM
    float alpha = 1.0f, beta = 0.0f;

    // Prepare device pointer arrays
    float** d_A_array;
    float** d_B_array;
    float** d_C_array;

    CHECK_CUDA(cudaMalloc(&d_A_array, num_groups * sizeof(float*)));
    CHECK_CUDA(cudaMalloc(&d_B_array, num_groups * sizeof(float*)));
    CHECK_CUDA(cudaMalloc(&d_C_array, num_groups * sizeof(float*)));

    CHECK_CUDA(cudaMemcpy(d_A_array, A_ptrs.data(),
                          num_groups * sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_array, B_ptrs.data(),
                          num_groups * sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C_array, C_ptrs.data(),
                          num_groups * sizeof(float*), cudaMemcpyHostToDevice));

    // Note: This is simplified - real implementation would need padding
    CHECK_CUBLAS(cublasSgemmBatched(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, max_M, K,
        &alpha,
        (const float**)d_B_array, N,
        (const float**)d_A_array, K,
        &beta,
        d_C_array, N,
        num_groups));

    cudaFree(d_A_array);
    cudaFree(d_B_array);
    cudaFree(d_C_array);
}

// ============================================================================
// Test and benchmark
// ============================================================================

int main() {
    printf("=== Grouped GEMM Demo ===\n\n");

    // Configuration
    const int num_experts = 8;
    const int K = 2048;  // Hidden dimension
    const int N = 2048;  // Output dimension

    // Variable number of tokens per expert (simulating MoE)
    std::vector<int> M_sizes = {156, 89, 201, 52, 134, 98, 187, 143};

    int total_tokens = 0;
    for (int m : M_sizes) total_tokens += m;

    printf("Configuration:\n");
    printf("  Number of experts: %d\n", num_experts);
    printf("  Hidden dimension (K): %d\n", K);
    printf("  Output dimension (N): %d\n", N);
    printf("  Total tokens: %d\n", total_tokens);
    printf("  Token distribution: [");
    for (int i = 0; i < num_experts; i++) {
        printf("%d", M_sizes[i]);
        if (i < num_experts - 1) printf(", ");
    }
    printf("]\n\n");

    // Compute offsets
    std::vector<int> group_offsets(num_experts + 1);
    group_offsets[0] = 0;
    for (int i = 0; i < num_experts; i++) {
        group_offsets[i + 1] = group_offsets[i] + M_sizes[i];
    }

    // Allocate host memory
    std::vector<float*> h_A_ptrs(num_experts);
    std::vector<float*> h_B_ptrs(num_experts);
    std::vector<float*> h_C_ptrs(num_experts);

    // Allocate concatenated arrays
    float *d_A_concat, *d_B_concat, *d_C_concat;
    int *d_M_sizes, *d_group_offsets;

    CHECK_CUDA(cudaMalloc(&d_A_concat, total_tokens * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B_concat, num_experts * K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C_concat, total_tokens * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_M_sizes, num_experts * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_group_offsets, (num_experts + 1) * sizeof(int)));

    // Initialize with random data
    cudaMemset(d_A_concat, 0, total_tokens * K * sizeof(float));
    cudaMemset(d_B_concat, 0, num_experts * K * N * sizeof(float));

    CHECK_CUDA(cudaMemcpy(d_M_sizes, M_sizes.data(),
                          num_experts * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_group_offsets, group_offsets.data(),
                          (num_experts + 1) * sizeof(int), cudaMemcpyHostToDevice));

    // Setup individual pointers for each expert
    for (int i = 0; i < num_experts; i++) {
        h_A_ptrs[i] = d_A_concat + group_offsets[i] * K;
        h_B_ptrs[i] = d_B_concat + i * K * N;
        h_C_ptrs[i] = d_C_concat + group_offsets[i] * N;
    }

    // Benchmark grouped GEMM
    const int warmup = 10;
    const int iters = 100;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch tiled grouped GEMM
    const int TILE_SIZE = 16;
    dim3 blockDim(TILE_SIZE, TILE_SIZE);

    // Calculate total number of tiles
    int total_tiles = 0;
    for (int i = 0; i < num_experts; i++) {
        int tiles_m = (M_sizes[i] + TILE_SIZE - 1) / TILE_SIZE;
        int tiles_n = (N + TILE_SIZE - 1) / TILE_SIZE;
        total_tiles += tiles_m * tiles_n;
    }

    // Simplified launch (one block per tile per group)
    // In production, use more sophisticated work distribution
    int num_blocks = total_tiles;

    for (int i = 0; i < warmup; i++) {
        grouped_gemm_tiled_kernel<TILE_SIZE><<<num_blocks, blockDim>>>(
            d_A_concat, d_B_concat, d_C_concat,
            d_group_offsets, d_M_sizes, K, N, num_experts);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        grouped_gemm_tiled_kernel<TILE_SIZE><<<num_blocks, blockDim>>>(
            d_A_concat, d_B_concat, d_C_concat,
            d_group_offsets, d_M_sizes, K, N, num_experts);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= iters;

    // Calculate FLOPS
    long long total_flops = 0;
    for (int i = 0; i < num_experts; i++) {
        total_flops += 2LL * M_sizes[i] * N * K;  // 2 * M * N * K
    }

    double tflops = (total_flops / 1e12) / (ms / 1000.0);

    printf("Grouped GEMM Performance:\n");
    printf("  Time: %.3f ms\n", ms);
    printf("  Throughput: %.1f TFLOPS\n", tflops);
    printf("  Total FLOPs: %.2f GFLOPS\n\n", total_flops / 1e9);

    // Compare with theoretical padded approach
    int max_M = *std::max_element(M_sizes.begin(), M_sizes.end());
    long long padded_flops = 2LL * num_experts * max_M * N * K;
    double efficiency = (double)total_flops / padded_flops * 100.0;

    printf("Comparison with Padded Batched GEMM:\n");
    printf("  Padded total FLOPs: %.2f GFLOPS\n", padded_flops / 1e9);
    printf("  Compute efficiency: %.1f%%\n", efficiency);
    printf("  Theoretical speedup: %.2fx\n\n", (double)padded_flops / total_flops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_A_concat);
    cudaFree(d_B_concat);
    cudaFree(d_C_concat);
    cudaFree(d_M_sizes);
    cudaFree(d_group_offsets);

    printf("Note: This is a simplified implementation.\n");
    printf("Production implementations (like DeepGEMM) achieve much higher performance\n");
    printf("through advanced optimizations like persistent kernels and work stealing.\n");

    return 0;
}
