#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// Solution: Grouped GEMM Kernel
// ============================================================================

/*
 * Design decisions:
 * 1. Use tile-based computation with shared memory
 * 2. Map one thread block to one output tile
 * 3. Linear search to find group from tile ID (could optimize with binary search)
 * 4. Handle boundary cases where M < TILE_SIZE
 */

template<int TILE_SIZE>
__global__ void grouped_gemm_kernel(
    const float* __restrict__ A_concat,
    const float* __restrict__ B_concat,
    float* __restrict__ C_concat,
    const int* __restrict__ group_offsets,
    const int* __restrict__ M_sizes,
    int K, int N, int num_groups) {

    // Shared memory for tiling
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global tile ID
    int global_tile_id = blockIdx.x;

    // Find which group this tile belongs to
    int group_id = 0;
    int tile_id_in_group = global_tile_id;

    for (int g = 0; g < num_groups; g++) {
        int M = M_sizes[g];
        int tiles_m = (M + TILE_SIZE - 1) / TILE_SIZE;
        int tiles_n = (N + TILE_SIZE - 1) / TILE_SIZE;
        int tiles_in_group = tiles_m * tiles_n;

        if (tile_id_in_group < tiles_in_group) {
            group_id = g;
            break;
        }

        tile_id_in_group -= tiles_in_group;
    }

    // Get group parameters
    int M = M_sizes[group_id];
    int offset = group_offsets[group_id];

    // Calculate tile position within group
    int tiles_n = (N + TILE_SIZE - 1) / TILE_SIZE;
    int tile_row = tile_id_in_group / tiles_n;
    int tile_col = tile_id_in_group % tiles_n;

    // Global row and column for this thread
    int row = tile_row * TILE_SIZE + ty;
    int col = tile_col * TILE_SIZE + tx;

    // Early exit if completely outside bounds
    if (row >= M) return;

    // Pointers to this group's data
    const float* A = A_concat + offset * K;
    const float* B = B_concat + group_id * K * N;
    float* C = C_concat + offset * N;

    // Accumulator
    float sum = 0.0f;

    // Tile across K dimension
    int num_tiles_k = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles_k; t++) {
        // Load tile of A into shared memory
        int k_idx = t * TILE_SIZE + tx;
        if (row < M && k_idx < K) {
            As[ty][tx] = A[row * K + k_idx];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load tile of B into shared memory
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
// Host function to launch grouped GEMM
// ============================================================================

void launch_grouped_gemm(
    const float* A_concat,
    const float* B_concat,
    float* C_concat,
    const int* group_offsets,
    const int* M_sizes,
    int K, int N, int num_groups) {

    const int TILE_SIZE = 16;

    // Calculate total number of tiles across all groups
    int total_tiles = 0;

    int* h_M_sizes = (int*)malloc(num_groups * sizeof(int));
    CHECK_CUDA(cudaMemcpy(h_M_sizes, M_sizes, num_groups * sizeof(int),
                          cudaMemcpyDeviceToHost));

    for (int g = 0; g < num_groups; g++) {
        int M = h_M_sizes[g];
        int tiles_m = (M + TILE_SIZE - 1) / TILE_SIZE;
        int tiles_n = (N + TILE_SIZE - 1) / TILE_SIZE;
        total_tiles += tiles_m * tiles_n;
    }

    free(h_M_sizes);

    // Launch kernel
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(total_tiles);

    grouped_gemm_kernel<TILE_SIZE><<<gridDim, blockDim>>>(
        A_concat, B_concat, C_concat, group_offsets, M_sizes, K, N, num_groups);

    CHECK_CUDA(cudaGetLastError());
}

// ============================================================================
// Reference CPU implementation
// ============================================================================

void grouped_gemm_cpu(
    const float* A_concat,
    const float* B_concat,
    float* C_concat,
    const int* group_offsets,
    const int* M_sizes,
    int K, int N, int num_groups) {

    for (int g = 0; g < num_groups; g++) {
        int M = M_sizes[g];
        int offset = group_offsets[g];

        const float* A = A_concat + offset * K;
        const float* B = B_concat + g * K * N;
        float* C = C_concat + offset * N;

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }
}

// ============================================================================
// Test harness
// ============================================================================

void test_grouped_gemm(const int* M_sizes, int num_groups, int K, int N) {
    printf("\n=== Test: num_groups=%d, K=%d, N=%d ===\n", num_groups, K, N);
    printf("M sizes: [");
    for (int i = 0; i < num_groups; i++) {
        printf("%d", M_sizes[i]);
        if (i < num_groups - 1) printf(", ");
    }
    printf("]\n");

    int* group_offsets = (int*)malloc((num_groups + 1) * sizeof(int));
    group_offsets[0] = 0;
    for (int i = 0; i < num_groups; i++) {
        group_offsets[i + 1] = group_offsets[i] + M_sizes[i];
    }
    int total_M = group_offsets[num_groups];

    printf("Total M: %d\n", total_M);

    float* h_A = (float*)malloc(total_M * K * sizeof(float));
    float* h_B = (float*)malloc(num_groups * K * N * sizeof(float));
    float* h_C_gpu = (float*)malloc(total_M * N * sizeof(float));
    float* h_C_cpu = (float*)malloc(total_M * N * sizeof(float));

    srand(42);
    for (int i = 0; i < total_M * K; i++) {
        h_A[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }
    for (int i = 0; i < num_groups * K * N; i++) {
        h_B[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }

    float *d_A, *d_B, *d_C;
    int *d_group_offsets, *d_M_sizes;

    CHECK_CUDA(cudaMalloc(&d_A, total_M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, num_groups * K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, total_M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_group_offsets, (num_groups + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_M_sizes, num_groups * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, total_M * K * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, num_groups * K * N * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_group_offsets, group_offsets,
                          (num_groups + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_M_sizes, M_sizes, num_groups * sizeof(int),
                          cudaMemcpyHostToDevice));

    launch_grouped_gemm(d_A, d_B, d_C, d_group_offsets, d_M_sizes, K, N, num_groups);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_C_gpu, d_C, total_M * N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    grouped_gemm_cpu(h_A, h_B, h_C_cpu, group_offsets, M_sizes, K, N, num_groups);

    float max_error = 0.0f;
    for (int i = 0; i < total_M * N; i++) {
        float error = fabsf(h_C_gpu[i] - h_C_cpu[i]);
        max_error = fmaxf(max_error, error);
    }

    printf("Max error: %.6e\n", max_error);
    if (max_error < 1e-3f) {
        printf("PASSED\n");
    } else {
        printf("FAILED\n");
    }

    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);
    free(group_offsets);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_group_offsets);
    cudaFree(d_M_sizes);
}

int main() {
    printf("=== Grouped GEMM Solution ===\n");

    {
        int M_sizes[] = {128, 128, 128, 128};
        test_grouped_gemm(M_sizes, 4, 256, 256);
    }

    {
        int M_sizes[] = {64, 128, 32, 256, 16, 192, 96, 48};
        test_grouped_gemm(M_sizes, 8, 512, 512);
    }

    {
        int M_sizes[] = {1, 10, 100, 1000};
        test_grouped_gemm(M_sizes, 4, 1024, 1024);
    }

    {
        int M_sizes[] = {8, 12, 16, 24, 32, 8, 16, 12, 20, 28, 16, 8, 24, 32, 16, 12};
        test_grouped_gemm(M_sizes, 16, 128, 128);
    }

    return 0;
}
