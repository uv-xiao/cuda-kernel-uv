#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include <stdlib.h>

namespace cg = cooperative_groups;

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

// Example 1: Using thread_block (equivalent to traditional block)
__global__ void sum_reduction_thread_block(int *output, const int *input, int n) {
    // Get thread block group
    cg::thread_block block = cg::this_thread_block();

    unsigned int tid = block.thread_rank(); // Same as threadIdx.x
    unsigned int i = block.group_index().x * block.size() + tid; // blockIdx.x * blockDim.x + threadIdx.x

    extern __shared__ int sdata[];

    sdata[tid] = (i < n) ? input[i] : 0;
    block.sync(); // Same as __syncthreads()

    // Reduction
    for (unsigned int s = block.size() / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        block.sync();
    }

    if (tid == 0) {
        output[block.group_index().x] = sdata[0];
    }
}

// Example 2: Using thread_block_tile for warp-level operations
__global__ void sum_reduction_tiled(int *output, const int *input, int n) {
    cg::thread_block block = cg::this_thread_block();

    // Create a tile of 32 threads (a warp)
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);

    unsigned int tid = block.thread_rank();
    unsigned int i = block.group_index().x * block.size() + tid;

    int value = (i < n) ? input[i] : 0;

    // Warp-level reduction using tile
    for (int offset = tile32.size() / 2; offset > 0; offset >>= 1) {
        value += tile32.shfl_down(value, offset);
    }

    // First thread in each warp writes to shared memory
    __shared__ int warp_sums[32];
    if (tile32.thread_rank() == 0) {
        warp_sums[tid / 32] = value;
    }
    block.sync();

    // First warp reduces the warp results
    if (tid < 32) {
        cg::thread_block_tile<32> first_warp = cg::tiled_partition<32>(block);
        int warp_count = (block.size() + 31) / 32;
        value = (tid < warp_count) ? warp_sums[tid] : 0;

        for (int offset = first_warp.size() / 2; offset > 0; offset >>= 1) {
            value += first_warp.shfl_down(value, offset);
        }

        if (first_warp.thread_rank() == 0) {
            output[block.group_index().x] = value;
        }
    }
}

// Example 3: Using coalesced_group for divergent execution
__global__ void process_with_coalesced(int *output, const int *input, int n, int threshold) {
    cg::thread_block block = cg::this_thread_block();

    unsigned int tid = block.thread_rank();
    unsigned int i = block.group_index().x * block.size() + tid;

    if (i >= n) return;

    int value = input[i];

    // Threads diverge based on condition
    if (value > threshold) {
        // Get the coalesced group of active threads
        cg::coalesced_group active = cg::coalesced_threads();

        // Only active threads participate
        int active_count = active.size();
        int active_rank = active.thread_rank();

        // Reduction among active threads only
        for (int offset = active.size() / 2; offset > 0; offset >>= 1) {
            value += active.shfl_down(value, offset);
        }

        if (active_rank == 0) {
            output[i] = value; // Sum of all active threads
        }
    } else {
        output[i] = 0;
    }
}

// Example 4: Comparison of voting operations
__global__ void vote_operations_comparison(int *all_results, int *any_results,
                                          int *ballot_count, const int *input,
                                          int n, int threshold) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);

    unsigned int tid = block.thread_rank();
    unsigned int i = block.group_index().x * block.size() + tid;

    int value = (i < n) ? input[i] : 0;
    bool predicate = (value >= threshold);

    // All threads in tile satisfy predicate?
    bool all_satisfy = tile32.all(predicate);

    // Any thread in tile satisfies predicate?
    bool any_satisfy = tile32.any(predicate);

    // Get ballot and count
    auto ballot = tile32.ballot(predicate);
    int count = __popc(ballot);

    if (tile32.thread_rank() == 0) {
        int warp_id = tid / 32;
        all_results[warp_id] = all_satisfy ? 1 : 0;
        any_results[warp_id] = any_satisfy ? 1 : 0;
        ballot_count[warp_id] = count;
    }
}

// Example 5: Using different tile sizes
template<int TileSize>
__global__ void tiled_reduction(int *output, const int *input, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<TileSize> tile = cg::tiled_partition<TileSize>(block);

    unsigned int tid = block.thread_rank();
    unsigned int i = block.group_index().x * block.size() + tid;

    int value = (i < n) ? input[i] : 0;

    // Reduction within tile
    for (int offset = tile.size() / 2; offset > 0; offset >>= 1) {
        value += tile.shfl_down(value, offset);
    }

    // First thread in each tile writes result
    __shared__ int tile_sums[256]; // Max 256 threads per block
    if (tile.thread_rank() == 0) {
        tile_sums[tid / TileSize] = value;
    }
    block.sync();

    // Final reduction (simple, not optimized)
    if (tid == 0) {
        int num_tiles = (block.size() + TileSize - 1) / TileSize;
        int sum = 0;
        for (int i = 0; i < num_tiles; i++) {
            sum += tile_sums[i];
        }
        output[block.group_index().x] = sum;
    }
}

// Example 6: Meta-group information
__global__ void print_group_info(int *output) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);
    cg::thread_block_tile<16> tile16 = cg::tiled_partition<16>(block);
    cg::thread_block_tile<8> tile8 = cg::tiled_partition<8>(block);

    unsigned int tid = block.thread_rank();

    // Only first thread of each group type prints
    if (tid == 0) {
        printf("Block info:\n");
        printf("  Size: %u threads\n", block.size());
        printf("  Dim: (%u, %u, %u)\n", block.group_dim().x,
               block.group_dim().y, block.group_dim().z);
    }

    if (tile32.thread_rank() == 0) {
        printf("  Tile32[%u]: size=%u, meta_group_rank=%u, meta_group_size=%u\n",
               tid / 32, tile32.size(), tile32.meta_group_rank(),
               tile32.meta_group_size());
    }

    if (tile16.thread_rank() == 0 && tid < 32) {
        printf("  Tile16[%u]: size=%u, meta_group_rank=%u, meta_group_size=%u\n",
               tid / 16, tile16.size(), tile16.meta_group_rank(),
               tile16.meta_group_size());
    }
}

void test_basic_operations() {
    printf("\n1. Basic Thread Block Operations:\n");

    const int n = 256;
    int *h_input = (int*)malloc(n * sizeof(int));
    int *h_output = (int*)malloc(sizeof(int));

    for (int i = 0; i < n; i++) {
        h_input[i] = 1;
    }

    int *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, n * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice));

    sum_reduction_thread_block<<<1, 256, 256 * sizeof(int)>>>(d_output, d_input, n);
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost));

    printf("   Sum of %d ones: %d (expected: %d) - %s\n",
           n, h_output[0], n, h_output[0] == n ? "PASS" : "FAIL");

    free(h_input);
    free(h_output);
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
}

void test_tiled_operations() {
    printf("\n2. Tiled Operations:\n");

    const int n = 256;
    int *h_input = (int*)malloc(n * sizeof(int));
    int *h_output = (int*)malloc(sizeof(int));

    for (int i = 0; i < n; i++) {
        h_input[i] = 1;
    }

    int *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, n * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice));

    sum_reduction_tiled<<<1, 256>>>(d_output, d_input, n);
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost));

    printf("   Tile-based sum: %d (expected: %d) - %s\n",
           h_output[0], n, h_output[0] == n ? "PASS" : "FAIL");

    free(h_input);
    free(h_output);
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
}

void test_vote_operations() {
    printf("\n3. Vote Operations:\n");

    const int n = 64; // 2 warps
    int *h_input = (int*)malloc(n * sizeof(int));

    // First warp: all above threshold
    // Second warp: half above, half below
    int threshold = 50;
    for (int i = 0; i < 32; i++) {
        h_input[i] = 60;  // All above
    }
    for (int i = 32; i < 48; i++) {
        h_input[i] = 60;  // Above
    }
    for (int i = 48; i < 64; i++) {
        h_input[i] = 40;  // Below
    }

    int *d_input, *d_all, *d_any, *d_count;
    int *h_all = (int*)malloc(2 * sizeof(int));
    int *h_any = (int*)malloc(2 * sizeof(int));
    int *h_count = (int*)malloc(2 * sizeof(int));

    CHECK_CUDA_ERROR(cudaMalloc(&d_input, n * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_all, 2 * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_any, 2 * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_count, 2 * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice));

    vote_operations_comparison<<<1, 64>>>(d_all, d_any, d_count, d_input, n, threshold);

    CHECK_CUDA_ERROR(cudaMemcpy(h_all, d_all, 2 * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_any, d_any, 2 * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_count, d_count, 2 * sizeof(int), cudaMemcpyDeviceToHost));

    printf("   Warp 0 (all above threshold):\n");
    printf("     all(): %d (expected: 1)\n", h_all[0]);
    printf("     any(): %d (expected: 1)\n", h_any[0]);
    printf("     ballot count: %d (expected: 32)\n", h_count[0]);

    printf("   Warp 1 (half above threshold):\n");
    printf("     all(): %d (expected: 0)\n", h_all[1]);
    printf("     any(): %d (expected: 1)\n", h_any[1]);
    printf("     ballot count: %d (expected: 16)\n", h_count[1]);

    free(h_input);
    free(h_all);
    free(h_any);
    free(h_count);
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_all));
    CHECK_CUDA_ERROR(cudaFree(d_any));
    CHECK_CUDA_ERROR(cudaFree(d_count));
}

void test_different_tile_sizes() {
    printf("\n4. Different Tile Sizes:\n");

    const int n = 256;
    int *h_input = (int*)malloc(n * sizeof(int));
    int *h_output = (int*)malloc(sizeof(int));

    for (int i = 0; i < n; i++) {
        h_input[i] = 1;
    }

    int *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, n * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice));

    // Test different tile sizes
    int sizes[] = {4, 8, 16, 32};
    for (int i = 0; i < 4; i++) {
        int tile_size = sizes[i];

        if (tile_size == 4) {
            tiled_reduction<4><<<1, 256>>>(d_output, d_input, n);
        } else if (tile_size == 8) {
            tiled_reduction<8><<<1, 256>>>(d_output, d_input, n);
        } else if (tile_size == 16) {
            tiled_reduction<16><<<1, 256>>>(d_output, d_input, n);
        } else {
            tiled_reduction<32><<<1, 256>>>(d_output, d_input, n);
        }

        CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost));
        printf("   Tile size %2d: sum=%d (expected: %d) - %s\n",
               tile_size, h_output[0], n, h_output[0] == n ? "PASS" : "FAIL");
    }

    free(h_input);
    free(h_output);
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
}

int main() {
    printf("========== Cooperative Groups Basics ==========\n");

    test_basic_operations();
    test_tiled_operations();
    test_vote_operations();
    test_different_tile_sizes();

    printf("\n5. Group Information:\n");
    int *d_dummy;
    CHECK_CUDA_ERROR(cudaMalloc(&d_dummy, sizeof(int)));
    print_group_info<<<1, 128>>>(d_dummy);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaFree(d_dummy));

    printf("\n========== All Tests Completed Successfully ==========\n");

    return 0;
}
