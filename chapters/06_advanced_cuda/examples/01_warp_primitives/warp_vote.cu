#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

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

// Demonstrate __all_sync - check if predicate is true for all threads
__global__ void vote_all_kernel(int *results, int *input, int threshold) {
    int tid = threadIdx.x;
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;

    int value = input[tid];

    // Check if ALL threads in the warp have value >= threshold
    int all_above = __all_sync(0xffffffff, value >= threshold);

    // Only lane 0 of each warp writes the result
    if (lane_id == 0) {
        results[warp_id] = all_above;
    }
}

// Demonstrate __any_sync - check if predicate is true for any thread
__global__ void vote_any_kernel(int *results, int *input, int threshold) {
    int tid = threadIdx.x;
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;

    int value = input[tid];

    // Check if ANY thread in the warp has value >= threshold
    int any_above = __any_sync(0xffffffff, value >= threshold);

    // Only lane 0 of each warp writes the result
    if (lane_id == 0) {
        results[warp_id] = any_above;
    }
}

// Demonstrate __ballot_sync - get bitmask of predicate results
__global__ void vote_ballot_kernel(unsigned int *results, int *input, int threshold) {
    int tid = threadIdx.x;
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;

    int value = input[tid];

    // Get bitmask: bit i is set if thread i has value >= threshold
    unsigned int ballot = __ballot_sync(0xffffffff, value >= threshold);

    // Only lane 0 of each warp writes the result
    if (lane_id == 0) {
        results[warp_id] = ballot;
    }
}

// Practical example: Early termination in convergence check
__global__ void convergence_check_kernel(int *converged, float *values,
                                         float *old_values, float epsilon) {
    int tid = threadIdx.x;
    int lane_id = tid % warpSize;

    float diff = fabsf(values[tid] - old_values[tid]);

    // Check if all threads in the warp have converged
    int warp_converged = __all_sync(0xffffffff, diff < epsilon);

    // Use ballot to count how many threads have converged
    unsigned int ballot = __ballot_sync(0xffffffff, diff < epsilon);
    int converged_count = __popc(ballot); // Population count (number of 1s)

    if (lane_id == 0) {
        // All threads in warp converged
        converged[tid / warpSize] = warp_converged;
    }
}

// Practical example: Conflict detection in hash table
__global__ void conflict_detection_kernel(int *has_conflict, int *hash_values) {
    int tid = threadIdx.x;
    int lane_id = tid % warpSize;

    int my_hash = hash_values[tid];

    // Check if any other thread has the same hash (simple conflict detection)
    int conflict = 0;
    for (int i = 0; i < warpSize; i++) {
        int other_hash = __shfl_sync(0xffffffff, my_hash, i);
        if (i != lane_id && other_hash == my_hash) {
            conflict = 1;
            break;
        }
    }

    // Check if ANY thread detected a conflict
    int warp_has_conflict = __any_sync(0xffffffff, conflict);

    if (lane_id == 0) {
        has_conflict[tid / warpSize] = warp_has_conflict;
    }
}

// Practical example: Stream compaction (count matching elements)
__global__ void stream_compact_count_kernel(int *count, int *input, int target) {
    int tid = threadIdx.x;
    int lane_id = tid % warpSize;

    int value = input[tid];
    int matches = (value == target);

    // Get bitmask of matches
    unsigned int ballot = __ballot_sync(0xffffffff, matches);

    // Count number of matches using popcount
    int match_count = __popc(ballot);

    if (lane_id == 0) {
        count[tid / warpSize] = match_count;
    }
}

// Practical example: Dynamic load balancing decision
__global__ void load_balance_kernel(int *use_fast_path, int *work_sizes,
                                    int work_threshold) {
    int tid = threadIdx.x;
    int lane_id = tid % warpSize;

    int my_work = work_sizes[tid];

    // If all threads have small work, use fast path
    int all_small = __all_sync(0xffffffff, my_work < work_threshold);

    // If any thread has large work, use robust path
    int any_large = __any_sync(0xffffffff, my_work >= work_threshold);

    if (lane_id == 0) {
        use_fast_path[tid / warpSize] = all_small;
    }
}

void print_binary(unsigned int val) {
    for (int i = 31; i >= 0; i--) {
        printf("%d", (val >> i) & 1);
        if (i % 8 == 0 && i > 0) printf(" ");
    }
}

int main() {
    const int warp_count = 2;
    const int size = warpSize * warp_count;

    // Allocate host memory
    int *h_input = (int*)malloc(size * sizeof(int));
    int *h_results = (int*)malloc(warp_count * sizeof(int));
    unsigned int *h_ballot = (unsigned int*)malloc(warp_count * sizeof(unsigned int));

    // Allocate device memory
    int *d_input, *d_results;
    unsigned int *d_ballot;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, size * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_results, warp_count * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_ballot, warp_count * sizeof(unsigned int)));

    printf("========== CUDA Warp Vote Operations Demo ==========\n\n");
    printf("Testing with %d threads (%d warps)\n\n", size, warp_count);

    // Test 1: __all_sync with all values above threshold
    printf("1. __all_sync - All threads above threshold:\n");
    for (int i = 0; i < size; i++) {
        h_input[i] = 10; // All values are 10
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size * sizeof(int),
                                 cudaMemcpyHostToDevice));
    vote_all_kernel<<<1, size>>>(d_results, d_input, 5);
    CHECK_CUDA_ERROR(cudaMemcpy(h_results, d_results, warp_count * sizeof(int),
                                 cudaMemcpyDeviceToHost));
    printf("   Threshold: 5, All values: 10\n");
    for (int i = 0; i < warp_count; i++) {
        printf("   Warp %d: __all_sync = %d (expected: 1)\n", i, h_results[i]);
    }
    printf("\n");

    // Test 2: __all_sync with one value below threshold
    printf("2. __all_sync - One thread below threshold:\n");
    for (int i = 0; i < size; i++) {
        h_input[i] = 10;
    }
    h_input[5] = 3; // One value below threshold in warp 0
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size * sizeof(int),
                                 cudaMemcpyHostToDevice));
    vote_all_kernel<<<1, size>>>(d_results, d_input, 5);
    CHECK_CUDA_ERROR(cudaMemcpy(h_results, d_results, warp_count * sizeof(int),
                                 cudaMemcpyDeviceToHost));
    printf("   Threshold: 5, One value in warp 0 is 3\n");
    for (int i = 0; i < warp_count; i++) {
        printf("   Warp %d: __all_sync = %d (expected: %d)\n",
               i, h_results[i], (i == 0) ? 0 : 1);
    }
    printf("\n");

    // Test 3: __any_sync with no values above threshold
    printf("3. __any_sync - No threads above threshold:\n");
    for (int i = 0; i < size; i++) {
        h_input[i] = 3; // All values below threshold
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size * sizeof(int),
                                 cudaMemcpyHostToDevice));
    vote_any_kernel<<<1, size>>>(d_results, d_input, 5);
    CHECK_CUDA_ERROR(cudaMemcpy(h_results, d_results, warp_count * sizeof(int),
                                 cudaMemcpyDeviceToHost));
    printf("   Threshold: 5, All values: 3\n");
    for (int i = 0; i < warp_count; i++) {
        printf("   Warp %d: __any_sync = %d (expected: 0)\n", i, h_results[i]);
    }
    printf("\n");

    // Test 4: __any_sync with one value above threshold
    printf("4. __any_sync - One thread above threshold:\n");
    for (int i = 0; i < size; i++) {
        h_input[i] = 3;
    }
    h_input[warpSize + 10] = 10; // One value above threshold in warp 1
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size * sizeof(int),
                                 cudaMemcpyHostToDevice));
    vote_any_kernel<<<1, size>>>(d_results, d_input, 5);
    CHECK_CUDA_ERROR(cudaMemcpy(h_results, d_results, warp_count * sizeof(int),
                                 cudaMemcpyDeviceToHost));
    printf("   Threshold: 5, One value in warp 1 is 10\n");
    for (int i = 0; i < warp_count; i++) {
        printf("   Warp %d: __any_sync = %d (expected: %d)\n",
               i, h_results[i], (i == 1) ? 1 : 0);
    }
    printf("\n");

    // Test 5: __ballot_sync with alternating pattern
    printf("5. __ballot_sync - Alternating pattern:\n");
    for (int i = 0; i < size; i++) {
        h_input[i] = (i % 2 == 0) ? 10 : 3; // Even lanes above, odd below
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size * sizeof(int),
                                 cudaMemcpyHostToDevice));
    vote_ballot_kernel<<<1, size>>>(d_ballot, d_input, 5);
    CHECK_CUDA_ERROR(cudaMemcpy(h_ballot, d_ballot, warp_count * sizeof(unsigned int),
                                 cudaMemcpyDeviceToHost));
    printf("   Threshold: 5, Even lanes: 10, Odd lanes: 3\n");
    for (int i = 0; i < warp_count; i++) {
        printf("   Warp %d ballot: ", i);
        print_binary(h_ballot[i]);
        printf(" (0x%08x)\n", h_ballot[i]);
        printf("           Count: %d threads above threshold\n", __builtin_popcount(h_ballot[i]));
    }
    printf("   Note: Bit i=1 means lane i has value >= threshold\n\n");

    // Test 6: __ballot_sync for stream compaction
    printf("6. __ballot_sync - Stream compaction (count target value):\n");
    int target = 42;
    for (int i = 0; i < size; i++) {
        h_input[i] = (i % 5 == 0) ? target : i; // Every 5th element is target
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size * sizeof(int),
                                 cudaMemcpyHostToDevice));
    stream_compact_count_kernel<<<1, size>>>(d_results, d_input, target);
    CHECK_CUDA_ERROR(cudaMemcpy(h_results, d_results, warp_count * sizeof(int),
                                 cudaMemcpyDeviceToHost));
    printf("   Target value: %d (appears at indices 0, 5, 10, 15, ...)\n", target);
    for (int i = 0; i < warp_count; i++) {
        printf("   Warp %d: %d matches found\n", i, h_results[i]);
    }
    printf("\n");

    // Test 7: Conflict detection
    printf("7. Conflict detection in hash values:\n");
    int *d_conflict;
    int *h_conflict = (int*)malloc(warp_count * sizeof(int));
    CHECK_CUDA_ERROR(cudaMalloc(&d_conflict, warp_count * sizeof(int)));

    // No conflicts
    for (int i = 0; i < size; i++) {
        h_input[i] = i; // All unique
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size * sizeof(int),
                                 cudaMemcpyHostToDevice));
    conflict_detection_kernel<<<1, size>>>(d_conflict, d_input);
    CHECK_CUDA_ERROR(cudaMemcpy(h_conflict, d_conflict, warp_count * sizeof(int),
                                 cudaMemcpyDeviceToHost));
    printf("   All unique values:\n");
    for (int i = 0; i < warp_count; i++) {
        printf("   Warp %d: conflict = %d (expected: 0)\n", i, h_conflict[i]);
    }

    // With conflicts
    for (int i = 0; i < size; i++) {
        h_input[i] = i / 2; // Pairs have same value
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size * sizeof(int),
                                 cudaMemcpyHostToDevice));
    conflict_detection_kernel<<<1, size>>>(d_conflict, d_input);
    CHECK_CUDA_ERROR(cudaMemcpy(h_conflict, d_conflict, warp_count * sizeof(int),
                                 cudaMemcpyDeviceToHost));
    printf("   Pairs share same value:\n");
    for (int i = 0; i < warp_count; i++) {
        printf("   Warp %d: conflict = %d (expected: 1)\n", i, h_conflict[i]);
    }

    // Cleanup
    free(h_input);
    free(h_results);
    free(h_ballot);
    free(h_conflict);
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_results));
    CHECK_CUDA_ERROR(cudaFree(d_ballot));
    CHECK_CUDA_ERROR(cudaFree(d_conflict));

    printf("\n========== All Vote Operations Completed Successfully ==========\n");

    return 0;
}
