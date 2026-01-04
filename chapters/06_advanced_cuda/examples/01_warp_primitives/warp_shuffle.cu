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

// Demonstrate __shfl_sync - broadcast from a specific lane
__global__ void shuffle_broadcast_kernel(int *output, int value_to_broadcast) {
    int tid = threadIdx.x;
    int lane_id = tid % warpSize;

    // Each thread starts with its lane ID
    int my_value = lane_id;

    // All threads in the warp read from lane 0
    // mask 0xffffffff means all 32 threads participate
    int broadcast_value = __shfl_sync(0xffffffff, my_value, 0);

    output[tid] = broadcast_value;
}

// Demonstrate __shfl_up_sync - shift data down (read from lower lane)
__global__ void shuffle_up_kernel(int *output, int *input, int delta) {
    int tid = threadIdx.x;
    int lane_id = tid % warpSize;

    int my_value = input[tid];

    // Read from lane (lane_id - delta)
    // Threads with lane_id < delta will get their own value (no change)
    int shifted_value = __shfl_up_sync(0xffffffff, my_value, delta);

    output[tid] = shifted_value;
}

// Demonstrate __shfl_down_sync - shift data up (read from higher lane)
__global__ void shuffle_down_kernel(int *output, int *input, int delta) {
    int tid = threadIdx.x;
    int lane_id = tid % warpSize;

    int my_value = input[tid];

    // Read from lane (lane_id + delta)
    // Threads with lane_id >= (32 - delta) will get their own value
    int shifted_value = __shfl_down_sync(0xffffffff, my_value, delta);

    output[tid] = shifted_value;
}

// Demonstrate __shfl_xor_sync - butterfly exchange pattern
__global__ void shuffle_xor_kernel(int *output, int *input, int lane_mask) {
    int tid = threadIdx.x;
    int lane_id = tid % warpSize;

    int my_value = input[tid];

    // Read from lane (lane_id XOR lane_mask)
    // This creates a butterfly pattern useful for reductions
    int xor_value = __shfl_xor_sync(0xffffffff, my_value, lane_mask);

    output[tid] = xor_value;
}

// Practical example: Parallel prefix sum within a warp using shuffle
__global__ void warp_prefix_sum_kernel(int *output, int *input) {
    int tid = threadIdx.x;
    int lane_id = tid % warpSize;

    int value = input[tid];

    // Perform log2(32) = 5 iterations
    // Each iteration doubles the distance
    for (int offset = 1; offset < warpSize; offset *= 2) {
        int neighbor = __shfl_up_sync(0xffffffff, value, offset);
        if (lane_id >= offset) {
            value += neighbor;
        }
    }

    output[tid] = value;
}

// Practical example: Reverse array within warp using shuffle
__global__ void warp_reverse_kernel(int *output, int *input) {
    int tid = threadIdx.x;
    int lane_id = tid % warpSize;

    int my_value = input[tid];

    // Read from lane (31 - lane_id) to reverse
    int reversed_value = __shfl_sync(0xffffffff, my_value, warpSize - 1 - lane_id);

    output[tid] = reversed_value;
}

// Practical example: Warp-level sum reduction
__global__ void warp_sum_reduction_kernel(int *output, int *input) {
    int tid = threadIdx.x;

    int value = input[tid];

    // Perform tree-based reduction using XOR shuffle
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(0xffffffff, value, offset);
    }

    // Lane 0 has the sum for the entire warp
    if ((tid % warpSize) == 0) {
        output[tid / warpSize] = value;
    }
}

void print_array(const char* name, int *arr, int size) {
    printf("%s: ", name);
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main() {
    const int warp_count = 2;
    const int size = warpSize * warp_count;

    // Allocate host memory
    int *h_input = (int*)malloc(size * sizeof(int));
    int *h_output = (int*)malloc(size * sizeof(int));

    // Initialize input with sequential values
    for (int i = 0; i < size; i++) {
        h_input[i] = i;
    }

    // Allocate device memory
    int *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, size * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, size * sizeof(int)));

    printf("========== CUDA Warp Shuffle Operations Demo ==========\n\n");
    printf("Testing with %d threads (%d warps of %d threads each)\n\n",
           size, warp_count, warpSize);

    // Test 1: Broadcast
    printf("1. __shfl_sync (Broadcast from lane 0):\n");
    shuffle_broadcast_kernel<<<1, size>>>(d_output, 42);
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size * sizeof(int),
                                 cudaMemcpyDeviceToHost));
    print_array("   Input ", h_input, warpSize);
    print_array("   Output", h_output, warpSize);
    printf("   All threads should have value 0 (broadcasted from lane 0)\n\n");

    // Test 2: Shuffle up (delta = 1)
    printf("2. __shfl_up_sync (delta = 1):\n");
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size * sizeof(int),
                                 cudaMemcpyHostToDevice));
    shuffle_up_kernel<<<1, size>>>(d_output, d_input, 1);
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size * sizeof(int),
                                 cudaMemcpyDeviceToHost));
    print_array("   Input ", h_input, warpSize);
    print_array("   Output", h_output, warpSize);
    printf("   Each thread reads from previous lane (lane 0 unchanged)\n\n");

    // Test 3: Shuffle down (delta = 2)
    printf("3. __shfl_down_sync (delta = 2):\n");
    shuffle_down_kernel<<<1, size>>>(d_output, d_input, 2);
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size * sizeof(int),
                                 cudaMemcpyDeviceToHost));
    print_array("   Input ", h_input, warpSize);
    print_array("   Output", h_output, warpSize);
    printf("   Each thread reads from lane+2 (last 2 lanes unchanged)\n\n");

    // Test 4: Shuffle XOR (mask = 1)
    printf("4. __shfl_xor_sync (mask = 1):\n");
    shuffle_xor_kernel<<<1, size>>>(d_output, d_input, 1);
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size * sizeof(int),
                                 cudaMemcpyDeviceToHost));
    print_array("   Input ", h_input, warpSize);
    print_array("   Output", h_output, warpSize);
    printf("   Pairs swap: (0↔1), (2↔3), (4↔5), etc.\n\n");

    // Test 5: Shuffle XOR (mask = 16)
    printf("5. __shfl_xor_sync (mask = 16):\n");
    shuffle_xor_kernel<<<1, size>>>(d_output, d_input, 16);
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size * sizeof(int),
                                 cudaMemcpyDeviceToHost));
    print_array("   Input ", h_input, warpSize);
    print_array("   Output", h_output, warpSize);
    printf("   Halves swap: lanes [0-15] ↔ lanes [16-31]\n\n");

    // Test 6: Warp prefix sum
    printf("6. Warp Prefix Sum (Inclusive Scan):\n");
    for (int i = 0; i < size; i++) {
        h_input[i] = 1; // All ones for easy verification
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size * sizeof(int),
                                 cudaMemcpyHostToDevice));
    warp_prefix_sum_kernel<<<1, size>>>(d_output, d_input);
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size * sizeof(int),
                                 cudaMemcpyDeviceToHost));
    print_array("   Input ", h_input, warpSize);
    print_array("   Output", h_output, warpSize);
    printf("   Cumulative sum: each position = count of 1s up to that point\n\n");

    // Test 7: Warp reverse
    printf("7. Warp Reverse:\n");
    for (int i = 0; i < size; i++) {
        h_input[i] = i;
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size * sizeof(int),
                                 cudaMemcpyHostToDevice));
    warp_reverse_kernel<<<1, size>>>(d_output, d_input);
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size * sizeof(int),
                                 cudaMemcpyDeviceToHost));
    print_array("   Input ", h_input, warpSize);
    print_array("   Output", h_output, warpSize);
    printf("   Array reversed within each warp\n\n");

    // Test 8: Warp reduction (sum)
    printf("8. Warp Sum Reduction:\n");
    for (int i = 0; i < size; i++) {
        h_input[i] = i;
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size * sizeof(int),
                                 cudaMemcpyHostToDevice));
    warp_sum_reduction_kernel<<<1, size>>>(d_output, d_input);
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, warp_count * sizeof(int),
                                 cudaMemcpyDeviceToHost));
    print_array("   Input (warp 0)", h_input, warpSize);
    printf("   Sum of warp 0: %d (expected: %d)\n", h_output[0],
           (warpSize - 1) * warpSize / 2);
    print_array("   Input (warp 1)", h_input + warpSize, warpSize);
    printf("   Sum of warp 1: %d (expected: %d)\n", h_output[1],
           (warpSize + (size - 1)) * warpSize / 2);

    // Cleanup
    free(h_input);
    free(h_output);
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));

    printf("\n========== All Shuffle Operations Completed Successfully ==========\n");

    return 0;
}
