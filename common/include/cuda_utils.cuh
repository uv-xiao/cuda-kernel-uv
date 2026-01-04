#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Error checking macro
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Check for errors after kernel launch
#define CUDA_CHECK_LAST()                                                      \
    do {                                                                       \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA kernel error at %s:%d: %s\n", __FILE__,      \
                    __LINE__, cudaGetErrorString(err));                        \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Synchronize and check for errors
#define CUDA_SYNC_CHECK()                                                      \
    do {                                                                       \
        cudaDeviceSynchronize();                                               \
        CUDA_CHECK_LAST();                                                     \
    } while (0)

// Ceiling division
template <typename T>
__host__ __device__ inline T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

// Print device properties
inline void print_device_info() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    printf("Device: %s\n", props.name);
    printf("  Compute capability: %d.%d\n", props.major, props.minor);
    printf("  Total global memory: %.2f GB\n",
           props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Shared memory per block: %zu KB\n",
           props.sharedMemPerBlock / 1024);
    printf("  Registers per block: %d\n", props.regsPerBlock);
    printf("  Warp size: %d\n", props.warpSize);
    printf("  Max threads per block: %d\n", props.maxThreadsPerBlock);
    printf("  Max threads dim: (%d, %d, %d)\n", props.maxThreadsDim[0],
           props.maxThreadsDim[1], props.maxThreadsDim[2]);
    printf("  Max grid size: (%d, %d, %d)\n", props.maxGridSize[0],
           props.maxGridSize[1], props.maxGridSize[2]);
    printf("  Number of SMs: %d\n", props.multiProcessorCount);
    printf("  Memory clock rate: %.2f GHz\n", props.memoryClockRate / 1e6);
    printf("  Memory bus width: %d bits\n", props.memoryBusWidth);
    printf("  Peak memory bandwidth: %.2f GB/s\n",
           2.0 * props.memoryClockRate * (props.memoryBusWidth / 8) / 1e6);
    printf("\n");
}

// Get number of SMs
inline int get_num_sms() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    return props.multiProcessorCount;
}

// Calculate theoretical occupancy
inline void print_occupancy(const void *kernel, int block_size,
                            size_t dynamic_smem = 0) {
    int max_active_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, kernel,
                                                   block_size, dynamic_smem);
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    int max_threads = props.maxThreadsPerMultiProcessor;
    float occupancy =
        (float)(max_active_blocks * block_size) / max_threads * 100.0f;

    printf("Occupancy: %.1f%% (%d blocks of %d threads per SM)\n", occupancy,
           max_active_blocks, block_size);
}
