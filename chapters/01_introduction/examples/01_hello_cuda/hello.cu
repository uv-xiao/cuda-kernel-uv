/**
 * @file hello.cu
 * @brief First CUDA kernel - Hello from GPU threads
 *
 * This example demonstrates the fundamental concepts of CUDA programming:
 * - Writing a simple __global__ kernel function
 * - Launching kernels with execution configuration <<<grid, block>>>
 * - Understanding thread and block indexing
 * - Basic thread hierarchy (threads, blocks, grids)
 *
 * Learning Objectives:
 * 1. Understand the __global__ function qualifier
 * 2. Learn how to launch CUDA kernels
 * 3. Use threadIdx and blockIdx built-in variables
 * 4. Explore the relationship between threads, blocks, and grids
 *
 * CUDA Programming Guide Reference:
 * - Section 3.2.2: Thread Hierarchy
 * - Appendix B.1: Function Execution Space Specifiers
 */

#include <stdio.h>
#include <cuda_runtime.h>

/**
 * @brief Simple CUDA kernel that prints thread and block information
 *
 * This kernel demonstrates:
 * - The __global__ qualifier (callable from host, runs on device)
 * - Built-in variables: threadIdx, blockIdx, blockDim, gridDim
 * - How each thread gets a unique combination of indices
 *
 * Each thread in the grid will execute this function exactly once.
 * The thread can identify itself using threadIdx and blockIdx.
 *
 * Built-in Variables:
 * - threadIdx.{x,y,z}: Thread's index within its block (0 to blockDim-1)
 * - blockIdx.{x,y,z}: Block's index within the grid (0 to gridDim-1)
 * - blockDim.{x,y,z}: Number of threads per block (specified at launch)
 * - gridDim.{x,y,z}: Number of blocks in the grid (specified at launch)
 *
 * Note: printf from device code is supported but has limitations:
 * - Output may be buffered and appear out of order
 * - Large amounts of printf can impact performance
 * - Buffer size is limited (configurable with cudaDeviceSetLimit)
 */
__global__ void helloKernel() {
    // Calculate the global thread ID in 1D grid
    // This is the standard pattern for finding a thread's unique index
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread prints its identifying information
    // Note: Output order is not guaranteed due to parallel execution
    printf("Hello from thread %d in block %d (global ID: %d)\n",
           threadIdx.x,    // Thread's local index within the block
           blockIdx.x,     // Block's index within the grid
           globalThreadId); // Thread's unique index across entire grid
}

/**
 * @brief Demonstrates multi-dimensional indexing
 *
 * CUDA supports 1D, 2D, and 3D organizations for both blocks and threads.
 * This is useful for naturally representing multi-dimensional data (images, volumes).
 *
 * For a 2D grid of 2D blocks:
 * - threadIdx.{x,y} gives position within the block
 * - blockIdx.{x,y} gives block position within the grid
 */
__global__ void hello2DKernel() {
    // Calculate 2D global thread indices
    int globalThreadX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalThreadY = blockIdx.y * blockDim.y + threadIdx.y;

    printf("Hello from thread (%d,%d) in block (%d,%d) - global position: (%d,%d)\n",
           threadIdx.x, threadIdx.y,      // Thread position in block
           blockIdx.x, blockIdx.y,        // Block position in grid
           globalThreadX, globalThreadY); // Global thread position
}

/**
 * @brief Helper function to check CUDA errors
 *
 * Best Practice: Always check CUDA API return values!
 * Many CUDA functions return cudaError_t to indicate success/failure.
 *
 * @param err CUDA error code to check
 * @param msg Context message to print if error occurred
 */
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    printf("=== CUDA Hello World Example ===\n\n");

    // ========================================================================
    // Example 1: Simple 1D kernel launch
    // ========================================================================
    printf("--- Example 1: 1D Grid of 1D Blocks ---\n");

    // Define execution configuration
    // Syntax: kernelName<<<gridDim, blockDim>>>(args...)
    // gridDim: Number of blocks in the grid
    // blockDim: Number of threads per block

    int numBlocks = 2;           // Launch 2 blocks
    int threadsPerBlock = 4;     // Each block contains 4 threads

    printf("Launching kernel with %d blocks, %d threads per block\n",
           numBlocks, threadsPerBlock);
    printf("Total threads: %d\n\n", numBlocks * threadsPerBlock);

    // Launch the kernel
    // <<<...>>> is CUDA's execution configuration syntax
    // The kernel launch is asynchronous - control returns to CPU immediately
    helloKernel<<<numBlocks, threadsPerBlock>>>();

    // Check for kernel launch errors
    // cudaGetLastError() retrieves the last error from a runtime call
    checkCudaError(cudaGetLastError(), "Kernel launch");

    // Wait for all device operations to complete
    // Kernel launches are asynchronous, so we need to synchronize
    // to ensure all printf output is flushed before continuing
    checkCudaError(cudaDeviceSynchronize(), "Device synchronization");

    printf("\n");

    // ========================================================================
    // Example 2: Exploring different configurations
    // ========================================================================
    printf("--- Example 2: Different Thread Organization ---\n");

    // Try different configurations that give the same total thread count
    // This demonstrates flexibility in organizing parallelism

    printf("Config A: 4 blocks x 2 threads = 8 total threads\n");
    helloKernel<<<4, 2>>>();
    checkCudaError(cudaDeviceSynchronize(), "Device sync - Config A");

    printf("\nConfig B: 1 block x 8 threads = 8 total threads\n");
    helloKernel<<<1, 8>>>();
    checkCudaError(cudaDeviceSynchronize(), "Device sync - Config B");

    printf("\n");

    // ========================================================================
    // Example 3: 2D kernel launch
    // ========================================================================
    printf("--- Example 3: 2D Grid of 2D Blocks ---\n");

    // Define 2D dimensions using dim3 structure
    // dim3 is a CUDA built-in type for specifying dimensions
    // Unspecified dimensions default to 1

    dim3 gridDim(2, 2);    // 2x2 grid of blocks (4 blocks total)
    dim3 blockDim(2, 2);   // 2x2 threads per block (4 threads per block)

    printf("Launching 2D kernel:\n");
    printf("  Grid: %d x %d blocks\n", gridDim.x, gridDim.y);
    printf("  Block: %d x %d threads\n", blockDim.x, blockDim.y);
    printf("  Total threads: %d\n\n",
           gridDim.x * gridDim.y * blockDim.x * blockDim.y);

    hello2DKernel<<<gridDim, blockDim>>>();
    checkCudaError(cudaGetLastError(), "2D Kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "Device sync - 2D kernel");

    printf("\n");

    // ========================================================================
    // Example 4: Understanding thread/block limits
    // ========================================================================
    printf("--- Example 4: Hardware Limitations ---\n");

    // Query device properties to understand hardware constraints
    cudaDeviceProp prop;
    checkCudaError(cudaGetDeviceProperties(&prop, 0), "Get device properties");

    printf("Device: %s\n", prop.name);
    printf("Maximum threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Maximum block dimensions: (%d, %d, %d)\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Maximum grid dimensions: (%d, %d, %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    // Best Practice: Use multiples of warp size (32) for block dimensions
    printf("\nRecommended block sizes (multiples of warp size 32):\n");
    printf("  - 128 threads (4 warps)\n");
    printf("  - 256 threads (8 warps) [common choice]\n");
    printf("  - 512 threads (16 warps)\n");

    printf("\n=== Explanation of Output ===\n");
    printf("- Thread output may appear in any order (parallel execution)\n");
    printf("- Global ID = blockIdx.x * blockDim.x + threadIdx.x\n");
    printf("- Different configurations can have different performance\n");
    printf("- Block size affects occupancy and resource usage\n");

    printf("\n=== Key Takeaways ===\n");
    printf("1. __global__ functions are kernels that run on the GPU\n");
    printf("2. Kernels are launched with <<<grid, block>>> syntax\n");
    printf("3. Each thread has unique threadIdx and blockIdx\n");
    printf("4. Total threads = gridDim * blockDim (for each dimension)\n");
    printf("5. Kernel launches are asynchronous by default\n");
    printf("6. Use cudaDeviceSynchronize() to wait for completion\n");

    return 0;
}

/**
 * ============================================================================
 * COMPILATION AND EXECUTION
 * ============================================================================
 *
 * Compile with nvcc (NVIDIA CUDA Compiler):
 *   nvcc -o hello hello.cu
 *
 * Or with CMake:
 *   mkdir build && cd build
 *   cmake ..
 *   make
 *
 * Run:
 *   ./hello
 *
 * Expected behavior:
 * - Multiple lines of output from different threads
 * - Output may appear in non-sequential order
 * - Each thread prints its unique identifying information
 *
 * ============================================================================
 * EXERCISES TO TRY
 * ============================================================================
 *
 * 1. Modify the number of blocks and threads - observe the output
 * 2. Try launching with 1024 threads per block (the typical maximum)
 * 3. Create a 3D kernel using dim3(x, y, z) for both grid and block
 * 4. Add threadIdx.z and blockIdx.z to your printf statements
 * 5. Calculate total number of threads for different configurations
 * 6. What happens if you launch with 0 blocks or 0 threads?
 * 7. Remove cudaDeviceSynchronize() - does output still appear?
 *
 * ============================================================================
 * COMMON ERRORS AND DEBUGGING
 * ============================================================================
 *
 * Error: "invalid configuration argument"
 * - Cause: Block size exceeds hardware limits
 * - Solution: Reduce threads per block (max is typically 1024)
 *
 * Error: No output appears
 * - Cause: Forgot to synchronize before program exit
 * - Solution: Add cudaDeviceSynchronize() after kernel launch
 *
 * Error: "no kernel image is available for execution"
 * - Cause: GPU compute capability mismatch
 * - Solution: Compile with appropriate -arch flag (e.g., -arch=sm_75)
 *
 * ============================================================================
 */
