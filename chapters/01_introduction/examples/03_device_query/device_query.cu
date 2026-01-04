/**
 * @file device_query.cu
 * @brief Query and display CUDA device properties
 *
 * This program demonstrates how to programmatically query information
 * about available CUDA devices. Understanding your hardware capabilities
 * is essential for:
 * - Optimizing kernel configurations
 * - Understanding performance limitations
 * - Writing portable code across different GPUs
 * - Debugging hardware-specific issues
 *
 * Learning Objectives:
 * 1. Enumerate CUDA-capable devices
 * 2. Query detailed device properties
 * 3. Understand compute capability and its implications
 * 4. Learn hardware limits (threads, memory, blocks, etc.)
 * 5. Interpret specifications for optimization
 *
 * CUDA Programming Guide Reference:
 * - Section 3.2.6: Device Enumeration
 * - Appendix F: Compute Capabilities
 */

#include <stdio.h>
#include <cuda_runtime.h>

/**
 * @brief Convert bytes to human-readable format
 */
void printMemorySize(const char* label, size_t bytes) {
    const double GB = 1024.0 * 1024.0 * 1024.0;
    const double MB = 1024.0 * 1024.0;
    const double KB = 1024.0;

    if (bytes >= GB) {
        printf("  %s: %.2f GB\n", label, bytes / GB);
    } else if (bytes >= MB) {
        printf("  %s: %.2f MB\n", label, bytes / MB);
    } else if (bytes >= KB) {
        printf("  %s: %.2f KB\n", label, bytes / KB);
    } else {
        printf("  %s: %zu bytes\n", label, bytes);
    }
}

/**
 * @brief Get compute capability description
 */
const char* getComputeCapabilityName(int major, int minor) {
    int version = major * 10 + minor;

    switch (version) {
        case 30: case 32: case 35: case 37:
            return "Kepler";
        case 50: case 52: case 53:
            return "Maxwell";
        case 60: case 61: case 62:
            return "Pascal";
        case 70: case 72: case 75:
            return "Volta/Turing";
        case 80: case 86: case 87:
            return "Ampere";
        case 89:
            return "Ada Lovelace";
        case 90:
            return "Hopper";
        default:
            return "Unknown";
    }
}

/**
 * @brief Print detailed information about a CUDA device
 */
void printDeviceProperties(int device) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get properties for device %d: %s\n",
                device, cudaGetErrorString(err));
        return;
    }

    printf("\n");
    printf("========================================================================\n");
    printf("DEVICE %d: %s\n", device, prop.name);
    printf("========================================================================\n");

    // ========================================================================
    // Compute Capability
    // ========================================================================
    printf("\n--- Compute Capability ---\n");
    printf("  Compute Capability: %d.%d (%s architecture)\n",
           prop.major, prop.minor, getComputeCapabilityName(prop.major, prop.minor));
    printf("  CUDA Cores: %d SMs x %d cores/SM = ~%d cores (approx)\n",
           prop.multiProcessorCount,
           _ConvertSMVer2Cores(prop.major, prop.minor),
           prop.multiProcessorCount * _ConvertSMVer2Cores(prop.major, prop.minor));
    printf("  Streaming Multiprocessors (SMs): %d\n", prop.multiProcessorCount);

    // ========================================================================
    // Memory Information
    // ========================================================================
    printf("\n--- Memory ---\n");
    printMemorySize("Total Global Memory", prop.totalGlobalMem);
    printMemorySize("Shared Memory per Block", prop.sharedMemPerBlock);
    printMemorySize("Shared Memory per SM", prop.sharedMemPerMultiprocessor);
    printf("  32-bit Registers per Block: %d\n", prop.regsPerBlock);
    printf("  32-bit Registers per SM: %d\n", prop.regsPerMultiprocessor);
    printMemorySize("Constant Memory", prop.totalConstMem);
    printMemorySize("L2 Cache Size", prop.l2CacheSize);
    printf("  Memory Bus Width: %d-bit\n", prop.memoryBusWidth);
    printf("  Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
    printf("  Peak Memory Bandwidth: %.2f GB/s\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);

    // ========================================================================
    // Thread/Block/Grid Limits
    // ========================================================================
    printf("\n--- Thread Organization Limits ---\n");
    printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Max Block Dimensions: (%d, %d, %d)\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Max Grid Dimensions: (%d, %d, %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Warp Size: %d threads\n", prop.warpSize);
    printf("  Max Blocks per SM: %d (theoretical)\n",
           prop.maxThreadsPerMultiProcessor / prop.maxThreadsPerBlock);

    // Calculate theoretical occupancy
    int maxBlocksPerSM = prop.maxThreadsPerMultiProcessor / 256; // Assuming 256 threads/block
    printf("  Theoretical Occupancy (256 threads/block): %d blocks/SM\n", maxBlocksPerSM);

    // ========================================================================
    // Performance Characteristics
    // ========================================================================
    printf("\n--- Performance ---\n");
    printf("  Clock Rate: %.2f GHz\n", prop.clockRate / 1e6);
    printf("  Theoretical Peak Performance (FP32):\n");
    double peakGFLOPS = (prop.multiProcessorCount *
                         _ConvertSMVer2Cores(prop.major, prop.minor) *
                         (prop.clockRate / 1e6) * 2); // 2 ops per FMA
    printf("    %.2f GFLOPS (assuming FMA)\n", peakGFLOPS);

    if (prop.major >= 6) { // FP16 support started with Pascal
        printf("  Theoretical Peak Performance (FP16):\n");
        double peakFP16 = peakGFLOPS * 2; // 2x throughput for FP16
        printf("    %.2f GFLOPS\n", peakFP16);
    }

    // ========================================================================
    // Feature Support
    // ========================================================================
    printf("\n--- Capabilities and Features ---\n");
    printf("  Concurrent Kernels: %s (max: %d)\n",
           prop.concurrentKernels ? "Yes" : "No",
           prop.concurrentKernels);
    printf("  Concurrent Copy and Execution: %s\n",
           prop.deviceOverlap ? "Yes" : "No");
    printf("  ECC Enabled: %s\n", prop.ECCEnabled ? "Yes" : "No");
    printf("  Integrated GPU: %s\n", prop.integrated ? "Yes" : "No");
    printf("  Unified Addressing: %s\n", prop.unifiedAddressing ? "Yes" : "No");
    printf("  Managed Memory: %s\n", prop.managedMemory ? "Yes" : "No");
    printf("  Cooperative Launch: %s\n",
           prop.cooperativeLaunch ? "Yes" : "No");
    printf("  Multi-Device Cooperative Launch: %s\n",
           prop.cooperativeMultiDeviceLaunch ? "Yes" : "No");

    // ========================================================================
    // Asynchronous Capabilities
    // ========================================================================
    printf("\n--- Asynchronous Execution ---\n");
    printf("  Async Engine Count: %d\n", prop.asyncEngineCount);
    printf("  Number of Copy Engines: %d\n", prop.asyncEngineCount);
    printf("  Concurrent Memory Copies: ");
    if (prop.asyncEngineCount == 0) {
        printf("No\n");
    } else if (prop.asyncEngineCount == 1) {
        printf("One direction (H2D or D2H)\n");
    } else {
        printf("Bidirectional (H2D and D2H simultaneously)\n");
    }

    // ========================================================================
    // Texture and Surface
    // ========================================================================
    printf("\n--- Texture/Surface ---\n");
    printf("  Max 1D Texture Size: %d\n", prop.maxTexture1D);
    printf("  Max 2D Texture Size: (%d, %d)\n",
           prop.maxTexture2D[0], prop.maxTexture2D[1]);
    printf("  Max 3D Texture Size: (%d, %d, %d)\n",
           prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]);
    printf("  Texture Alignment: %zu bytes\n", prop.textureAlignment);

    // ========================================================================
    // PCI Information
    // ========================================================================
    printf("\n--- PCI Information ---\n");
    printf("  PCI Bus ID: %d\n", prop.pciBusID);
    printf("  PCI Device ID: %d\n", prop.pciDeviceID);
    printf("  PCI Domain ID: %d\n", prop.pciDomainID);

    // ========================================================================
    // Recommended Configurations
    // ========================================================================
    printf("\n--- Recommended Kernel Configurations ---\n");
    printf("  Block Size (threads): 128, 256, or 512 (multiples of %d)\n", prop.warpSize);
    printf("  Grid Size (blocks): At least %d (2x SMs) for good occupancy\n",
           prop.multiProcessorCount * 2);
    printf("  Shared Memory Usage: Keep under %.2f KB per block\n",
           prop.sharedMemPerBlock / 1024.0);

    // Calculate some example configurations
    printf("\n--- Example Valid Configurations ---\n");
    int blockSizes[] = {128, 256, 512};
    for (int i = 0; i < 3; i++) {
        int blockSize = blockSizes[i];
        if (blockSize <= prop.maxThreadsPerBlock) {
            int maxBlocksPerSM = prop.maxThreadsPerMultiProcessor / blockSize;
            int totalBlocks = maxBlocksPerSM * prop.multiProcessorCount;
            printf("  %d threads/block: Max %d blocks/SM, %d blocks total, %d total threads\n",
                   blockSize, maxBlocksPerSM, totalBlocks, totalBlocks * blockSize);
        }
    }

    printf("\n");
}

/**
 * @brief Helper function to convert SM version to cores per SM
 * Note: This is approximate as actual core count varies by architecture
 */
int _ConvertSMVer2Cores(int major, int minor) {
    // Approximate CUDA cores per SM
    switch ((major << 4) + minor) {
        case 0x30: // Kepler (GK10x)
        case 0x32: // Kepler (GK10x)
        case 0x35: // Kepler (GK11x)
        case 0x37: // Kepler (GK21x)
            return 192;
        case 0x50: // Maxwell
        case 0x52: // Maxwell
        case 0x53: // Maxwell
            return 128;
        case 0x60: // Pascal
        case 0x61: // Pascal
        case 0x62: // Pascal
            return 64;
        case 0x70: // Volta
        case 0x72: // Xavier
        case 0x75: // Turing
            return 64;
        case 0x80: // Ampere (GA100)
        case 0x86: // Ampere (GA10x)
        case 0x87: // Ampere (GA10x)
            return 128;
        case 0x89: // Ada Lovelace
            return 128;
        case 0x90: // Hopper
            return 128;
        default:
            return 64; // Default conservative estimate
    }
}

/**
 * @brief Print memory usage information
 */
void printMemoryUsage(int device) {
    cudaSetDevice(device);

    size_t free_mem, total_mem;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get memory info: %s\n", cudaGetErrorString(err));
        return;
    }

    size_t used_mem = total_mem - free_mem;
    double used_percent = (used_mem * 100.0) / total_mem;

    printf("\n--- Current Memory Usage ---\n");
    printMemorySize("Total Memory", total_mem);
    printMemorySize("Used Memory", used_mem);
    printMemorySize("Free Memory", free_mem);
    printf("  Usage: %.1f%%\n", used_percent);
}

/**
 * @brief Compare multiple devices
 */
void compareDevices(int deviceCount) {
    if (deviceCount < 2) {
        return;
    }

    printf("\n========================================================================\n");
    printf("DEVICE COMPARISON\n");
    printf("========================================================================\n\n");

    printf("%-4s %-30s %-8s %-12s %-12s %-8s\n",
           "ID", "Name", "CC", "Memory", "SMs", "Clock");
    printf("------------------------------------------------------------------------\n");

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("%-4d %-30s %d.%-6d %.2f GB     %-12d %.2f GHz\n",
               i,
               prop.name,
               prop.major, prop.minor,
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
               prop.multiProcessorCount,
               prop.clockRate / 1e6);
    }
    printf("\n");
}

int main() {
    printf("================================================================================\n");
    printf("CUDA DEVICE QUERY\n");
    printf("================================================================================\n");

    // ========================================================================
    // Check CUDA Driver and Runtime Versions
    // ========================================================================
    int driverVersion = 0, runtimeVersion = 0;

    cudaError_t err = cudaDriverGetVersion(&driverVersion);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get driver version: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    err = cudaRuntimeGetVersion(&runtimeVersion);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get runtime version: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("\n--- CUDA Version Information ---\n");
    printf("  CUDA Driver Version: %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);

    // ========================================================================
    // Enumerate Devices
    // ========================================================================
    int deviceCount = 0;
    err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("\n--- Device Count ---\n");
    printf("  Detected %d CUDA-capable device(s)\n", deviceCount);

    if (deviceCount == 0) {
        printf("\nNo CUDA-capable devices found!\n");
        printf("Possible reasons:\n");
        printf("  1. No NVIDIA GPU installed\n");
        printf("  2. GPU drivers not installed\n");
        printf("  3. GPU compute mode disabled\n");
        return EXIT_FAILURE;
    }

    // ========================================================================
    // Query Each Device
    // ========================================================================
    for (int device = 0; device < deviceCount; device++) {
        printDeviceProperties(device);
        printMemoryUsage(device);
    }

    // ========================================================================
    // Compare Devices (if multiple)
    // ========================================================================
    if (deviceCount > 1) {
        compareDevices(deviceCount);

        printf("--- Multi-GPU Configuration ---\n");
        printf("  Number of GPUs: %d\n", deviceCount);
        printf("  Peer-to-Peer Access:\n");

        for (int i = 0; i < deviceCount; i++) {
            for (int j = 0; j < deviceCount; j++) {
                if (i != j) {
                    int canAccessPeer;
                    cudaDeviceCanAccessPeer(&canAccessPeer, i, j);
                    printf("    GPU %d -> GPU %d: %s\n",
                           i, j, canAccessPeer ? "Yes" : "No");
                }
            }
        }
        printf("\n");
    }

    // ========================================================================
    // Summary and Recommendations
    // ========================================================================
    printf("========================================================================\n");
    printf("SUMMARY AND RECOMMENDATIONS\n");
    printf("========================================================================\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // Use first device

    printf("Primary Device: %s\n\n", prop.name);

    printf("Kernel Launch Guidelines:\n");
    printf("  1. Use block sizes: 128, 256, or 512 threads\n");
    printf("  2. Launch at least %d blocks for good occupancy\n",
           prop.multiProcessorCount * 2);
    printf("  3. Keep shared memory usage under %.0f KB per block\n",
           prop.sharedMemPerBlock / 1024.0);
    printf("  4. Total threads should be >> %d for full utilization\n",
           prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount);

    printf("\nMemory Optimization:\n");
    printf("  1. Total available memory: %.2f GB\n",
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  2. L2 cache: %.2f MB - optimize for cache reuse\n",
           prop.l2CacheSize / (1024.0 * 1024.0));
    printf("  3. Peak memory bandwidth: %.2f GB/s\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("  4. Use coalesced memory access patterns\n");

    printf("\nPortability:\n");
    printf("  1. Compile with -arch=sm_%d%d for this GPU\n", prop.major, prop.minor);
    printf("  2. For wider compatibility, use -arch=sm_50 or higher\n");
    printf("  3. Check compute capability before using advanced features\n");

    printf("\n");
    printf("================================================================================\n");

    return EXIT_SUCCESS;
}

/**
 * ============================================================================
 * COMPILATION AND EXECUTION
 * ============================================================================
 *
 * Compile:
 *   nvcc -o device_query device_query.cu
 *
 * Run:
 *   ./device_query
 *
 * Compare with nvidia-smi:
 *   nvidia-smi
 *
 * ============================================================================
 * KEY TAKEAWAYS
 * ============================================================================
 *
 * 1. Always query device properties before optimization
 * 2. Compute capability determines available features
 * 3. Hardware limits constrain kernel configurations
 * 4. Memory bandwidth often limits performance
 * 5. Different GPUs require different optimization strategies
 *
 * ============================================================================
 * IMPORTANT PROPERTIES
 * ============================================================================
 *
 * maxThreadsPerBlock: Maximum threads in a single block (typically 1024)
 * maxThreadsPerMultiProcessor: Max concurrent threads per SM (1024-2048)
 * multiProcessorCount: Number of SMs (determines parallelism)
 * sharedMemPerBlock: Shared memory available per block (48-164 KB)
 * totalGlobalMem: Total device memory (8-80 GB typical)
 * warpSize: Threads per warp (always 32)
 * major.minor: Compute capability (determines feature set)
 *
 * ============================================================================
 */
