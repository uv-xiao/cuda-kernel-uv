# Example 03: Device Query

## Overview

This program queries and displays detailed information about your CUDA-capable GPUs. Understanding your hardware is crucial for writing efficient, portable CUDA code. This tool helps you discover:

- What GPU(s) you have
- Their capabilities and limitations
- Optimal configuration parameters
- Memory specifications and bandwidth
- Supported features

Think of this as your GPU's "spec sheet" - essential information for optimization and debugging.

## Learning Objectives

1. **Device Enumeration**
   - Count available CUDA devices
   - Query CUDA driver and runtime versions
   - Select and switch between devices

2. **Hardware Specifications**
   - Understand compute capability
   - Interpret memory hierarchy
   - Learn thread organization limits
   - Discover performance characteristics

3. **Optimization Guidance**
   - Determine optimal block sizes
   - Calculate theoretical performance
   - Understand resource constraints
   - Identify architectural features

4. **Portable Programming**
   - Write code that adapts to different GPUs
   - Check feature availability at runtime
   - Compile for appropriate architectures

## Building and Running

### Using CMake

```bash
cd /home/uvxiao/cuda-kernel-tutorial/chapters/01_introduction/examples/03_device_query
mkdir build && cd build
cmake ..
make
./device_query
```

### Using nvcc

```bash
nvcc -o device_query device_query.cu
./device_query
```

## Sample Output

```
================================================================================
CUDA DEVICE QUERY
================================================================================

--- CUDA Version Information ---
  CUDA Driver Version: 12.2
  CUDA Runtime Version: 12.2

--- Device Count ---
  Detected 1 CUDA-capable device(s)

========================================================================
DEVICE 0: NVIDIA GeForce RTX 3080
========================================================================

--- Compute Capability ---
  Compute Capability: 8.6 (Ampere architecture)
  CUDA Cores: 68 SMs x 128 cores/SM = ~8704 cores (approx)
  Streaming Multiprocessors (SMs): 68

--- Memory ---
  Total Global Memory: 10.00 GB
  Shared Memory per Block: 48.00 KB
  Shared Memory per SM: 100.00 KB
  32-bit Registers per Block: 65536
  32-bit Registers per SM: 65536
  Constant Memory: 64.00 KB
  L2 Cache Size: 5.00 MB
  Memory Bus Width: 320-bit
  Memory Clock Rate: 9.50 GHz
  Peak Memory Bandwidth: 760.00 GB/s

--- Thread Organization Limits ---
  Max Threads per Block: 1024
  Max Threads per SM: 1536
  Max Block Dimensions: (1024, 1024, 64)
  Max Grid Dimensions: (2147483647, 65535, 65535)
  Warp Size: 32 threads
  Max Blocks per SM: 6 (theoretical)

--- Performance ---
  Clock Rate: 1.71 GHz
  Theoretical Peak Performance (FP32):
    29.78 TFLOPS (assuming FMA)
  Theoretical Peak Performance (FP16):
    59.56 TFLOPS

--- Capabilities and Features ---
  Concurrent Kernels: Yes (max: 128)
  Concurrent Copy and Execution: Yes
  ECC Enabled: No
  Integrated GPU: No
  Unified Addressing: Yes
  Managed Memory: Yes
  Cooperative Launch: Yes
  Multi-Device Cooperative Launch: Yes

... (more output)

--- Recommended Kernel Configurations ---
  Block Size (threads): 128, 256, or 512 (multiples of 32)
  Grid Size (blocks): At least 136 (2x SMs) for good occupancy
  Shared Memory Usage: Keep under 48.00 KB per block
```

## Key Properties Explained

### Compute Capability

**Format**: `major.minor` (e.g., 8.6, 7.5, 6.1)

The compute capability defines:
- What CUDA features are available
- Instruction set architecture
- Hardware capabilities

**Common Versions:**
- **3.x**: Kepler (GTX 700, Tesla K-series) - Basic features
- **5.x**: Maxwell (GTX 900) - Improved efficiency
- **6.x**: Pascal (GTX 10xx, Tesla P100) - FP16, NVLink
- **7.0-7.2**: Volta (V100) - Tensor cores, independent thread scheduling
- **7.5**: Turing (RTX 20xx, GTX 16xx) - RT cores, INT8
- **8.0-8.6**: Ampere (A100, RTX 30xx) - 2x throughput, BF16
- **8.9**: Ada Lovelace (RTX 40xx) - Ada features
- **9.0**: Hopper (H100) - Transformer engine, FP8

**Usage:**
```cuda
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

if (prop.major < 6) {
    printf("Warning: This GPU is quite old\n");
}

if (prop.major >= 7) {
    // Can use Tensor Cores
}
```

### Memory Specifications

**Global Memory**: Main GPU DRAM
- Typical: 8GB (RTX 2070) to 80GB (A100)
- Latency: 200-400 cycles
- Bandwidth: 200-900 GB/s
- Where most data resides

**Shared Memory**: Fast on-chip memory
- Typical: 48-164 KB per block
- Latency: ~4 cycles
- Bandwidth: 10+ TB/s
- Programmer-managed cache

**L2 Cache**: Hardware-managed cache
- Typical: 1-6 MB
- Automatically caches global memory
- Transparent to programmer

**Registers**: Fastest storage
- 64K 32-bit registers per SM
- Private to each thread
- Compiler-managed

### Thread Limits

**Max Threads per Block**: 1024
- Hardware limit for a single block
- Often use 256 or 512 in practice
- Must be ≤ 1024

**Max Threads per SM**: 1536-2048
- Maximum concurrent threads per SM
- Determines occupancy
- Shared among all blocks on the SM

**Max Blocks per SM**: Varies
- Limited by threads, registers, shared memory
- Typically 16-32 blocks
- Affects occupancy

**Warp Size**: Always 32
- Fundamental execution unit
- Threads in a warp execute in lockstep
- Block size should be multiple of 32

### Grid Limits

```
Max Grid Dimensions:
  X: 2,147,483,647 (2^31 - 1)
  Y: 65,535
  Z: 65,535
```

Practically unlimited in X dimension, constrained in Y and Z.

## Understanding Occupancy

**Occupancy** = Active Warps / Maximum Warps per SM

Higher occupancy (50-100%) generally better for:
- Memory-bound kernels
- Hiding memory latency
- Utilizing GPU resources

**Factors Limiting Occupancy:**
1. **Registers per thread**: Too many → fewer threads fit
2. **Shared memory per block**: Too much → fewer blocks fit
3. **Threads per block**: Too few → wastes resources
4. **Blocks per SM**: Limited by hardware

**Example Calculation:**
```
GPU: RTX 3080
- Max threads per SM: 1536
- Block size: 256 threads
- Blocks per SM: 1536 / 256 = 6 blocks
- Occupancy: (6 * 256) / 1536 = 100%
```

## Performance Metrics

### Theoretical Peak Performance

**FP32 (Single Precision):**
```
GFLOPS = SMs × Cores/SM × Clock (GHz) × 2 (FMA)
```

For RTX 3080:
```
68 SMs × 128 cores × 1.71 GHz × 2 = 29,786 GFLOPS ≈ 30 TFLOPS
```

**Memory Bandwidth:**
```
Bandwidth = Memory Clock × Bus Width × 2 (DDR) / 8
```

For RTX 3080:
```
9.5 GHz × 320 bits × 2 / 8 = 760 GB/s
```

### Achieved vs. Peak Performance

Most applications achieve:
- **10-30%** of peak FLOPS (due to memory bottlenecks)
- **40-80%** of peak bandwidth (well-optimized kernels)

## Using Device Properties in Code

### Query Device at Runtime

```cuda
int deviceCount;
cudaGetDeviceCount(&deviceCount);

for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);

    printf("Device %d: %s\n", i, prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Global Memory: %.2f GB\n",
           prop.totalGlobalMem / (1024.0*1024.0*1024.0));
}
```

### Adaptive Block Size

```cuda
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

// Use recommended block size
int blockSize = 256;
if (prop.maxThreadsPerBlock < 256) {
    blockSize = prop.maxThreadsPerBlock;
}

int gridSize = (n + blockSize - 1) / blockSize;
kernel<<<gridSize, blockSize>>>(args);
```

### Check Feature Availability

```cuda
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

if (prop.managedMemory) {
    // Can use Unified Memory
    cudaMallocManaged(&ptr, size);
} else {
    // Use explicit allocation
    cudaMalloc(&ptr, size);
}

if (prop.concurrentKernels) {
    // Can use streams for concurrent execution
}
```

### Optimize for Architecture

```cuda
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

// Adjust shared memory based on availability
int sharedMemSize;
if (prop.sharedMemPerBlock >= 64 * 1024) {
    sharedMemSize = 64 * 1024;  // Use 64KB
} else {
    sharedMemSize = 48 * 1024;  // Use 48KB
}

kernel<<<grid, block, sharedMemSize>>>(args);
```

## Multi-GPU Systems

### Enumerate Multiple GPUs

```cuda
int deviceCount;
cudaGetDeviceCount(&deviceCount);

printf("Found %d GPUs:\n", deviceCount);
for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("  GPU %d: %s\n", i, prop.name);
}
```

### Select GPU

```cuda
// Use GPU 1 instead of default GPU 0
cudaSetDevice(1);

// Now all CUDA operations use GPU 1
cudaMalloc(&ptr, size);
kernel<<<grid, block>>>(args);
```

### Check Peer Access

```cuda
int canAccess;
cudaDeviceCanAccessPeer(&canAccess, 0, 1);

if (canAccess) {
    cudaDeviceEnablePeerAccess(1, 0);
    // Can now directly access GPU 1's memory from GPU 0
}
```

## Compilation for Different Architectures

### Specify Compute Capability

```bash
# Compile for specific GPU
nvcc -arch=sm_86 code.cu  # RTX 30xx (Ampere)
nvcc -arch=sm_75 code.cu  # RTX 20xx (Turing)
nvcc -arch=sm_60 code.cu  # GTX 10xx (Pascal)

# Compile for multiple architectures
nvcc -gencode arch=compute_75,code=sm_75 \
     -gencode arch=compute_86,code=sm_86 \
     code.cu
```

### Find Your GPU's Architecture

```bash
# Using this program
./device_query | grep "Compute Capability"

# Using nvidia-smi
nvidia-smi --query-gpu=compute_cap --format=csv
```

## Common Device Properties to Check

### For Optimization

```cuda
prop.maxThreadsPerBlock          // Block size limit
prop.sharedMemPerBlock           // Shared memory available
prop.regsPerBlock                // Register limit
prop.maxThreadsPerMultiProcessor // Occupancy calculation
prop.multiProcessorCount         // Parallelism scale
```

### For Features

```cuda
prop.major, prop.minor      // Compute capability
prop.concurrentKernels      // Stream support
prop.managedMemory          // Unified Memory
prop.cooperativeLaunch      // Cooperative groups
```

### For Memory

```cuda
prop.totalGlobalMem        // Total VRAM
prop.l2CacheSize           // Cache size
prop.memoryBusWidth        // Bandwidth factor
prop.memoryClockRate       // Bandwidth factor
```

## Troubleshooting

### Issue: No CUDA Devices Found

**Possible Causes:**
1. No NVIDIA GPU installed
2. GPU drivers not installed
3. Wrong driver version

**Solution:**
```bash
# Check driver
nvidia-smi

# If command not found, install drivers:
# Ubuntu: sudo apt install nvidia-driver-535
# Or download from NVIDIA website
```

### Issue: Compute Capability Too Low

**Symptom**: Code won't run on old GPU
**Solution**: Compile for lower compute capability or remove advanced features

```bash
# Instead of:
nvcc -arch=sm_86 code.cu

# Use:
nvcc -arch=sm_50 code.cu  # Maxwell and later
```

### Issue: Out of Memory

**Solution**: Check available memory before allocation
```cuda
size_t free, total;
cudaMemGetInfo(&free, &total);
printf("Free: %.2f GB\n", free/(1024.0*1024.0*1024.0));
```

## Comparison with nvidia-smi

This program provides more detail than `nvidia-smi`:

```bash
# Basic info
nvidia-smi

# Detailed query
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
```

Our program shows:
- Theoretical performance calculations
- Detailed thread limits
- Recommended configurations
- Feature availability
- Optimization guidance

## Next Steps

After understanding your GPU:

1. **Apply Knowledge to Examples**
   - Revisit vector_add with optimal block size
   - Understand why certain configurations work better

2. **Optimize for Your Hardware**
   - Use shared memory within limits
   - Choose block sizes based on max threads
   - Scale grid size with SM count

3. **Move to Exercises**
   - Exercise 01: Vector Subtraction
   - Exercise 02: SAXPY
   - Apply device properties to optimize solutions

4. **Explore Advanced Topics**
   - Chapter 02: Memory optimization
   - Chapter 03: Shared memory
   - Architecture-specific optimizations

## References

- **CUDA Programming Guide**: [Compute Capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)
- **Device Properties**: [cudaDeviceProp](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html)
- **Architecture Whitepapers**: [NVIDIA Technical Docs](https://www.nvidia.com/en-us/data-center/resources/gpu-architecture/)

---

**Understanding your hardware is the first step to optimization!** Use this tool to discover your GPU's capabilities and limits before diving into performance tuning.
