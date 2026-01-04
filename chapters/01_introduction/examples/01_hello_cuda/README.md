# Example 01: Hello CUDA

## Overview

This is your first CUDA program! It demonstrates the fundamental structure of a CUDA application by launching simple kernels that print thread and block information. This example introduces the core concepts of CUDA's thread hierarchy and kernel launching.

## Learning Objectives

After working through this example, you will understand:

1. **CUDA Kernel Basics**
   - How to write a `__global__` kernel function
   - The execution configuration syntax `<<<gridDim, blockDim>>>`
   - Asynchronous kernel launches and synchronization

2. **Thread Hierarchy**
   - The relationship between grids, blocks, and threads
   - Built-in variables: `threadIdx`, `blockIdx`, `blockDim`, `gridDim`
   - Calculating global thread IDs
   - 1D, 2D, and 3D thread organizations

3. **CUDA Programming Workflow**
   - Setting up a basic CUDA program structure
   - Error checking for CUDA API calls
   - Using `cudaDeviceSynchronize()` to wait for GPU operations

## Key Concepts

### The __global__ Qualifier

```cuda
__global__ void myKernel() {
    // This function runs on the GPU
    // Can be called from host (CPU) code
}
```

The `__global__` keyword marks a function as a CUDA kernel that:
- Executes on the device (GPU)
- Is callable from the host (CPU)
- Must return `void`
- Runs in parallel across many threads

### Kernel Launch Syntax

```cuda
myKernel<<<numBlocks, threadsPerBlock>>>();
```

The triple angle brackets `<<<...>>>` specify the **execution configuration**:
- **First parameter**: Number of blocks in the grid (can be `int` or `dim3`)
- **Second parameter**: Number of threads per block (can be `int` or `dim3`)
- Optional third parameter: Shared memory size (bytes)
- Optional fourth parameter: CUDA stream

### Thread Indexing

Every thread can identify itself using built-in variables:

```cuda
// 1D indexing
int tid = blockIdx.x * blockDim.x + threadIdx.x;

// 2D indexing
int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

// 3D indexing
int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
int tid_z = blockIdx.z * blockDim.z + threadIdx.z;
```

### Multi-Dimensional Grids

CUDA supports 1D, 2D, and 3D organizations using the `dim3` type:

```cuda
dim3 gridDim(2, 2);    // 2x2 grid = 4 blocks
dim3 blockDim(4, 4);   // 4x4 threads per block = 16 threads
myKernel<<<gridDim, blockDim>>>();
// Total threads: 4 blocks * 16 threads = 64 threads
```

## Code Structure

### helloKernel()
Simple 1D kernel that prints thread and block IDs. Demonstrates:
- Basic thread identification
- Global thread ID calculation
- Printf from device code

### hello2DKernel()
2D version that shows how to work with 2D grids and blocks. Useful for:
- Image processing (x, y coordinates)
- Matrix operations (row, column indices)
- 2D spatial data

### Main Function Examples
1. **Example 1**: Basic 1D launch with 2 blocks, 4 threads each
2. **Example 2**: Different configurations with same thread count
3. **Example 3**: 2D grid and block dimensions
4. **Example 4**: Querying hardware limitations

## Building and Running

### Using CMake (Recommended)

```bash
# From this directory
mkdir build && cd build
cmake ..
make
./hello_cuda
```

### Using nvcc Directly

```bash
nvcc -o hello_cuda hello.cu
./hello_cuda
```

### Specifying GPU Architecture

If you get "no kernel image available" errors, specify your GPU architecture:

```bash
# For RTX 30xx (Ampere)
nvcc -arch=sm_86 -o hello_cuda hello.cu

# For RTX 20xx (Turing)
nvcc -arch=sm_75 -o hello_cuda hello.cu

# For GTX 10xx (Pascal)
nvcc -arch=sm_61 -o hello_cuda hello.cu
```

Check your GPU's compute capability:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

## Expected Output

The program produces several sections of output:

```
=== CUDA Hello World Example ===

--- Example 1: 1D Grid of 1D Blocks ---
Launching kernel with 2 blocks, 4 threads per block
Total threads: 8

Hello from thread 0 in block 0 (global ID: 0)
Hello from thread 1 in block 0 (global ID: 1)
Hello from thread 2 in block 0 (global ID: 2)
Hello from thread 3 in block 0 (global ID: 3)
Hello from thread 0 in block 1 (global ID: 4)
Hello from thread 1 in block 1 (global ID: 5)
Hello from thread 2 in block 1 (global ID: 6)
Hello from thread 3 in block 1 (global ID: 7)

[... more output ...]
```

**Important Notes:**
- Thread output order is **not guaranteed** - threads execute in parallel
- Messages may appear interleaved or out of sequence
- This is normal and expected behavior for parallel execution
- Each run might produce different ordering

## Experiments to Try

### 1. Change Block and Thread Counts
```cuda
// Try these configurations:
helloKernel<<<1, 16>>>();      // 1 block, 16 threads
helloKernel<<<8, 32>>>();      // 8 blocks, 32 threads each
helloKernel<<<256, 256>>>();   // 256 blocks, 256 threads each
```

Observe how the output changes with different configurations.

### 2. Maximum Threads Per Block
```cuda
// Most GPUs support up to 1024 threads per block
helloKernel<<<1, 1024>>>();

// What happens with more than 1024?
// helloKernel<<<1, 2048>>>();  // This will fail!
```

### 3. 3D Grids
Extend the example to use 3D dimensions:
```cuda
dim3 grid(2, 2, 2);   // 2x2x2 = 8 blocks
dim3 block(4, 4, 4);  // 4x4x4 = 64 threads per block
```

### 4. Remove Synchronization
Comment out `cudaDeviceSynchronize()` and see what happens:
```cuda
helloKernel<<<2, 4>>>();
// cudaDeviceSynchronize();  // Commented out
```
Does output still appear? Why or why not?

### 5. Add Error Injection
Try invalid configurations to see error handling:
```cuda
// Too many threads per block
helloKernel<<<1, 2048>>>();
checkCudaError(cudaGetLastError(), "Invalid config");
```

## Common Issues and Solutions

### Issue: No Output Appears
**Cause**: Program exits before GPU completes execution
**Solution**: Add `cudaDeviceSynchronize()` before program ends

### Issue: "invalid configuration argument"
**Cause**: Block dimensions exceed hardware limits (typically 1024 threads)
**Solution**: Reduce threads per block
```cuda
// Check limits:
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
```

### Issue: "no kernel image is available"
**Cause**: Compiled for wrong GPU architecture
**Solution**: Specify correct compute capability:
```bash
nvcc -arch=sm_86 hello.cu  # For your GPU architecture
```

### Issue: Output Order is Random
**Cause**: This is expected behavior
**Solution**: No fix needed - parallel execution has no guaranteed order

## Understanding the Output

### Global Thread ID Calculation

For 1D:
```
Block 0, Thread 0: globalID = 0 * 4 + 0 = 0
Block 0, Thread 1: globalID = 0 * 4 + 1 = 1
Block 0, Thread 2: globalID = 0 * 4 + 2 = 2
Block 0, Thread 3: globalID = 0 * 4 + 3 = 3
Block 1, Thread 0: globalID = 1 * 4 + 0 = 4
Block 1, Thread 1: globalID = 1 * 4 + 1 = 5
...
```

For 2D:
```
Block (0,0), Thread (0,0): global = (0*2+0, 0*2+0) = (0,0)
Block (0,0), Thread (1,0): global = (0*2+1, 0*2+0) = (1,0)
Block (1,0), Thread (0,0): global = (1*2+0, 0*2+0) = (2,0)
...
```

## Performance Considerations

### Block Size Selection
- **Too small** (e.g., 32): Underutilizes GPU, poor occupancy
- **Too large** (e.g., 1024): May limit occupancy due to resource constraints
- **Recommended**: 128-512 threads, typically 256
- **Rule**: Use multiples of 32 (warp size)

### Grid Size Selection
- Should be large enough to saturate the GPU
- Typically many more blocks than streaming multiprocessors (SMs)
- For data-parallel work: `gridSize = (dataSize + blockSize - 1) / blockSize`

## Next Steps

After understanding this example:

1. **Move to Example 02**: Vector Addition
   - Applies these concepts to real data processing
   - Introduces memory management
   - Shows complete GPU workflow

2. **Explore Example 03**: Device Query
   - Learn about hardware capabilities
   - Understand resource limits
   - Query GPU properties programmatically

3. **Try the Exercises**:
   - Exercise 01: Vector Subtraction
   - Exercise 02: SAXPY Operation

## Additional Resources

- **CUDA Programming Guide**: [Thread Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy)
- **CUDA Samples**: Check `0_Simple/vectorAdd` in CUDA samples
- **NVIDIA Blog**: [Easy Introduction to CUDA](https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/)

## Questions to Consider

1. Why is the output order non-deterministic?
2. What happens to unused threads if data size isn't divisible by block size?
3. How would you organize threads for a 2D matrix?
4. What are the trade-offs between many small blocks vs. few large blocks?
5. Why is 32 (warp size) significant for block dimensions?

---

**Congratulations!** You've launched your first CUDA kernels. Understanding this foundation is crucial for all GPU programming that follows.
