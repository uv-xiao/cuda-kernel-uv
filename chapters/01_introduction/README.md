# Chapter 01: GPU & CUDA Introduction

Welcome to the first chapter of the CUDA Kernel Development Tutorial! This chapter introduces you to GPU computing with NVIDIA CUDA, covering fundamental concepts and getting you started with your first CUDA programs.

## Learning Goals

By the end of this chapter, you will be able to:

1. **Understand GPU Architecture Fundamentals**
   - Differentiate between CPU and GPU computing paradigms
   - Explain the SIMT (Single Instruction, Multiple Thread) execution model
   - Describe the streaming multiprocessor (SM) architecture
   - Understand warps and their role in GPU execution

2. **Master CUDA Programming Basics**
   - Write and launch simple CUDA kernels
   - Use the CUDA execution configuration syntax `<<<...>>>`
   - Query GPU device properties programmatically
   - Handle CUDA errors properly

3. **Comprehend CUDA's Thread Hierarchy**
   - Organize threads into blocks and grids
   - Calculate global thread IDs from block and thread indices
   - Choose appropriate grid and block dimensions
   - Understand the limitations and best practices for thread organization

4. **Navigate CUDA Memory Spaces**
   - Distinguish between host (CPU) and device (GPU) memory
   - Use `cudaMalloc` and `cudaFree` for device memory management
   - Transfer data between host and device with `cudaMemcpy`
   - Understand the basic memory hierarchy (global, shared, registers)

5. **Implement Data-Parallel Algorithms**
   - Apply the CUDA programming model to simple parallel problems
   - Implement vector operations on the GPU
   - Measure and compare GPU vs CPU performance
   - Debug and validate CUDA programs

## Prerequisites

### Required Knowledge
- **C/C++ Programming**: Solid understanding of C or C++ syntax, pointers, and memory management
- **Computer Architecture Basics**: Familiarity with processors, memory, and basic parallelism concepts
- **Command Line**: Ability to compile and run programs from the terminal

### Required Software
- **NVIDIA GPU**: CUDA-capable GPU with compute capability 3.0 or higher
- **CUDA Toolkit**: Version 11.0 or later (12.x recommended)
- **C++ Compiler**: GCC 7.x or later (Linux), MSVC 2019+ (Windows), or Clang
- **CMake**: Version 3.18 or later for building examples
- **Python 3.x**: (Optional) For running test scripts

### Recommended Background
- Basic understanding of parallel programming concepts (optional but helpful)
- Familiarity with linear algebra operations (vectors, matrices)

## Key Concepts

### 1. GPU Architecture Overview

**CPU vs GPU Paradigm:**
- **CPU**: Optimized for sequential execution, complex control flow, and low-latency operations
  - Few powerful cores (4-32 typically)
  - Large caches for fast data access
  - Branch prediction and out-of-order execution
  - Best for: Serial tasks, complex logic, irregular memory access

- **GPU**: Optimized for massive data parallelism and high throughput
  - Thousands of lightweight cores (2,000-10,000+)
  - Smaller caches, massive memory bandwidth
  - SIMT execution model
  - Best for: Regular, data-parallel computations

**Streaming Multiprocessor (SM):**
- The fundamental processing unit of NVIDIA GPUs
- Contains multiple CUDA cores, special function units, and registers
- Modern GPUs have 50-100+ SMs
- Each SM can execute multiple thread blocks concurrently

**Warps:**
- A warp is a group of 32 threads that execute together in SIMT fashion
- All threads in a warp execute the same instruction simultaneously
- Branch divergence within a warp reduces efficiency
- Warp scheduling is hardware-managed and transparent to programmers

### 2. CUDA Thread Hierarchy

CUDA organizes parallel execution into a three-level hierarchy:

```
Grid (1D, 2D, or 3D)
├── Block 0 (1D, 2D, or 3D)
│   ├── Thread 0
│   ├── Thread 1
│   └── ...
├── Block 1
│   ├── Thread 0
│   └── ...
└── ...
```

**Thread:**
- The smallest unit of execution
- Has a unique ID within its block: `threadIdx.{x,y,z}`
- Executes the kernel code

**Block:**
- A group of threads that can cooperate via shared memory
- Has a unique ID within the grid: `blockIdx.{x,y,z}`
- All threads in a block execute on the same SM
- Maximum: 1024 threads per block (typical)

**Grid:**
- A collection of blocks executing the same kernel
- Blocks within a grid execute independently
- Grid dimensions specified at kernel launch: `<<<gridDim, blockDim>>>`

**Global Thread ID Calculation:**
For 1D grids and blocks:
```cuda
int tid = blockIdx.x * blockDim.x + threadIdx.x;
```

For 2D grids and blocks:
```cuda
int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
```

### 3. CUDA Memory Model

CUDA exposes several memory spaces with different scopes and performance characteristics:

**Host Memory:**
- Resides on the CPU (system RAM)
- Allocated with standard C/C++ functions: `malloc`, `new`
- Not directly accessible from GPU kernels

**Device Memory Spaces:**

1. **Global Memory**
   - Largest memory space (typically 8GB-80GB)
   - Accessible by all threads in all blocks
   - Highest latency (hundreds of cycles)
   - Persistent across kernel launches
   - Allocated with `cudaMalloc`, freed with `cudaFree`

2. **Shared Memory**
   - Small, fast memory (48KB-164KB per block)
   - Shared among threads within a block
   - Low latency (a few cycles)
   - Declared with `__shared__` keyword
   - Lifetime limited to block execution

3. **Registers**
   - Fastest memory, private to each thread
   - Automatically managed by compiler
   - Limited quantity (64K 32-bit registers per SM)
   - Excessive register usage limits occupancy

4. **Constant Memory**
   - Read-only, cached
   - 64KB total
   - Efficient for values read by all threads

5. **Texture/Surface Memory**
   - Specialized read-only memory with spatial caching
   - Optimized for 2D/3D locality

**Memory Transfer:**
```cuda
// Host to Device
cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);

// Device to Host
cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost);

// Device to Device
cudaMemcpy(d_ptr1, d_ptr2, size, cudaMemcpyDeviceToDevice);
```

### 4. CUDA Kernel Basics

**Kernel Declaration:**
```cuda
__global__ void myKernel(int* data, int n) {
    // Kernel code executes on GPU
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        data[tid] = tid;
    }
}
```

**Kernel Launch:**
```cuda
int threadsPerBlock = 256;
int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
myKernel<<<numBlocks, threadsPerBlock>>>(d_data, n);
```

**CUDA Function Qualifiers:**
- `__global__`: Kernel function, called from host, executed on device
- `__device__`: Device function, called from device, executed on device
- `__host__`: Host function (default), called from host, executed on host
- `__host__ __device__`: Can be compiled for both host and device

### 5. Error Handling

**Always check CUDA API calls:**
```cuda
cudaError_t err = cudaMalloc(&d_data, size);
if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}
```

**Check kernel launch errors:**
```cuda
myKernel<<<grid, block>>>(args);
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
}
cudaDeviceSynchronize(); // Wait for kernel to complete
```

## Reading Materials

### Essential Resources

1. **NVIDIA CUDA C++ Programming Guide**
   - [Official Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
   - Focus on:
     - Chapter 2: Programming Model
     - Chapter 3: Programming Interface (Sections 3.1-3.2)
     - Appendix B: C++ Language Extensions
     - Appendix F: Compute Capabilities

2. **CUDA C++ Best Practices Guide**
   - [Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
   - Focus on:
     - Chapter 2: Heterogeneous Computing
     - Chapter 9: Memory Optimization (Overview)

3. **CUDA Toolkit Documentation**
   - [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
   - Reference for all CUDA runtime functions

### Recommended Reading

1. **"Professional CUDA C Programming" by John Cheng, Max Grossman, Ty McKercher**
   - Chapters 1-3 for foundational concepts

2. **"Programming Massively Parallel Processors" by David Kirk and Wen-mei Hwu**
   - Chapters 1-4 for GPU architecture and basic CUDA

3. **NVIDIA Blog Posts**
   - [An Easy Introduction to CUDA C and C++](https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/)
   - [Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)

### Online Courses

1. **NVIDIA DLI - Fundamentals of Accelerated Computing with CUDA C/C++**
   - [NVIDIA Deep Learning Institute](https://www.nvidia.com/en-us/training/)

2. **Coursera - GPU Programming Specialization**
   - By Johns Hopkins University

## Chapter Outline

### Examples

This chapter includes three hands-on examples that progressively introduce CUDA concepts:

#### 1. Hello CUDA (`examples/01_hello_cuda/`)
- **Concepts**: Basic kernel structure, thread/block IDs, kernel launch
- **Skills**: Writing and launching your first CUDA kernel
- **Code**: Simple kernel that prints thread and block IDs
- **Learning Focus**: Understanding the thread hierarchy

#### 2. Vector Addition (`examples/02_vector_add/`)
- **Concepts**: Memory management, data transfer, error handling, timing
- **Skills**: Implementing a complete GPU-accelerated algorithm
- **Code**: Adding two vectors element-wise on the GPU
- **Learning Focus**: Full workflow from allocation to execution to cleanup

#### 3. Device Query (`examples/03_device_query/`)
- **Concepts**: GPU properties, compute capability, hardware limits
- **Skills**: Programmatically querying device information
- **Code**: Retrieving and displaying GPU specifications
- **Learning Focus**: Understanding hardware constraints and capabilities

### Exercises

Two exercises to reinforce your learning:

#### 1. Vector Subtraction (`exercises/01_vector_subtract/`)
- **Objective**: Modify vector addition to perform subtraction
- **Difficulty**: Beginner
- **Skills Practiced**: Kernel modification, understanding data flow
- **Includes**: Problem description, starter code, solution, automated tests

#### 2. SAXPY (`exercises/02_saxpy/`)
- **Objective**: Implement SAXPY operation (y = a*x + y)
- **Difficulty**: Beginner
- **Skills Practiced**: Multiple inputs, scalar-vector operations
- **Includes**: Problem description, starter code, solution, automated tests

### Building the Examples

All examples use CMake for easy building:

```bash
# From chapter directory
cd /home/uvxiao/cuda-kernel-tutorial/chapters/01_introduction
mkdir build && cd build
cmake ..
make

# Run examples
./examples/01_hello_cuda/hello_cuda
./examples/02_vector_add/vector_add
./examples/03_device_query/device_query

# Run exercise solutions
./exercises/01_vector_subtract/vector_subtract_solution
./exercises/02_saxpy/saxpy_solution
```

## Getting Started

1. **Verify Your Setup**
   ```bash
   nvcc --version          # Check CUDA compiler
   nvidia-smi              # Check GPU status
   ```

2. **Start with Examples**
   - Begin with `examples/01_hello_cuda/`
   - Read the code comments carefully
   - Modify and experiment with thread/block dimensions

3. **Work Through Examples in Order**
   - Each example builds on previous concepts
   - Run and understand output before moving forward

4. **Attempt Exercises**
   - Try solving exercises before looking at solutions
   - Use provided test scripts to validate your solutions

5. **Experiment and Explore**
   - Modify examples to see effects
   - Try different grid/block configurations
   - Profile performance with different input sizes

## Common Pitfalls for Beginners

1. **Forgetting to Check Errors**
   - Always check return values of CUDA API calls
   - Use `cudaGetLastError()` after kernel launches

2. **Memory Confusion**
   - Host pointers cannot be dereferenced in device code
   - Device pointers cannot be dereferenced in host code
   - Use `cudaMemcpy` for transfer, not direct assignment

3. **Incorrect Thread ID Calculation**
   - Remember: `tid = blockIdx.x * blockDim.x + threadIdx.x`
   - Always add bounds checking: `if (tid < n)`

4. **Not Synchronizing**
   - `cudaMemcpy` is synchronous, but kernel launches are asynchronous
   - Use `cudaDeviceSynchronize()` when needed

5. **Inefficient Configurations**
   - Thread block size should typically be a multiple of 32 (warp size)
   - Common block sizes: 128, 256, 512

## Next Steps

After completing this chapter, you'll be ready for:
- **Chapter 02**: Memory Management and Optimization
- **Chapter 03**: Thread Synchronization and Cooperation
- **Chapter 04**: Performance Optimization Fundamentals

## Additional Resources

- **CUDA Samples**: `/usr/local/cuda/samples/` (if CUDA installed in default location)
- **NVIDIA Developer Forums**: https://forums.developer.nvidia.com/
- **CUDA GitHub**: https://github.com/NVIDIA/cuda-samples
- **Stack Overflow**: Tag `cuda` for community support

---

**Note**: This tutorial assumes CUDA 11.0 or later. Some features may differ in older versions. Always refer to the official NVIDIA documentation for your specific CUDA version.

Happy learning, and welcome to GPU programming!
