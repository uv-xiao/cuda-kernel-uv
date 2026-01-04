# Example 02: Vector Addition

## Overview

This example demonstrates a complete CUDA application workflow by implementing vector addition on the GPU. Unlike the "Hello CUDA" example which focused on thread organization, this example shows real data processing including memory management, data transfer, performance measurement, and result validation.

**Operation**: `C[i] = A[i] + B[i]` for all elements i

This is a perfect introductory GPU algorithm because:
- Each element is computed independently (embarrassingly parallel)
- Memory access is coalesced (sequential)
- Simple computation shows pure memory bandwidth limits
- Easy to verify correctness

## Learning Objectives

1. **Complete CUDA Workflow**
   - Allocate memory on both host and device
   - Transfer data between CPU and GPU
   - Execute kernels and retrieve results
   - Validate GPU computations
   - Clean up resources properly

2. **Memory Management**
   - Use `cudaMalloc()` and `cudaFree()`
   - Understand `cudaMemcpy()` directions
   - Query GPU memory usage
   - Distinguish host vs device pointers

3. **Kernel Implementation**
   - Calculate global thread indices
   - Implement bounds checking
   - Handle irregular array sizes
   - Design data-parallel algorithms

4. **Performance Measurement**
   - Use CUDA events for GPU timing
   - Measure transfer overhead
   - Compare CPU vs GPU performance
   - Calculate effective bandwidth

5. **Error Handling**
   - Check all CUDA API calls
   - Use error checking macros
   - Handle kernel launch failures
   - Debug incorrect results

## Key Concepts

### The Complete CUDA Workflow

```
┌─────────────────────────────────────────────┐
│ 1. Allocate Host Memory (malloc)           │
├─────────────────────────────────────────────┤
│ 2. Initialize Data on Host                 │
├─────────────────────────────────────────────┤
│ 3. Allocate Device Memory (cudaMalloc)     │
├─────────────────────────────────────────────┤
│ 4. Transfer H → D (cudaMemcpy H2D)        │
├─────────────────────────────────────────────┤
│ 5. Launch Kernel <<<grid, block>>>        │
├─────────────────────────────────────────────┤
│ 6. Transfer D → H (cudaMemcpy D2H)        │
├─────────────────────────────────────────────┤
│ 7. Verify Results                          │
├─────────────────────────────────────────────┤
│ 8. Free Device Memory (cudaFree)           │
├─────────────────────────────────────────────┤
│ 9. Free Host Memory (free)                 │
└─────────────────────────────────────────────┘
```

### Memory Spaces

```
CPU (Host)                           GPU (Device)
┌──────────────┐                    ┌──────────────┐
│ float* h_a   │ ──cudaMemcpy H2D→ │ float* d_a   │
│ float* h_b   │ ──cudaMemcpy H2D→ │ float* d_b   │
│ float* h_c   │ ←─cudaMemcpy D2H── │ float* d_c   │
└──────────────┘                    └──────────────┘
  malloc/free                         cudaMalloc/cudaFree
```

**Critical Rules:**
- Cannot dereference device pointers on host
- Cannot dereference host pointers on device
- Must use `cudaMemcpy()` for transfer
- Host and device have separate address spaces

### Bounds Checking Pattern

```cuda
__global__ void kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {  // Critical bounds check!
        data[i] = ...;
    }
}
```

**Why?** When `n` is not a multiple of block size:
- We round up grid size: `gridSize = (n + blockSize - 1) / blockSize`
- This creates extra threads: `gridSize * blockSize > n`
- Extra threads must not access invalid memory

**Example:**
- n = 1000, blockSize = 256
- gridSize = (1000 + 255) / 256 = 4 blocks
- Total threads = 4 * 256 = 1024
- Extra threads = 1024 - 1000 = 24 (must be bounds-checked)

### Grid Size Calculation

```cuda
int threadsPerBlock = 256;  // Common choices: 128, 256, 512
int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;  // Ceiling division
```

This formula ensures enough threads to cover all elements:
- If n = 1000, blockSize = 256: gridSize = 4 blocks (1024 threads)
- If n = 1024, blockSize = 256: gridSize = 4 blocks (1024 threads)
- If n = 1025, blockSize = 256: gridSize = 5 blocks (1280 threads)

### CUDA Error Checking Macro

```cuda
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Usage:
CUDA_CHECK(cudaMalloc(&d_ptr, size));
CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice));
```

## Building and Running

### Using CMake

```bash
cd /home/uvxiao/cuda-kernel-tutorial/chapters/01_introduction/examples/02_vector_add
mkdir build && cd build
cmake ..
make
./vector_add
```

### Using nvcc

```bash
nvcc -o vector_add vector_add.cu
./vector_add
```

### Running with Different Sizes

```bash
./vector_add 1000000      # 1M elements (default)
./vector_add 10000000     # 10M elements
./vector_add 100000000    # 100M elements
```

## Expected Output

```
=== CUDA Vector Addition Example ===

Vector size: 1000000 elements (3.81 MB per vector)

--- Memory Allocation ---
Allocated host memory: 15.26 MB

--- Initializing Data ---
Initialized input vectors with random values

--- Device Memory Allocation ---
Allocated device memory: 11.44 MB
GPU memory: 156.50 MB used, 7977.50 MB free, 8134.00 MB total

--- Host to Device Transfer ---
Transferred 7.63 MB to device in 1.23 ms (5.90 GB/s)

--- Kernel Launch Configuration ---
Grid size: 3907 blocks
Block size: 256 threads
Total threads: 1000192 (covers 1000000 elements)
Extra threads: 192 (will be bounds-checked)

--- Launching Kernel ---
Kernel executed in 0.15 ms
Effective bandwidth: 152.38 GB/s

--- Device to Host Transfer ---
Transferred 3.81 MB from device in 0.65 ms (5.58 GB/s)

--- CPU Execution ---
CPU execution time: 2.45 ms

--- Verification ---
SUCCESS: GPU results match CPU results!

First 10 results:
  i   A[i]      B[i]      CPU       GPU
  0    0.421    0.789    1.210    1.210
  1    0.156    0.923    1.079    1.079
  2    0.687    0.234    0.921    0.921
  ...

--- Performance Summary ---
CPU time:                   2.45 ms
GPU kernel time:            0.15 ms
GPU transfer time:          1.88 ms (H2D: 1.23, D2H: 0.65)
Total GPU time:             2.03 ms

Speedup (kernel only): 16.33x
Speedup (with transfer): 1.21x

--- Cleanup ---
Released all resources

=== Program Complete ===
```

## Performance Analysis

### Understanding the Metrics

1. **Kernel Time** (~0.15 ms for 1M elements)
   - Pure computation time on GPU
   - Usually very fast for simple operations
   - Shows GPU's raw processing power

2. **Transfer Time** (~1.88 ms total)
   - H2D (Host to Device): Copying input data
   - D2H (Device to Host): Copying results
   - Often dominates for small problems
   - Limited by PCIe bandwidth (~16 GB/s typical)

3. **Effective Bandwidth**
   - Formula: `(bytes_read + bytes_written) / time`
   - For vector add: 3 * n * sizeof(float) / kernel_time
   - Theoretical max for modern GPUs: 500-900 GB/s
   - This simple operation is memory-bound

4. **Speedup**
   - **Kernel only**: GPU kernel vs CPU (often 10-100x)
   - **With transfer**: Total GPU vs CPU (often 1-5x for small data)
   - Transfer overhead is why GPU needs large data to excel

### When Does GPU Win?

**GPU is faster when:**
- Data size is large (millions of elements)
- Computation is reused (data stays on GPU)
- Operation is compute-intensive (not just memory copy)
- Many independent operations (data parallelism)

**GPU is slower when:**
- Data size is small (transfer overhead dominates)
- Operation is very simple (vector add is memory-bound)
- Single operation (no data reuse)
- Complex control flow (branch divergence)

### Bandwidth Calculation Example

For 1M elements:
- Read A: 1M * 4 bytes = 4 MB
- Read B: 1M * 4 bytes = 4 MB
- Write C: 1M * 4 bytes = 4 MB
- **Total: 12 MB**
- Time: 0.15 ms
- **Bandwidth: 12 MB / 0.00015 s = 80 GB/s**

## Experiments to Try

### 1. Vary Vector Size

```bash
./vector_add 1000        # 1K - GPU slower due to overhead
./vector_add 100000      # 100K - Break-even point
./vector_add 10000000    # 10M - GPU clearly faster
./vector_add 100000000   # 100M - Large speedup
```

Observe how speedup changes with data size.

### 2. Vary Block Size

Modify the code:
```cuda
int threadsPerBlock = 128;  // Try: 32, 64, 128, 256, 512, 1024
```

Questions:
- Does block size affect performance?
- What happens with non-multiple of 32?
- What's the optimal block size for your GPU?

### 3. Remove Bounds Checking

Comment out the `if (i < n)` check:
```cuda
__global__ void vectorAddKernel(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // if (i < n) {  // Commented out!
        c[i] = a[i] + b[i];
    // }
}
```

Run with n not divisible by block size. What happens?
- Invalid memory access
- Kernel fails or corrupts data
- This demonstrates why bounds checking is critical

### 4. Measure Only Kernel Time

Use CUDA events (already in code):
```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);
kernel<<<grid, block>>>(args);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float milliseconds;
cudaEventElapsedTime(&milliseconds, start, stop);
```

This gives accurate GPU-only timing.

### 5. Increase Arithmetic Intensity

Make the kernel do more work per memory access:
```cuda
__global__ void vectorAddComplexKernel(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float result = a[i] + b[i];
        // Add more computation
        for (int j = 0; j < 100; j++) {
            result = result * 1.001f;
        }
        c[i] = result;
    }
}
```

Does GPU speedup improve with more computation?

## Common Issues and Solutions

### Issue 1: GPU Slower Than CPU

**Symptom**: Speedup < 1.0, GPU takes longer
**Causes:**
- Data size too small (transfer dominates)
- Operation too simple (memory-bound)
- Single operation (no amortization of transfer)

**Solutions:**
- Increase data size
- Batch multiple operations
- Keep data on GPU for multiple kernels
- Use streams for overlap

### Issue 2: "invalid configuration argument"

**Symptom**: Kernel launch fails
**Cause**: Too many threads per block (max is typically 1024)
**Solution:**
```cuda
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
// Use smaller block size
```

### Issue 3: Results Don't Match CPU

**Symptom**: Verification fails
**Possible Causes:**
1. Forgot bounds checking
2. Incorrect global index calculation
3. Memory not initialized
4. Wrong cudaMemcpy direction

**Debug Steps:**
```cuda
// Print first few results
for (int i = 0; i < 10; i++) {
    printf("%d: CPU=%f, GPU=%f, diff=%e\n",
           i, cpu[i], gpu[i], fabs(cpu[i] - gpu[i]));
}

// Check memory was copied
cudaMemcpy(h_check, d_a, sizeof(float) * 10, cudaMemcpyDeviceToHost);
printf("First 10 elements of d_a: ");
for (int i = 0; i < 10; i++) printf("%.2f ", h_check[i]);
```

### Issue 4: Out of Memory

**Symptom**: `cudaMalloc` fails
**Solution:**
```cuda
size_t free_mem, total_mem;
cudaMemGetInfo(&free_mem, &total_mem);
printf("Free: %.2f GB, Total: %.2f GB\n",
       free_mem/(1024.0*1024.0*1024.0),
       total_mem/(1024.0*1024.0*1024.0));
// Reduce data size or use multiple batches
```

### Issue 5: Slow Performance

**Possible Causes:**
- Non-coalesced memory access
- Wrong block size
- Low occupancy

**Check:**
```bash
# Profile with nvprof (CUDA < 11) or nsys (CUDA 11+)
nvprof ./vector_add
nsys profile ./vector_add
```

## Code Deep Dive

### Memory Allocation Pattern

```cuda
// Host memory (CPU RAM)
float* h_a = (float*)malloc(bytes);
float* h_b = (float*)malloc(bytes);
float* h_c = (float*)malloc(bytes);

// Device memory (GPU DRAM)
float *d_a, *d_b, *d_c;
cudaMalloc(&d_a, bytes);  // Note: pass address of pointer
cudaMalloc(&d_b, bytes);
cudaMalloc(&d_c, bytes);

// Transfer to device
cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

// Execute kernel
kernel<<<grid, block>>>(d_a, d_b, d_c, n);  // Use device pointers

// Transfer back
cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

// Cleanup
cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
free(h_a); free(h_b); free(h_c);
```

### Error Checking Best Practices

```cuda
// Method 1: Inline checking
cudaError_t err = cudaMalloc(&d_ptr, size);
if (err != cudaSuccess) {
    fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));
    exit(1);
}

// Method 2: Macro (cleaner)
#define CUDA_CHECK(call) /* ... */
CUDA_CHECK(cudaMalloc(&d_ptr, size));

// Method 3: Wrapper function
void checkCuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d - %s\n",
                file, line, cudaGetErrorString(err));
        exit(1);
    }
}
#define CHECK(call) checkCuda(call, __FILE__, __LINE__)
```

## Next Steps

After mastering this example:

1. **Example 03: Device Query**
   - Learn to query GPU properties
   - Understand hardware capabilities
   - Optimize for specific GPU architectures

2. **Exercises**
   - Exercise 01: Implement vector subtraction
   - Exercise 02: Implement SAXPY (y = a*x + y)

3. **Advanced Topics** (future chapters)
   - Shared memory optimization
   - Memory coalescing patterns
   - Stream parallelism
   - Unified memory

## References

- **CUDA Programming Guide**: [Memory Management](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-management)
- **CUDA Best Practices**: [Transfers](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#host-device-data-transfer)
- **CUDA Runtime API**: [cudaMemcpy](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)

---

**Congratulations!** You now understand the complete CUDA workflow. This pattern (allocate, transfer, execute, retrieve) is the foundation for all GPU computing.
