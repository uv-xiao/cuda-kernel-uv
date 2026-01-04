# Exercise 02: Parallel Dot Product

## Objective

Implement an efficient parallel dot product using reduction techniques and shared memory.

## Problem Description

Given two vectors A and B of length N, compute their dot product:

```
dot(A, B) = A[0]*B[0] + A[1]*B[1] + ... + A[N-1]*B[N-1]
```

**Example:**
```
A = [1, 2, 3, 4]
B = [5, 6, 7, 8]
dot(A, B) = 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
```

## Your Tasks

### Part 1: Naive Implementation

Implement a basic dot product using atomic operations.

```cuda
__global__ void dotProductNaive(float *A, float *B, float *result, int n);
```

**Requirements:**
- Each thread computes one product
- Use `atomicAdd()` to accumulate result
- Handle boundary conditions

### Part 2: Shared Memory Reduction

Implement dot product using shared memory for efficient reduction.

```cuda
__global__ void dotProductShared(float *A, float *B, float *partial, int n);
```

**Requirements:**
- Each block produces one partial result
- Use shared memory for block-level reduction
- Implement sequential addressing to avoid divergence
- Store partial results to global memory

### Part 3: Optimized with Unrolled Warp

Optimize the reduction by unrolling the last warp.

```cuda
__global__ void dotProductOptimized(float *A, float *B, float *partial, int n);
```

**Requirements:**
- Same as Part 2 but unroll last warp
- No `__syncthreads()` in last warp
- Use `volatile` keyword appropriately

## Performance Goals

For N = 64M elements on a modern GPU:

| Implementation | Target Time |
|----------------|-------------|
| Naive (atomic) | < 10 ms |
| Shared Memory | < 2 ms |
| Optimized | < 1 ms |

## Starter Code Structure

The `starter.cu` file provides:
- Vector allocation and initialization
- Timing infrastructure
- CPU reference implementation
- Host-side reduction of partial results
- Main driver code

You need to implement:
1. `dotProductNaive()` kernel
2. `dotProductShared()` kernel
3. `dotProductOptimized()` kernel

## Hints

### Hint 1: Atomic Approach (Naive)

```cuda
__global__ void dotProductNaive(float *A, float *B, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float product = A[idx] * B[idx];
        atomicAdd(result, product);
    }
}
```

### Hint 2: Shared Memory Reduction

```cuda
__global__ void dotProductShared(float *A, float *B, float *partial, int n) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and multiply
    sdata[tid] = (idx < n) ? A[idx] * B[idx] : 0.0f;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}
```

### Hint 3: Warp Reduction

```cuda
__device__ void warpReduce(volatile float *sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

// In kernel:
for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}

if (tid < 32) warpReduce(sdata, tid);
```

### Hint 4: First Add During Load

For extra performance:
```cuda
int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

// Load two elements per thread
float sum = 0.0f;
if (idx < n) sum = A[idx] * B[idx];
if (idx + blockDim.x < n) sum += A[idx + blockDim.x] * B[idx + blockDim.x];

sdata[tid] = sum;
```

## Testing

The test script will:
1. Compile your code
2. Run all kernels with multiple vector sizes
3. Verify correctness against CPU reference
4. Measure and compare performance
5. Check error tolerance (1e-3 for floating point)

Run tests:
```bash
python test.py
```

## Verification

Your implementation is correct if:
1. Result matches CPU reference (within tolerance)
2. Works for various vector sizes
3. Handles non-power-of-2 sizes
4. No race conditions

## Evaluation Criteria

- **Correctness** (50%): Results match reference
- **Performance** (30%): Meets timing targets
- **Code Quality** (20%): Clean, well-commented code

## Common Mistakes to Avoid

1. **Atomic Contention**
   - All threads atomically updating same location
   - Use shared memory reduction instead

2. **Missing Synchronization**
   - Must sync after each reduction step
   - Exception: last warp doesn't need sync

3. **Boundary Conditions**
   - Vector size may not be multiple of block size
   - Pad with zeros for inactive threads

4. **Divergence**
   - Use sequential addressing, not interleaved
   - First half of threads active, not even threads

5. **Incorrect Multi-stage Reduction**
   - Must reduce partial results on host or GPU
   - Single block can't handle all data

## Bonus Challenges

1. **Multi-Element Per Thread**: Process 4-8 elements per thread
2. **Grid-Stride Loop**: Handle arbitrarily large vectors
3. **Shuffle Instructions**: Use `__shfl_down_sync()` for warp reduction
4. **Cooperative Groups**: Modern CUDA reduction patterns

## Learning Objectives

After completing this exercise, you should understand:
- Parallel reduction patterns
- Shared memory optimization
- Warp-level operations
- Avoiding thread divergence
- Multi-stage reductions

## Resources

- Example 03 in this chapter (reduction.cu)
- NVIDIA: "Optimizing Parallel Reduction in CUDA"
- CUDA Programming Guide: Warp Shuffle Functions

Good luck!
