# Example 03: Parallel Reduction

## Overview

Parallel reduction is a fundamental pattern in parallel computing. This example demonstrates how to efficiently sum an array on the GPU, showcasing progressive optimization techniques.

## The Reduction Problem

**Goal**: Sum all elements of an array

```
Input:  [1, 2, 3, 4, 5, 6, 7, 8]
Output: 36
```

**Challenge**: This is inherently sequential on CPU but can be parallelized using a tree-based approach.

## Tree-Based Parallel Reduction

```
Level 0: [1] [2] [3] [4] [5] [6] [7] [8]
         ↓ ↓   ↓ ↓   ↓ ↓   ↓ ↓
Level 1:  [3]   [7]  [11]  [15]
           ↓ ↓         ↓ ↓
Level 2:    [10]       [26]
             ↓ ↓
Level 3:      [36]
```

**Complexity**: O(log n) steps instead of O(n)

## Optimization Progression

This example implements 5 versions, each improving upon the previous:

### Version 1: Interleaved Addressing (Naive)

```cuda
for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
```

**Problems:**
- **Warp divergence**: `tid % (2*s) == 0` causes half the threads to be idle
- Only threads with even IDs participate initially
- Performance: ~40% of peak

**Example with 8 threads:**
```
Step 1: threads 0,2,4,6 active (divergent warp)
Step 2: threads 0,4 active
Step 3: thread 0 active
```

### Version 2: Sequential Addressing

```cuda
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
```

**Improvements:**
- **No divergence**: First half of threads always active together
- Better warp utilization
- Performance: ~2x faster than V1

**Example with 8 threads:**
```
Step 1: threads 0-3 active (no divergence)
Step 2: threads 0-1 active
Step 3: thread 0 active
```

### Version 3: First Add During Load

```cuda
int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

// Load and add two elements
sdata[tid] = g_data[idx] + g_data[idx + blockDim.x];
```

**Improvements:**
- **Half the blocks**: Each thread loads 2 elements
- Reduces kernel launch overhead
- Better utilization of memory bandwidth
- Performance: ~1.5x faster than V2

**Impact:**
- Before: 16M elements → 65,536 blocks
- After: 16M elements → 32,768 blocks

### Version 4: Unroll Last Warp

```cuda
for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}

// Last warp - no sync needed
if (tid < 32) warpReduce(sdata, tid);
```

```cuda
__device__ void warpReduce(volatile float *sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}
```

**Improvements:**
- **No synchronization in last warp**: Threads in same warp execute in lockstep
- `volatile` prevents compiler optimization
- Eliminates 6 synchronization barriers
- Performance: ~1.2x faster than V3

**Why it works:**
- Warp = 32 threads executing simultaneously
- Within a warp, operations are implicitly synchronized
- No need for `__syncthreads()` when s < 32

### Version 5: Multiple Elements Per Thread

```cuda
float sum = 0.0f;
for (int i = idx; i < n; i += gridSize) {
    sum += g_data[i];
    if (i + blockDim.x < n) {
        sum += g_data[i + blockDim.x];
    }
}
sdata[tid] = sum;
```

**Improvements:**
- **Grid-stride loop**: Each thread processes multiple elements
- Increases arithmetic intensity
- Reduces number of kernel launches
- Better instruction-level parallelism
- Performance: ~1.5x faster than V4

**Example:**
- Array: 16M elements
- Blocks: 1024, Threads: 256
- Grid size: 1024 * 256 * 2 = 524,288
- Each thread processes: 16M / 524K ≈ 31 elements

## Performance Analysis

### Typical Results (RTX 3080)

| Version | Time (ms) | Speedup | Bandwidth (GB/s) |
|---------|-----------|---------|------------------|
| V1: Interleaved | 0.850 | 1.0x | 75 |
| V2: Sequential | 0.425 | 2.0x | 150 |
| V3: First Add | 0.280 | 3.0x | 228 |
| V4: Unroll Warp | 0.235 | 3.6x | 272 |
| V5: Multi-Element | 0.156 | 5.4x | 410 |

### Optimization Impact

```
Improvement breakdown:
- Sequential addressing: 2.0x
- First add during load: 1.5x
- Unroll last warp: 1.2x
- Multiple elements: 1.5x
Total: ~5.4x improvement
```

## Key Concepts

### 1. Warp Divergence

**Problem:**
```cuda
if (tid % 2 == 0) {  // Divergent!
    // Even threads
} else {
    // Odd threads
}
```

Threads in a warp execute in lockstep. Divergent branches serialize execution.

**Solution:**
```cuda
if (tid < half) {  // No divergence
    // First half of threads
}
```

### 2. Shared Memory Pattern

```cuda
__shared__ float sdata[256];

// Load
sdata[tid] = g_data[idx];
__syncthreads();

// Reduce
for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
}

// Write result
if (tid == 0) g_out[blockIdx.x] = sdata[0];
```

### 3. Warp-Level Primitives

Modern CUDA provides shuffle instructions for warp-level reduction:

```cuda
__inline__ __device__
float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
```

### 4. Recursive Reduction

For large arrays, reduce in stages:
1. Each block reduces to one value
2. Recursively reduce block results
3. Continue until single value remains

## Common Pitfalls

### 1. Missing Synchronization

```cuda
// WRONG - race condition
sdata[tid] = data[idx];
float val = sdata[tid + 1];  // May read uninitialized data

// CORRECT
sdata[tid] = data[idx];
__syncthreads();
float val = sdata[tid + 1];
```

### 2. Too Many Synchronizations

```cuda
// WRONG - unnecessary sync in last warp
for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();  // Not needed when s < 32
}

// CORRECT
for (int s = blockDim.x/2; s > 32; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
}
if (tid < 32) warpReduce(sdata, tid);
```

### 3. Incorrect Boundary Handling

```cuda
// WRONG - may access out of bounds
sdata[tid] = g_data[idx] + g_data[idx + blockDim.x];

// CORRECT
sdata[tid] = (idx < n) ? g_data[idx] : 0.0f;
if (idx + blockDim.x < n) sdata[tid] += g_data[idx + blockDim.x];
```

## Building and Running

```bash
mkdir build && cd build
cmake ..
make

./reduction
```

## Expected Output

```
=== CUDA Reduction Optimization ===

Array size: 16777216 elements (64.00 MB)

Computing reference result on CPU...
CPU result: 8388019.500000 (245.67 ms)

Version 1: Interleaved Addressing (divergent warps)
----------------------------------------------------
Result: 8388019.500000
Error: 0.000000%
Time: 0.850 ms
Speedup vs CPU: 289.02x

Version 2: Sequential Addressing (no divergence)
-------------------------------------------------
Result: 8388019.500000
Error: 0.000000%
Time: 0.425 ms
Speedup vs V1: 2.00x

...

=== Performance Summary ===
Overall speedup vs CPU: 1575.26x
```

## Advanced Topics

### CUB Library

NVIDIA's CUB library provides highly optimized reduction:

```cuda
#include <cub/cub.cuh>

cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n);
```

### Thrust Library

Even simpler high-level interface:

```cuda
#include <thrust/reduce.h>

float sum = thrust::reduce(d_vec.begin(), d_vec.end());
```

### Cooperative Groups

Modern CUDA provides cooperative groups for cleaner code:

```cuda
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);
val = cg::reduce(tile32, val, cg::plus<float>());
```

## Next Steps

- Implement other reduction operations (max, min, product)
- Try segmented reduction (reduce multiple arrays)
- Study atomic-based reduction for irregular patterns
- Profile with Nsight Compute to understand bottlenecks

## References

- Mark Harris: "Optimizing Parallel Reduction in CUDA"
- CUDA Toolkit Documentation: CUB Library
- NVIDIA Blog: "Cooperative Groups"
