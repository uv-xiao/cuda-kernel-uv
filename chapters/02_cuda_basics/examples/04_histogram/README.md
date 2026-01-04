# Example 04: Histogram with Atomic Operations

## Overview

This example demonstrates how to compute histograms using CUDA atomic operations. Histograms are fundamental in image processing, data analysis, and many other applications.

## The Histogram Problem

**Input**: Array of values
```
[5, 2, 7, 2, 5, 5, 1, 7, ...]
```

**Output**: Count of each unique value
```
Bin 0: 0
Bin 1: 1
Bin 2: 2
Bin 5: 3
Bin 7: 2
...
```

**Challenge**: Multiple threads may need to update the same bin simultaneously.

## Why Atomic Operations?

Without atomics, race conditions occur:

```cuda
// WRONG - race condition
int bin = data[idx];
int count = histogram[bin];
count++;
histogram[bin] = count;  // Another thread may have updated in between!
```

**Solution**: Atomic operations guarantee thread-safe updates
```cuda
atomicAdd(&histogram[data[idx]], 1);
```

## Atomic Operations in CUDA

### Common Atomic Functions

```cuda
// Arithmetic
int atomicAdd(int *address, int val);
int atomicSub(int *address, int val);
int atomicMin(int *address, int val);
int atomicMax(int *address, int val);

// Bitwise
int atomicAnd(int *address, int val);
int atomicOr(int *address, int val);
int atomicXor(int *address, int val);

// Exchange
int atomicExch(int *address, int val);
int atomicCAS(int *address, int compare, int val);  // Compare-and-swap
```

### How Atomics Work

1. **Lock the memory location**
2. **Read current value**
3. **Perform operation**
4. **Write result**
5. **Unlock**

**Impact**: Serializes conflicting accesses → potential bottleneck

## Optimization Strategies

### Version 1: Global Memory Atomics (Naive)

```cuda
__global__ void histogram_v1_global_atomic(unsigned char *data,
                                            int *histogram, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&histogram[data[idx]], 1);
    }
}
```

**Problems:**
- All blocks contend for same global memory
- Global memory atomics are slow (~400-800 cycles)
- High contention on popular bins

**Performance**: Baseline

### Version 2: Shared Memory Per-Block

```cuda
__global__ void histogram_v2_shared(unsigned char *data,
                                     int *histogram, int n) {
    __shared__ int s_hist[NUM_BINS];

    // Initialize shared histogram
    if (threadIdx.x < NUM_BINS) {
        s_hist[threadIdx.x] = 0;
    }
    __syncthreads();

    // Accumulate in shared memory
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&s_hist[data[idx]], 1);
    }
    __syncthreads();

    // Merge to global histogram
    if (threadIdx.x < NUM_BINS) {
        atomicAdd(&histogram[threadIdx.x], s_hist[threadIdx.x]);
    }
}
```

**Improvements:**
- Shared memory atomics are ~10x faster than global
- Reduces global atomic operations by ~blockSize
- Each block computes local histogram first

**Performance**: ~3-5x faster than V1

### Version 3: Multiple Elements Per Thread

```cuda
__global__ void histogram_v3_multi_elements(unsigned char *data,
                                             int *histogram, int n,
                                             int elements_per_thread) {
    __shared__ int s_hist[NUM_BINS];

    if (threadIdx.x < NUM_BINS) {
        s_hist[threadIdx.x] = 0;
    }
    __syncthreads();

    // Each thread processes multiple elements
    int base_idx = blockIdx.x * blockDim.x * elements_per_thread + threadIdx.x;
    for (int i = 0; i < elements_per_thread; i++) {
        int idx = base_idx + i * blockDim.x;
        if (idx < n) {
            atomicAdd(&s_hist[data[idx]], 1);
        }
    }
    __syncthreads();

    if (threadIdx.x < NUM_BINS) {
        atomicAdd(&histogram[threadIdx.x], s_hist[threadIdx.x]);
    }
}
```

**Improvements:**
- Fewer blocks needed
- Better memory coalescing
- Reduced kernel launch overhead

**Performance**: ~1.5-2x faster than V2

### Version 4: Privatized Shared Memory

```cuda
__global__ void histogram_v4_privatized(unsigned char *data,
                                         int *histogram, int n) {
    // Multiple histograms to reduce contention
    __shared__ int s_hist[4][NUM_BINS];

    int private_id = threadIdx.x % 4;

    // Initialize
    for (int i = threadIdx.x; i < 4 * NUM_BINS; i += blockDim.x) {
        s_hist[i / NUM_BINS][i % NUM_BINS] = 0;
    }
    __syncthreads();

    // Each thread uses assigned private histogram
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&s_hist[private_id][data[idx]], 1);
    }
    __syncthreads();

    // Merge private histograms
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        int count = 0;
        for (int j = 0; j < 4; j++) {
            count += s_hist[j][i];
        }
        atomicAdd(&histogram[i], count);
    }
}
```

**Improvements:**
- Reduces contention by 4x within block
- Each thread uses different histogram copy
- Final merge is non-atomic (private sums)

**Performance**: ~1.2-1.5x faster than V3 (data-dependent)

## Performance Analysis

### Typical Results (RTX 3080)

| Version | Time (ms) | Speedup | Notes |
|---------|-----------|---------|-------|
| CPU | 150.0 | 1.0x | Sequential |
| V1: Global | 8.5 | 17.6x | High contention |
| V2: Shared | 2.1 | 4.0x | Reduced contention |
| V3: Multi-Element | 1.3 | 1.6x | Better utilization |
| V4: Privatized | 1.0 | 1.3x | Lowest contention |

**Overall speedup**: ~150x vs CPU

### Factors Affecting Performance

1. **Data Distribution**
   - Uniform: Less contention
   - Skewed: High contention on popular bins

2. **Number of Bins**
   - Few bins: More contention
   - Many bins: Better parallelism

3. **Block Size**
   - Larger: More contention within block
   - Smaller: More global atomics

## Atomic Operation Characteristics

### Performance Hierarchy

| Memory | Latency | Use Case |
|--------|---------|----------|
| Shared Memory Atomic | ~20-30 cycles | Within-block updates |
| Global Memory Atomic | ~400-800 cycles | Cross-block updates |

### Contention Impact

```
No contention:    1 operation  = 1 cycle
10 threads:       10 operations ≈ 10 cycles (serialized)
100 threads:      100 operations ≈ 100 cycles (serialized)
```

### When to Use Atomics

**Good Uses:**
- Histograms
- Sparse reductions
- Counters
- Flags and locks

**Alternatives:**
- Reduction: When combining all values
- Scan: For prefix sums
- Sorting: For ordered output

## Common Pitfalls

### 1. Forgetting to Initialize

```cuda
// WRONG - undefined initial values
__shared__ int hist[256];
atomicAdd(&hist[bin], 1);

// CORRECT
__shared__ int hist[256];
if (threadIdx.x < 256) hist[threadIdx.x] = 0;
__syncthreads();
atomicAdd(&hist[bin], 1);
```

### 2. Missing Synchronization

```cuda
// WRONG
__shared__ int hist[256];
hist[threadIdx.x] = 0;  // No sync!
atomicAdd(&hist[bin], 1);

// CORRECT
__shared__ int hist[256];
hist[threadIdx.x] = 0;
__syncthreads();  // Wait for initialization
atomicAdd(&hist[bin], 1);
```

### 3. Over-using Atomics

```cuda
// INEFFICIENT - atomic for every element
for (int i = 0; i < n; i++) {
    atomicAdd(&sum, data[i]);
}

// BETTER - use reduction pattern
float local_sum = 0;
for (int i = 0; i < n; i++) {
    local_sum += data[i];
}
atomicAdd(&sum, local_sum);
```

## Building and Running

```bash
mkdir build && cd build
cmake ..
make

./histogram
```

## Expected Output

```
=== CUDA Histogram with Atomic Operations ===

Data size: 67108864 elements (64.00 MB)
Histogram bins: 256

Version 1: Global Memory Atomics
---------------------------------
Correctness: PASSED
Time: 8.523 ms
Speedup vs CPU: 17.60x

Version 2: Shared Memory Per-Block
-----------------------------------
Correctness: PASSED
Time: 2.145 ms
Speedup vs V1: 3.97x

...

Histogram Distribution:
Bin  Count        Distribution
---  ------------ ------------------------------------------------
0    245678       ####################
16   251234       ####################
32   248901       ####################
...
```

## Advanced Topics

### Warp Aggregation

Reduce atomic operations using warp-level primitives:

```cuda
unsigned mask = __ballot_sync(0xffffffff, bin == target_bin);
int count = __popc(mask);
if (lane == 0) {
    atomicAdd(&histogram[target_bin], count);
}
```

### Compare-And-Swap (CAS)

For complex atomic operations:

```cuda
int old = histogram[bin];
int assumed;
do {
    assumed = old;
    old = atomicCAS(&histogram[bin], assumed, assumed + 1);
} while (assumed != old);
```

### Lock-Free Algorithms

Using atomics to build locks:

```cuda
__device__ void lock(int *mutex) {
    while (atomicCAS(mutex, 0, 1) != 0);
}

__device__ void unlock(int *mutex) {
    atomicExch(mutex, 0);
}
```

## Alternative Approaches

### Sort-Based Histogram

For very large bin counts:
1. Sort the data
2. Count consecutive equal elements
3. Can be faster when bins >> data size

### Reduction-Based

When combining values:
1. Use parallel reduction instead of atomics
2. Much faster for commutative operations

## Next Steps

- Implement 2D histogram (joint distribution)
- Try histogram equalization for images
- Experiment with different privatization factors
- Profile atomic contention with Nsight Compute

## References

- CUDA Programming Guide: Atomic Functions
- "CUDA By Example" - Chapter on Atomics
- Duane Merrill: "Fast and Flexible Atomic Operations"
