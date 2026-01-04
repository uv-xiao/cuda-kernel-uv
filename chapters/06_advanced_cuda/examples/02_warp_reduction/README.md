# Warp Reduction Examples

This directory demonstrates warp-level reductions using shuffle instructions and compares them with traditional shared memory approaches.

## Overview

Reduction is one of the most fundamental parallel operations. Warp-level reductions using shuffle instructions offer significant performance advantages over shared memory for small-scale reductions.

## Files

- `warp_reduce.cu` - Complete reduction implementations and performance comparison
- `CMakeLists.txt` - Build configuration

## Reduction Strategies

### 1. Shared Memory Reduction (Traditional)

**Approach:**
```cuda
__shared__ int sdata[BLOCK_SIZE];
sdata[tid] = input[i];
__syncthreads();

for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
```

**Characteristics:**
- Works for any block size
- Requires shared memory allocation
- Multiple `__syncthreads()` calls
- Potential bank conflicts
- Higher latency (~20-30 cycles per load/store)

**Advantages:**
- Handles inter-warp communication
- Works across entire block
- Well-understood pattern

**Disadvantages:**
- Slower than warp primitives
- Uses limited shared memory resource
- Synchronization overhead

### 2. Warp Shuffle Reduction (Modern)

**Approach:**
```cuda
int value = input[tid];

// XOR-based butterfly reduction
for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, offset);
}
// Lane 0 now has the sum
```

**Characteristics:**
- Operates entirely in registers
- No synchronization needed (within warp)
- Very low latency (~1-2 cycles)
- No shared memory usage
- No bank conflicts possible

**Advantages:**
- Much faster than shared memory
- Minimal code
- No resource constraints
- Better instruction-level parallelism

**Disadvantages:**
- Limited to warp size (32 elements)
- Requires Compute Capability 3.0+

### 3. Hybrid Approach (Best of Both)

**For blocks larger than warp:**

```cuda
// Step 1: Each warp reduces independently using shuffle
int value = input[tid];
value = warp_reduce_sum(value);

// Step 2: Warp leaders write to shared memory
__shared__ int warp_sums[32];
if (lane_id == 0) {
    warp_sums[warp_id] = value;
}
__syncthreads();

// Step 3: First warp reduces warp results
if (warp_id == 0) {
    value = (tid < num_warps) ? warp_sums[tid] : 0;
    value = warp_reduce_sum(value);
}
```

**This combines:**
- Fast warp-level reductions (most of the work)
- Minimal shared memory (only for warp results)
- Single synchronization point

## Performance Comparison

### Typical Results (1024 elements)

```
Shared Memory: 15.2 us
Warp Shuffle:  8.7 us
Speedup:       1.75x
```

### Why is Warp Shuffle Faster?

1. **Lower Latency**: Register operations vs. memory operations
   - Shuffle: 1-2 cycles
   - Shared memory: 20-30 cycles

2. **No Synchronization Overhead**: Warps are implicitly synchronized

3. **Better Occupancy**: No shared memory pressure

4. **Fewer Instructions**: Direct register-to-register communication

## Reduction Operations

### Sum Reduction
```cuda
__device__ int warp_reduce_sum(int value) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffff, value, offset);
    }
    return value;
}
```

**Use cases:** Counting, averaging, integration

### Max Reduction
```cuda
__device__ int warp_reduce_max(int value) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        value = max(value, __shfl_xor_sync(0xffffffff, value, offset));
    }
    return value;
}
```

**Use cases:** Finding maximum value, peak detection

### Min Reduction
```cuda
__device__ int warp_reduce_min(int value) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        value = min(value, __shfl_xor_sync(0xffffffff, value, offset));
    }
    return value;
}
```

**Use cases:** Finding minimum value, threshold detection

### Custom Operations

Any **associative** operation works:
- Bitwise operations (AND, OR, XOR)
- Geometric mean (requires careful handling)
- Custom comparators

**Note:** Operation must be associative: `(a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)`

## The Butterfly Pattern

The XOR-based butterfly pattern is optimal for reductions:

```
Iteration 1 (offset=16): Pairs 16 apart exchange
  0↔16, 1↔17, 2↔18, ..., 15↔31

Iteration 2 (offset=8): Pairs 8 apart exchange
  0↔8, 1↔9, ..., 7↔15, 16↔24, ...

Iteration 3 (offset=4): Pairs 4 apart exchange
  0↔4, 1↔5, 2↔6, 3↔7, ...

Iteration 4 (offset=2): Pairs 2 apart exchange
  0↔2, 1↔3, 4↔6, 5↔7, ...

Iteration 5 (offset=1): Adjacent pairs exchange
  0↔1, 2↔3, 4↔5, 6↔7, ...

Result: Lane 0 has the final reduced value
```

**Why XOR?**
- Symmetric: Each thread knows its partner
- No branches needed
- Optimal communication pattern
- Minimal network distance (for hardware)

## Alternative: Sequential Shuffle Up

```cuda
int value = input[tid];
for (int offset = 1; offset < warpSize; offset *= 2) {
    int neighbor = __shfl_up_sync(0xffffffff, value, offset);
    if (lane_id >= offset) {
        value += neighbor;
    }
}
// Lane 31 has the sum
```

**Differences:**
- Result in lane 31 (not lane 0)
- Requires branch (if statement)
- Slightly more work for some threads
- Useful for prefix sum (scan)

## Multi-Operation Reductions

You can perform multiple reductions simultaneously with minimal overhead:

```cuda
int sum = warp_reduce_sum(value);
int max_val = warp_reduce_max(value);
int min_val = warp_reduce_min(value);
```

**Compiler optimization:**
- Shuffle operations can be pipelined
- Single pass through data
- Minimal register pressure

**Use cases:**
- Computing statistics (mean, variance, min, max)
- Bounding box calculations
- Error metrics

## When to Use Each Approach

### Use Warp Shuffle When:
- Reducing 32 or fewer elements
- Operating within a single warp
- Performance is critical
- Shared memory is limited

### Use Shared Memory When:
- Reducing more than 32 elements per block
- Need inter-warp communication
- Complex access patterns
- Compatibility with older GPUs

### Use Hybrid When:
- Block size > warp size
- Want best of both worlds
- Typical production code

## Common Patterns

### Block-Wide Reduction
```cuda
// Each warp reduces to lane 0
int value = warp_reduce_sum(my_value);

// Collect warp results
__shared__ int warp_results[32];
if (lane_id == 0) {
    warp_results[warp_id] = value;
}
__syncthreads();

// First warp reduces all warp results
if (warp_id == 0) {
    value = (tid < num_warps) ? warp_results[tid] : 0;
    value = warp_reduce_sum(value);
    if (tid == 0) {
        output[blockIdx.x] = value;
    }
}
```

### Grid-Wide Reduction
```cuda
// Reduce within block (using above pattern)
if (tid == 0) {
    atomicAdd(&global_sum, block_sum);
}
```

Or use a two-kernel approach for better performance.

## Building and Running

```bash
mkdir build && cd build
cmake ..
make
./warp_reduce
```

## Expected Output

```
========== Warp Reduction Examples ==========

Array size: 1024 elements
CPU Reference: sum=50452, max=99, min=0

1. Shared Memory Reduction:
   GPU sum: 50452, CPU sum: 50452, PASS

2. Warp Shuffle Reduction:
   GPU sum: 50452, CPU sum: 50452, PASS

3. Single Warp Reduction (32 elements):
   GPU sum: 1564, CPU sum: 1564, PASS

4. Multiple Reductions (sum, max, min):
   Sum: GPU=50452, CPU=50452, PASS
   Max: GPU=99, CPU=99, PASS
   Min: GPU=0, CPU=0, PASS

Performance Comparison (1000 iterations):
  Shared Memory: 15.234 ms (avg: 15.234 us)
  Warp Shuffle:  8.672 ms (avg: 8.672 us)
  Speedup:       1.76x

========== All Tests Completed ==========
```

## Advanced Topics

### Segmented Reductions

Reduce multiple independent segments within a warp:

```cuda
// Each segment is identified by a segment_id
unsigned int ballot = __ballot_sync(0xffffffff, segment_id == my_segment);
// Only reduce with threads in same segment
int value = my_value;
for (int offset = 1; offset < warpSize; offset *= 2) {
    int other = __shfl_up_sync(ballot, value, offset);
    if ((ballot & ((1u << lane_id) - 1)) != 0) {
        value += other;
    }
}
```

### Warp-Aggregated Atomics

Use warp reduction before atomic operation:

```cuda
// Instead of 32 atomic adds
int value = my_contribution;
value = warp_reduce_sum(value);
if (lane_id == 0) {
    atomicAdd(&global_counter, value);  // Only 1 atomic instead of 32!
}
```

**Speedup:** 10-30x reduction in atomic contention

## References

- [Faster Parallel Reductions on Kepler](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/)
- [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- [CUDA C++ Programming Guide - Warp Shuffle](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions)
