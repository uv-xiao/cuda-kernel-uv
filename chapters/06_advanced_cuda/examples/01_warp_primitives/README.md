# Warp Primitives Examples

This directory contains examples demonstrating CUDA warp-level primitives for efficient intra-warp communication.

## Overview

Warp primitives allow threads within a warp (32 threads on all current NVIDIA GPUs) to communicate and synchronize efficiently without using shared memory. These operations are extremely fast and enable high-performance parallel algorithms.

## Files

- `warp_shuffle.cu` - Demonstrates all shuffle operations and practical applications
- `warp_vote.cu` - Demonstrates vote operations and use cases
- `CMakeLists.txt` - Build configuration

## Warp Shuffle Operations

### `__shfl_sync(mask, var, srcLane)`

Broadcasts a value from a specific lane to all threads in the warp.

**Parameters:**
- `mask`: Warp mask (usually `0xffffffff` for full warp)
- `var`: Variable to read
- `srcLane`: Source lane ID (0-31)

**Use cases:**
- Broadcasting a value (e.g., reading a shared configuration)
- Implementing leader election
- Distributing work from a single thread

**Example:**
```cuda
int value = __shfl_sync(0xffffffff, my_value, 0);
// All threads now have the value from lane 0
```

### `__shfl_up_sync(mask, var, delta)`

Each thread reads from a lane with a lower ID (lane_id - delta).

**Parameters:**
- `delta`: Offset to read from (positive integer)

**Behavior:**
- Threads with `lane_id < delta` receive their own value (unchanged)
- Other threads read from `lane_id - delta`

**Use cases:**
- Prefix sum (scan) operations
- Dependency chains where each element depends on previous
- Ring/circular buffer patterns

**Example:**
```cuda
// Shift right: each thread gets value from previous lane
int shifted = __shfl_up_sync(0xffffffff, my_value, 1);
```

### `__shfl_down_sync(mask, var, delta)`

Each thread reads from a lane with a higher ID (lane_id + delta).

**Parameters:**
- `delta`: Offset to read from (positive integer)

**Behavior:**
- Threads with `lane_id >= (warpSize - delta)` receive their own value
- Other threads read from `lane_id + delta`

**Use cases:**
- Reverse prefix sum
- Looking ahead in data streams
- Pipelining operations

**Example:**
```cuda
// Shift left: each thread gets value from next lane
int shifted = __shfl_down_sync(0xffffffff, my_value, 1);
```

### `__shfl_xor_sync(mask, var, laneMask)`

Each thread reads from lane (lane_id XOR laneMask).

**Parameters:**
- `laneMask`: Bitmask to XOR with lane ID

**Behavior:**
- Creates butterfly exchange patterns
- Thread i exchanges with thread (i XOR laneMask)

**Use cases:**
- Tree-based reductions (sum, max, min)
- Fast Fourier Transform (FFT)
- Butterfly communication patterns
- Parallel sorting networks

**Example:**
```cuda
// Exchange with lane 16 apart (swap upper/lower halves)
int partner_value = __shfl_xor_sync(0xffffffff, my_value, 16);
```

**Reduction pattern:**
```cuda
// Sum all values in warp using butterfly pattern
int value = my_value;
for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, offset);
}
// Lane 0 now has the sum
```

## Warp Vote Operations

### `__all_sync(mask, predicate)`

Returns true if the predicate is true for ALL threads in the warp.

**Use cases:**
- Checking convergence (all threads done)
- Validating invariants (all values in range)
- Early termination conditions
- Ensuring consistency across threads

**Example:**
```cuda
if (__all_sync(0xffffffff, error == 0)) {
    // All threads succeeded, proceed
}
```

### `__any_sync(mask, predicate)`

Returns true if the predicate is true for ANY thread in the warp.

**Use cases:**
- Early detection (any thread found result)
- Error propagation (any thread failed)
- Existence checks (any element matches)
- Load balancing decisions

**Example:**
```cuda
if (__any_sync(0xffffffff, found_target)) {
    // At least one thread found it, warp can exit
}
```

### `__ballot_sync(mask, predicate)`

Returns a 32-bit bitmask where bit i is set if the predicate is true for lane i.

**Use cases:**
- Stream compaction (filtering)
- Counting matching elements
- Building indices of active threads
- Conflict detection
- Sparse operations

**Example:**
```cuda
unsigned int ballot = __ballot_sync(0xffffffff, value > threshold);
int count = __popc(ballot); // Count how many threads passed
```

**Advanced pattern - compute position in output:**
```cuda
unsigned int ballot = __ballot_sync(0xffffffff, keep_element);
unsigned int mask = (1u << lane_id) - 1; // Mask for lanes before me
int position = __popc(ballot & mask); // My output position
```

## Performance Characteristics

### Latency
- Shuffle operations: ~1-2 clock cycles
- Vote operations: ~1 clock cycle
- Much faster than shared memory (~20-30 cycles)

### Advantages over Shared Memory
1. **No memory transactions**: Pure register operations
2. **No bank conflicts**: Not accessing memory
3. **Lower latency**: Direct register-to-register
4. **No synchronization needed**: Within warp, execution is synchronous
5. **Less code**: More concise than shared memory patterns

### When to Use Warp Primitives
- **Small reductions**: 32 elements or fewer
- **Intra-warp communication**: Data exchange within warp
- **Vote operations**: Quick decisions across warp
- **Prefix sums**: Within a warp

### When to Use Shared Memory
- **Larger reductions**: More than 32 elements
- **Inter-warp communication**: Between different warps
- **Complex access patterns**: Random access, reuse
- **Data staging**: Coalescing global memory accesses

## Building and Running

```bash
mkdir build && cd build
cmake ..
make

# Run shuffle examples
./warp_shuffle

# Run vote examples
./warp_vote
```

## Example Output

### Shuffle Operations
```
1. __shfl_sync (Broadcast from lane 0):
   Input : 0 1 2 3 4 5 ... 31
   Output: 0 0 0 0 0 0 ... 0
   All threads should have value 0 (broadcasted from lane 0)

2. __shfl_up_sync (delta = 1):
   Input : 0 1 2 3 4 5 ... 31
   Output: 0 0 1 2 3 4 ... 30
   Each thread reads from previous lane (lane 0 unchanged)
```

### Vote Operations
```
1. __all_sync - All threads above threshold:
   Threshold: 5, All values: 10
   Warp 0: __all_sync = 1 (expected: 1)

5. __ballot_sync - Alternating pattern:
   Threshold: 5, Even lanes: 10, Odd lanes: 3
   Warp 0 ballot: 01010101 01010101 01010101 01010101 (0x55555555)
   Count: 16 threads above threshold
```

## Common Patterns

### Warp Reduction (Sum)
```cuda
int value = input[tid];
for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, offset);
}
if (lane_id == 0) {
    output[warp_id] = value;
}
```

### Warp Prefix Sum
```cuda
int value = input[tid];
for (int offset = 1; offset < warpSize; offset *= 2) {
    int neighbor = __shfl_up_sync(0xffffffff, value, offset);
    if (lane_id >= offset) {
        value += neighbor;
    }
}
output[tid] = value;
```

### Stream Compaction
```cuda
int keep = (input[tid] > threshold);
unsigned int ballot = __ballot_sync(0xffffffff, keep);
unsigned int mask = (1u << lane_id) - 1;
int pos = __popc(ballot & mask);
if (keep) {
    output[base + pos] = input[tid];
}
```

## Requirements

- CUDA Compute Capability 3.0+ (for shuffle operations)
- CUDA Compute Capability 7.0+ (for vote operations with masks)
- Modern NVIDIA GPU (Kepler architecture or newer)

## References

- [CUDA C++ Programming Guide - Warp Shuffle](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions)
- [CUDA C++ Programming Guide - Warp Vote](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-vote-functions)
- [Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
- [Faster Parallel Reductions on Kepler](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/)
