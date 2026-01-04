# Exercise 1: Warp-Level Max Reduction

## Problem Description

Implement a warp-level **max reduction** using shuffle instructions. Your implementation should find the maximum value across all 32 threads in a warp using only warp primitives (no shared memory).

## Learning Objectives

- Understand how to use `__shfl_xor_sync` for reduction patterns
- Practice implementing associative operations with shuffle
- Compare performance with shared memory approaches

## Requirements

Implement the device function:

```cuda
__device__ int warp_reduce_max(int value);
```

This function should:
1. Accept an integer value from each thread in a warp
2. Use `__shfl_xor_sync` to perform a butterfly reduction
3. Return the maximum value across all 32 threads
4. The result should be valid in **all threads** (not just lane 0)

## Constraints

- Must use only `__shfl_xor_sync` (no other shuffle variants)
- No shared memory allowed
- No atomic operations
- Must work for negative numbers
- Should complete in O(log₂ 32) = 5 iterations

## Example

```
Input (per thread):  [5, 2, 9, 1, 7, 3, 8, 4, 6, 0, ...]
Output (all threads): 9
```

## Hints

1. The butterfly pattern with XOR works for any associative operation
2. For max reduction, use: `value = max(value, __shfl_xor_sync(...))`
3. Start with offset = 16, then 8, 4, 2, 1
4. The max operation is associative: `max(max(a,b),c) = max(a,max(b,c))`

## Testing

Run the test script to verify your implementation:

```bash
nvcc solution.cu -o warp_reduce_max
./warp_reduce_max
python test.py
```

## Bonus Challenges

1. Implement `warp_reduce_min` similarly
2. Implement `warp_reduce_argmax` that returns both the max value and its lane ID
3. Measure performance difference vs. shared memory for 32 elements
