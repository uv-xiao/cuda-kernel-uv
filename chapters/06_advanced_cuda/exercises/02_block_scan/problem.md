# Exercise 2: Block-Level Inclusive Scan

## Problem Description

Implement a block-level **inclusive prefix scan** (prefix sum) using warp primitives. Your implementation should compute the cumulative sum for all elements in a block.

## Learning Objectives

- Combine warp-level primitives with block-level coordination
- Use shared memory for inter-warp communication
- Understand the hierarchical nature of GPU algorithms

## Requirements

Implement the kernel:

```cuda
__global__ void block_scan_inclusive(int *output, const int *input, int n);
```

This kernel should:
1. Load data from global memory
2. Perform warp-level scans using `warp_scan_inclusive`
3. Coordinate between warps using shared memory
4. Write the final inclusive scan to global memory

## Algorithm Outline

1. Each warp performs an inclusive scan independently
2. The last thread of each warp writes the warp total to shared memory
3. The first warp scans the warp totals
4. Each thread adds its warp's offset to get the final result

## Example

```
Input:  [1, 1, 1, 1, 1, 1, ..., 1]  (256 elements)
Output: [1, 2, 3, 4, 5, 6, ..., 256]
```

## Constraints

- Block size: 256 threads
- Must use warp primitives for warp-level scans
- Use shared memory only for warp totals (not for main data)
- Should work for any input values (positive, negative, zero)

## Helper Function Provided

```cuda
__device__ int warp_scan_inclusive(int value) {
    for (int offset = 1; offset < warpSize; offset *= 2) {
        int neighbor = __shfl_up_sync(0xffffffff, value, offset);
        if ((threadIdx.x % warpSize) >= offset) {
            value += neighbor;
        }
    }
    return value;
}
```

## Testing

```bash
nvcc solution.cu -o block_scan
./block_scan
python test.py
```

## Bonus Challenges

1. Implement exclusive scan instead of inclusive
2. Support arbitrary block sizes (not just 256)
3. Extend to multi-block scan using atomics or multiple kernel launches
