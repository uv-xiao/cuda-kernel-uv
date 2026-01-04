# Exercise 1: Simple Grouped GEMM

## Objective

Implement a basic grouped GEMM kernel that processes multiple matrix multiplications with variable sizes in a single kernel launch. This exercise builds understanding of:
- Handling variable-size workloads
- Dynamic work distribution
- Efficient memory access patterns

## Background

Grouped GEMM is essential for Mixture-of-Experts (MoE) models where different "experts" receive different numbers of tokens. Traditional batched GEMM requires padding all groups to the same size, wasting compute. Your task is to implement a kernel that processes variable-size groups efficiently.

## Problem Statement

Implement a CUDA kernel with the following signature:

```cuda
__global__ void grouped_gemm_kernel(
    const float* A_concat,      // Concatenated input matrices
    const float* B_concat,      // Concatenated weight matrices
    float* C_concat,            // Concatenated output matrices
    const int* group_offsets,   // Cumulative offsets for A and C (length: num_groups+1)
    const int* M_sizes,         // M dimension for each group (length: num_groups)
    int K,                      // Shared K dimension
    int N,                      // Shared N dimension
    int num_groups              // Number of groups
);
```

## Input Format

### A_concat (Input matrices)
Groups are concatenated along the M dimension:
```
Group 0: M[0] x K
Group 1: M[1] x K
Group 2: M[2] x K
...
Total: (M[0] + M[1] + M[2] + ...) x K
```

### B_concat (Weight matrices)
Each group has its own weight matrix:
```
Group 0: K x N
Group 1: K x N
Group 2: K x N
...
Total: num_groups * (K x N)
```

### group_offsets
Cumulative sum of M sizes:
```
group_offsets = [0, M[0], M[0]+M[1], M[0]+M[1]+M[2], ...]
```

## Requirements

### Functional Requirements
1. Compute `C[i] = A[i] @ B[i]` for each group i
2. Handle variable M sizes (K and N are constant across groups)
3. Use shared memory tiling for performance
4. Support at least tile sizes of 16x16 or 32x32

### Performance Requirements
1. Achieve at least 60% of single-matrix cuBLAS performance
2. Handle groups with M ranging from 1 to 1024
3. Minimize warp divergence

### Constraints
1. Use only CUDA built-in functions (no CUTLASS or external libraries)
2. Tile size must be a power of 2
3. Handle edge cases (M < tile size, empty groups)

## Test Cases

Your implementation will be tested with:

### Test 1: Uniform sizes
```
num_groups = 4
M_sizes = [128, 128, 128, 128]
K = 256, N = 256
```

### Test 2: Variable sizes
```
num_groups = 8
M_sizes = [64, 128, 32, 256, 16, 192, 96, 48]
K = 512, N = 512
```

### Test 3: Extreme imbalance
```
num_groups = 4
M_sizes = [1, 10, 100, 1000]
K = 1024, N = 1024
```

### Test 4: Small groups
```
num_groups = 16
M_sizes = [8, 12, 16, 24, 32, 8, 16, 12, 20, 28, 16, 8, 24, 32, 16, 12]
K = 128, N = 128
```

## Hints

### Hint 1: Work Distribution
Consider how to map thread blocks to tiles across all groups:
```cuda
// One approach: tile ID encodes both group and tile position
int global_tile_id = blockIdx.x;

// Find which group this tile belongs to
int group_id = find_group_from_tile_id(global_tile_id, ...);
int local_tile_id = tile_id_within_group(global_tile_id, ...);
```

### Hint 2: Shared Memory Tiling
Standard GEMM tiling applies within each group:
```cuda
__shared__ float As[TILE_SIZE][TILE_SIZE];
__shared__ float Bs[TILE_SIZE][TILE_SIZE];

// Load tiles, compute partial products, accumulate...
```

### Hint 3: Handling Variable M
Some threads may need to handle tiles that extend beyond M:
```cuda
if (row >= M_current_group) {
    // This thread has no work in this tile
    As[ty][tx] = 0.0f;  // Zero out shared memory
}
```

### Hint 4: Group Finding
Precompute or use binary search to find which group a tile belongs to:
```cuda
// Linear search (simple but slow for many groups)
int group_id = 0;
for (int g = 0; g < num_groups; g++) {
    if (tile_id < tiles_in_group[g]) {
        group_id = g;
        break;
    }
    tile_id -= tiles_in_group[g];
}
```

## Evaluation Criteria

Your solution will be evaluated on:

1. **Correctness (40%)**
   - Numerically correct output (max error < 1e-3)
   - Handles all edge cases
   - No race conditions or synchronization bugs

2. **Performance (40%)**
   - Throughput (TFLOPS) on test cases
   - Efficiency vs theoretical peak
   - Scaling with number of groups

3. **Code Quality (20%)**
   - Clear, readable code
   - Good comments explaining key decisions
   - Proper error handling

## Starter Code

See `starter.cu` for a skeleton implementation with:
- Host function to launch kernel
- Test harness
- Reference CPU implementation for validation

## Solution

A reference solution is provided in `solution.cu`. Try to implement your own version first before looking at the solution!

## Extensions (Optional)

Once you have a working implementation, try these challenges:

1. **Persistent Kernel:** Keep thread blocks alive across multiple groups
2. **Work Stealing:** Dynamic task assignment for better load balancing
3. **FP8 Support:** Extend to support FP8 inputs with scaling
4. **Fused Epilogue:** Add activation function (e.g., ReLU, GELU) in-kernel

## Submission

Submit your implementation as `student_solution.cu` with:
- Complete kernel implementation
- Brief explanation of your approach (as comments)
- Any optimizations you applied

Good luck!
