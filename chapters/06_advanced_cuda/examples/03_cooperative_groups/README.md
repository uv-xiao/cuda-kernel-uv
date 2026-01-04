# Cooperative Groups Examples

This directory demonstrates the Cooperative Groups API for flexible thread synchronization and communication in CUDA.

## Overview

Cooperative Groups is a CUDA programming model that provides explicit, flexible, and safe thread synchronization primitives. It extends beyond the traditional block-level synchronization to support sub-block and grid-level operations.

## Files

- `cg_basics.cu` - Introduction to cooperative groups API
- `grid_sync.cu` - Grid-wide synchronization for multi-phase algorithms
- `CMakeLists.txt` - Build configuration

## What are Cooperative Groups?

Cooperative Groups provide a way to explicitly define and synchronize groups of threads at various granularities:

1. **Thread Block** (`thread_block`) - Traditional CUDA block
2. **Thread Block Tile** (`thread_block_tile<Size>`) - Sub-block groups (e.g., warps)
3. **Coalesced Group** (`coalesced_group`) - Active threads after divergence
4. **Grid Group** (`grid_group`) - All threads in entire kernel

## Key Benefits

### 1. More Expressive Code

**Traditional CUDA:**
```cuda
__syncthreads();
int value = __shfl_down_sync(0xffffffff, my_value, offset);
```

**Cooperative Groups:**
```cuda
auto block = cg::this_thread_block();
block.sync();

auto warp = cg::tiled_partition<32>(block);
int value = warp.shfl_down(my_value, offset);
```

### 2. Safer Synchronization

- Explicit group membership
- Compile-time checks where possible
- Clear intent in code

### 3. Flexible Grouping

- Create custom group sizes
- Handle divergent execution elegantly
- Support for grid-wide operations

### 4. Better Performance

- Compiler can optimize based on group information
- Reduced implicit synchronization
- More efficient warp-level operations

## Thread Block Operations

```cuda
cg::thread_block block = cg::this_thread_block();

// Thread identification
unsigned int tid = block.thread_rank();          // threadIdx.x
unsigned int bid = block.group_index().x;        // blockIdx.x
unsigned int block_size = block.size();          // blockDim.x

// Synchronization
block.sync();  // Same as __syncthreads()

// Dimensions
dim3 block_dim = block.group_dim();              // blockDim
dim3 block_idx = block.group_index();            // blockIdx
```

**When to use:**
- Replacement for traditional `__syncthreads()`
- Clearer code intent
- Integration with other CG features

## Thread Block Tiles

### Warp-sized Tiles (32 threads)

```cuda
cg::thread_block block = cg::this_thread_block();
cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

// Warp operations
int rank = warp.thread_rank();     // 0-31
int size = warp.size();            // 32

// Shuffle operations
int value = warp.shfl(my_value, src_lane);
int value = warp.shfl_up(my_value, delta);
int value = warp.shfl_down(my_value, delta);
int value = warp.shfl_xor(my_value, mask);

// Vote operations
bool all = warp.all(predicate);
bool any = warp.any(predicate);
auto ballot = warp.ballot(predicate);

// Synchronization
warp.sync();
```

**Advantages:**
- Cleaner than raw shuffle/vote intrinsics
- No need for explicit masks (0xffffffff)
- Type-safe operations
- Better documentation through types

### Custom Tile Sizes

```cuda
// 16-thread tiles
cg::thread_block_tile<16> tile16 = cg::tiled_partition<16>(block);

// 8-thread tiles
cg::thread_block_tile<8> tile8 = cg::tiled_partition<8>(block);

// 4-thread tiles
cg::thread_block_tile<4> tile4 = cg::tiled_partition<4>(block);
```

**Use cases:**
- SIMD-like operations (e.g., 4-wide vector ops)
- Algorithm-specific groupings
- Fine-grained parallelism control

**Meta-group information:**
```cuda
// How many tiles of this size in the block?
unsigned int meta_size = tile.meta_group_size();

// Which tile am I in?
unsigned int meta_rank = tile.meta_group_rank();
```

## Coalesced Groups

Handle divergent execution gracefully:

```cuda
if (condition) {  // Some threads take this branch
    cg::coalesced_group active = cg::coalesced_threads();

    // Only active threads participate
    int count = active.size();         // How many threads are active?
    int rank = active.thread_rank();   // My rank among active threads

    // Operations among active threads only
    int value = active.shfl(my_value, src);
    bool all = active.all(predicate);
}
```

**Applications:**
- Sparse data processing
- Dynamic load balancing
- Stream compaction
- Handling irregular workloads

**Example - compact active threads:**
```cuda
if (my_data != 0) {  // Sparse data
    cg::coalesced_group active = cg::coalesced_threads();

    // Compute output position based on active threads
    int output_pos = base_pos + active.thread_rank();
    output[output_pos] = my_data;
}
```

## Grid-Wide Synchronization

Synchronize ALL threads across ALL blocks in the entire kernel:

```cuda
__global__ void multi_phase_kernel(...) {
    cg::grid_group grid = cg::this_grid();

    // Phase 1: Compute something
    phase1_computation();

    grid.sync();  // ALL threads across ALL blocks wait here

    // Phase 2: Use results from phase 1
    phase2_computation();

    grid.sync();

    // Phase 3: Use results from phase 2
    phase3_computation();
}
```

### Requirements

1. **Hardware:** Compute Capability 6.0 or higher
2. **Launch API:** Must use `cudaLaunchCooperativeKernel()`
3. **Block limit:** Cannot exceed device's max resident blocks

**Checking support:**
```cuda
cudaDeviceProp props;
cudaGetDeviceProperties(&props, device);

if (props.cooperativeLaunch) {
    // Grid sync is supported
}
```

**Launching cooperative kernels:**
```cuda
void *args[] = {&arg1, &arg2, &arg3};

cudaLaunchCooperativeKernel(
    (void*)my_kernel,
    gridDim,
    blockDim,
    args,
    sharedMem,
    stream
);
```

### Use Cases for Grid Sync

1. **Iterative Algorithms:**
   - Jacobi/Gauss-Seidel solvers
   - Cellular automata
   - Game of Life
   - Heat diffusion

2. **Multi-phase Algorithms:**
   - Radix sort (multiple digit passes)
   - Graph algorithms (level-synchronous BFS)
   - Dynamic programming

3. **Global Coordination:**
   - Convergence checking
   - Global statistics
   - Barrier-based algorithms

### Performance Implications

**Benefits:**
- Eliminates kernel launch overhead between phases
- Reduces CPU-GPU synchronization
- Maintains all data in GPU memory
- Can be 2-10x faster than multiple kernel launches

**Costs:**
- All blocks must fit on GPU simultaneously
- May reduce maximum occupancy
- Grid stays active during sync (resources locked)

**Example - Multiple kernel launches:**
```cuda
for (int i = 0; i < iterations; i++) {
    phase1_kernel<<<grid, block>>>(...);
    cudaDeviceSynchronize();  // CPU-GPU sync (expensive!)

    phase2_kernel<<<grid, block>>>(...);
    cudaDeviceSynchronize();
}
```

**With grid sync (better):**
```cuda
// Single kernel launch for all iterations
multi_phase_kernel<<<grid, block>>>(..., iterations);
```

## Performance Comparison

### Warp Operations

**Traditional:**
```cuda
// Requires explicit mask
int value = __shfl_down_sync(0xffffffff, my_value, offset);

// Easy to forget or misuse mask
int value = __shfl_down_sync(0x0000ffff, my_value, offset);  // Only half warp!
```

**Cooperative Groups:**
```cuda
auto warp = cg::tiled_partition<32>(block);
int value = warp.shfl_down(my_value, offset);  // Cleaner, safer
```

**Performance:** Identical (compiled to same instructions)

### Grid Sync

**Without grid sync (multiple kernels):**
```
Time = N × (kernel_overhead + computation)
     ≈ N × (10-50μs + computation)
```

**With grid sync (single kernel):**
```
Time = kernel_overhead + N × computation
     ≈ 10-50μs + N × computation
```

**Speedup:** Significant for small N or fast kernels

## Code Patterns

### Warp Reduction with CG
```cuda
template<int TILE_SIZE>
__device__ int tile_reduce_sum(cg::thread_block_tile<TILE_SIZE> tile, int value) {
    for (int offset = tile.size() / 2; offset > 0; offset >>= 1) {
        value += tile.shfl_down(value, offset);
    }
    return value;
}
```

### Block Reduction with CG
```cuda
__device__ int block_reduce_sum(cg::thread_block block, int value) {
    extern __shared__ int sdata[];

    int tid = block.thread_rank();
    sdata[tid] = value;
    block.sync();

    for (int s = block.size() / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        block.sync();
    }

    return sdata[0];
}
```

### Hybrid Warp-Block Reduction
```cuda
__global__ void reduce(int *output, const int *input, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int tid = block.thread_rank();
    int i = blockIdx.x * blockDim.x + tid;

    int value = (i < n) ? input[i] : 0;

    // Warp-level reduction
    value = tile_reduce_sum(warp, value);

    // Collect warp results in shared memory
    __shared__ int warp_sums[32];
    if (warp.thread_rank() == 0) {
        warp_sums[tid / 32] = value;
    }
    block.sync();

    // First warp reduces warp results
    if (tid < 32) {
        cg::thread_block_tile<32> first_warp = cg::tiled_partition<32>(block);
        value = (tid < block.size() / 32) ? warp_sums[tid] : 0;
        value = tile_reduce_sum(first_warp, value);

        if (first_warp.thread_rank() == 0) {
            output[blockIdx.x] = value;
        }
    }
}
```

## Building and Running

```bash
mkdir build && cd build
cmake ..
make

# Run basic examples
./cg_basics

# Run grid sync examples (requires CC 6.0+)
./grid_sync
```

## Requirements

- **Cooperative Groups:** CUDA 9.0+
- **Grid Sync:** Compute Capability 6.0+ (Pascal or newer)
- **Include:** `#include <cooperative_groups.h>`
- **Namespace:** `namespace cg = cooperative_groups;`

## References

- [Cooperative Groups Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)
- [Cooperative Groups: Flexible CUDA Thread Programming](https://developer.nvidia.com/blog/cooperative-groups/)
- [Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
