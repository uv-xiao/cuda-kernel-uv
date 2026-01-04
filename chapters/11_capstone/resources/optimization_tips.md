# Optimization Tips for CUDA Kernels

A comprehensive guide to common optimization patterns and techniques for your capstone projects.

## Table of Contents

1. [Memory Optimization](#memory-optimization)
2. [Compute Optimization](#compute-optimization)
3. [Kernel Launch Configuration](#kernel-launch-configuration)
4. [GEMM-Specific Optimizations](#gemm-specific-optimizations)
5. [Attention-Specific Optimizations](#attention-specific-optimizations)
6. [Debugging Performance Issues](#debugging-performance-issues)

---

## Memory Optimization

### 1. Coalesced Global Memory Access

**Problem**: Uncoalesced access causes multiple memory transactions.

**Solution**: Ensure adjacent threads access adjacent memory locations.

```cuda
// Bad: Strided access
__global__ void bad_kernel(float* data, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = data[idx * stride];  // Strided access
}

// Good: Coalesced access
__global__ void good_kernel(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = data[idx];  // Adjacent threads access adjacent elements
}
```

**Verification**: Check `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` in ncu (should be >80%).

### 2. Shared Memory Optimization

**Use Shared Memory For**:
- Data reused across threads in a block
- Staging data before writing to global memory
- Implementing fast communication within block

```cuda
__global__ void tiled_kernel(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile into shared memory
        As[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        __syncthreads();

        // Compute using shared memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}
```

### 3. Avoid Bank Conflicts

**Problem**: Multiple threads in a warp access the same shared memory bank.

**32-bit shared memory banks**: 32 banks, 4-byte wide.

```cuda
// Bad: Bank conflicts
__shared__ float s_data[32][32];
float val = s_data[threadIdx.x][0];  // All threads access bank 0

// Good: No conflicts (broadcast)
float val = s_data[0][threadIdx.x];  // Each thread different bank

// Good: Padding to avoid conflicts
__shared__ float s_data[32][33];  // Extra column shifts banks
float val = s_data[threadIdx.x][threadIdx.y];
```

**Verification**: Check `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` in ncu (should be 0).

### 4. Use of Registers

**Maximize register usage for**:
- Frequently accessed variables
- Loop accumulators
- Intermediate results

```cuda
__global__ void register_blocking_kernel(...) {
    // Use registers for accumulation
    float acc[8][8];  // 64 registers

    // Initialize
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            acc[i][j] = 0.0f;
        }
    }

    // Compute and accumulate in registers
    // ...

    // Write back to global memory
}
```

**Watch out for**: Register spilling (check with `--resource-usage` flag).

### 5. Constant Memory

**Use for**: Small read-only data accessed by all threads.

```cuda
__constant__ float c_weights[256];

// Copy to constant memory
cudaMemcpyToSymbol(c_weights, h_weights, 256 * sizeof(float));

// Access in kernel (cached)
__global__ void kernel() {
    float w = c_weights[index];  // Fast cached access
}
```

### 6. Texture Memory

**Use for**: Read-only data with 2D spatial locality.

```cuda
// Bind texture
cudaTextureObject_t tex;
cudaTextureDesc texDesc;
memset(&texDesc, 0, sizeof(texDesc));
texDesc.readMode = cudaReadModeElementType;

cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

// Use in kernel
__global__ void kernel(cudaTextureObject_t tex) {
    float val = tex2D<float>(tex, x, y);  // Cached 2D access
}
```

---

## Compute Optimization

### 1. Minimize Warp Divergence

**Problem**: Threads in a warp take different execution paths.

```cuda
// Bad: Divergent branches
__global__ void divergent_kernel(float* data, int* flags) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (flags[idx] == 0) {
        // Path A
        data[idx] = expensive_computation_A();
    } else {
        // Path B
        data[idx] = expensive_computation_B();
    }
}

// Better: Minimize divergence
__global__ void less_divergent_kernel(float* data, int* flags) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result;

    // Both paths execute, but predicated
    float resultA = expensive_computation_A();
    float resultB = expensive_computation_B();

    // Single assignment (predicated)
    result = (flags[idx] == 0) ? resultA : resultB;
    data[idx] = result;
}

// Best: Partition data to reduce divergence
// Launch separate kernels for different paths
```

### 2. Use Fast Math

**Enable fast math for**: Non-critical floating-point operations.

```cuda
// Compile with: -use_fast_math

__global__ void kernel() {
    float x = 1.0f;

    // Fast approximations used automatically
    float y = __sinf(x);      // Fast sine
    float z = __expf(x);      // Fast exp
    float w = __fdividef(1.0f, x);  // Fast divide
}
```

**Trade-off**: Slightly less accuracy for speed.

### 3. Loop Unrolling

**Manually unroll small loops**:

```cuda
// Before
for (int i = 0; i < 4; i++) {
    sum += data[i];
}

// After (#pragma unroll)
#pragma unroll
for (int i = 0; i < 4; i++) {
    sum += data[i];
}

// Or manually:
sum += data[0];
sum += data[1];
sum += data[2];
sum += data[3];
```

### 4. Warp-Level Primitives

**Use warp shuffle for fast intra-warp communication**:

```cuda
// Warp-level reduction
__device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduction_kernel(float* input, float* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = input[idx];

    // Warp-level reduction (no shared memory needed)
    val = warp_reduce_sum(val);

    // First thread in warp writes result
    if (threadIdx.x % 32 == 0) {
        output[blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32] = val;
    }
}
```

### 5. Instruction-Level Parallelism (ILP)

**Increase ILP by processing multiple elements per thread**:

```cuda
__global__ void ilp_kernel(float* input, float* output) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;  // Process 4 elements

    // Load 4 elements (memory latency hidden)
    float v0 = input[idx + 0];
    float v1 = input[idx + 1];
    float v2 = input[idx + 2];
    float v3 = input[idx + 3];

    // Compute (operations can overlap)
    v0 = expensive_function(v0);
    v1 = expensive_function(v1);
    v2 = expensive_function(v2);
    v3 = expensive_function(v3);

    // Store results
    output[idx + 0] = v0;
    output[idx + 1] = v1;
    output[idx + 2] = v2;
    output[idx + 3] = v3;
}
```

---

## Kernel Launch Configuration

### 1. Choosing Block Size

**Rule of thumb**:
- Multiples of warp size (32)
- Common choices: 128, 256, 512
- Experiment to find optimal

```python
# Auto-tune block size
block_sizes = [64, 128, 256, 512]
best_time = float('inf')
best_block_size = 128

for block_size in block_sizes:
    grid_size = (N + block_size - 1) // block_size
    time = benchmark_kernel(grid_size, block_size)
    if time < best_time:
        best_time = time
        best_block_size = block_size
```

### 2. Occupancy Optimization

**Check occupancy**:
```bash
nvcc --ptxas-options=-v kernel.cu
# Look for: registers per thread, shared memory per block

# Or use CUDA Occupancy Calculator
```

**Improve occupancy**:
- Reduce register usage (`maxrregcount=N`)
- Reduce shared memory usage
- Increase block size (if possible)

**Note**: High occupancy doesn't always mean better performance!

### 3. Grid-Stride Loops

**For variable-size problems**:

```cuda
__global__ void grid_stride_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Grid-stride loop
    for (int i = idx; i < N; i += stride) {
        data[i] = process(data[i]);
    }
}

// Launch with fixed grid size
int numSMs;
cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
int blockSize = 256;
int gridSize = 32 * numSMs;  // 32 blocks per SM
grid_stride_kernel<<<gridSize, blockSize>>>(data, N);
```

---

## GEMM-Specific Optimizations

### 1. Tiling Strategy

**2D Tiling for GEMM**:

```cuda
// Optimal tile sizes (tune for your GPU)
#define TILE_M 128
#define TILE_N 128
#define TILE_K 16

__global__ void gemm_tiled(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Each thread computes multiple output elements
    float acc[8][8] = {0};

    for (int t = 0; t < K; t += TILE_K) {
        // Load tiles
        // Compute
        // Accumulate
    }

    // Write results
}
```

### 2. Register Blocking

**Compute multiple outputs per thread**:

```cuda
// Each thread computes 8x8 output tile
#define REG_M 8
#define REG_N 8

__global__ void gemm_register_blocking(...) {
    float acc[REG_M][REG_N];
    float a_frag[REG_M];
    float b_frag[REG_N];

    // Load, compute, accumulate in registers
}
```

### 3. Vectorized Memory Access

**Use vector loads when possible**:

```cuda
// Load 4 floats at once
float4 data = reinterpret_cast<float4*>(ptr)[idx];
float v0 = data.x;
float v1 = data.y;
float v2 = data.z;
float v3 = data.w;
```

### 4. Double Buffering

**Hide memory latency with double buffering**:

```cuda
__shared__ float As[2][TILE_M][TILE_K];
__shared__ float Bs[2][TILE_K][TILE_N];

int write_idx = 0;
int read_idx = 1;

// Load first tile
load_tile(As[write_idx], Bs[write_idx], ...);
__syncthreads();

for (int t = TILE_K; t < K; t += TILE_K) {
    // Swap buffers
    write_idx = 1 - write_idx;
    read_idx = 1 - read_idx;

    // Load next tile while computing current tile
    load_tile(As[write_idx], Bs[write_idx], ...);
    compute_tile(As[read_idx], Bs[read_idx], ...);
    __syncthreads();
}
```

---

## Attention-Specific Optimizations

### 1. Online Softmax

**Avoid materializing full attention matrix**:

```cuda
// Traditional softmax (requires full matrix)
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        S[i][j] = exp(Q[i] * K[j]);
    }
    float sum = reduce_sum(S[i]);
    for (int j = 0; j < N; j++) {
        S[i][j] /= sum;
    }
}

// Online softmax (Flash Attention style)
float max_val = -INFINITY;
float sum = 0.0f;
float acc[D] = {0};

for (int j = 0; j < N; j++) {
    float score = Q[i] * K[j];
    float new_max = fmaxf(max_val, score);

    // Rescale previous sum and accumulator
    float rescale = expf(max_val - new_max);
    sum = sum * rescale;
    for (int d = 0; d < D; d++) {
        acc[d] *= rescale;
    }

    // Update with new element
    float exp_score = expf(score - new_max);
    sum += exp_score;
    for (int d = 0; d < D; d++) {
        acc[d] += exp_score * V[j][d];
    }

    max_val = new_max;
}

// Final normalization
for (int d = 0; d < D; d++) {
    output[d] = acc[d] / sum;
}
```

### 2. Tiling for Attention

**Tile Q, K, V to fit in shared memory**:

```cuda
#define BLOCK_Q 64
#define BLOCK_K 64

__global__ void flash_attention(...) {
    __shared__ float Q_tile[BLOCK_Q][HEAD_DIM];
    __shared__ float K_tile[BLOCK_K][HEAD_DIM];
    __shared__ float V_tile[BLOCK_K][HEAD_DIM];

    // Outer loop over K tiles
    for (int k_start = 0; k_start < seq_len; k_start += BLOCK_K) {
        // Load K, V tiles
        // Inner loop over Q tiles
        for (int q_start = 0; q_start < seq_len; q_start += BLOCK_Q) {
            // Load Q tile
            // Compute attention for this tile
            // Update output with online softmax
        }
    }
}
```

### 3. Sparse Attention Optimizations

**For sparse patterns**:

```cuda
// Store only non-zero indices
__global__ void sparse_attention(
    float* Q, float* K, float* V,
    int* q_indices,  // Query indices
    int* k_indices,  // Key indices (for each query)
    int* k_counts,   // Number of keys per query
    float* output
) {
    int q_idx = blockIdx.x;
    int q = q_indices[q_idx];

    int k_start = k_counts[q_idx];
    int k_end = k_counts[q_idx + 1];

    float max_score = -INFINITY;
    float sum = 0.0f;

    // Only process non-zero entries
    for (int i = k_start; i < k_end; i++) {
        int k = k_indices[i];
        float score = dot(Q[q], K[k]);
        // online softmax...
    }
}
```

---

## Debugging Performance Issues

### Issue: Low Performance Despite Optimizations

**Checklist**:
1. Profile first - verify assumptions
2. Check memory bandwidth utilization
3. Check SM utilization
4. Look for unexpected bottlenecks

### Issue: Performance Varies with Input Size

**Check**:
- Small inputs: kernel launch overhead dominates
- Large inputs: may exceed cache capacity
- Non-power-of-2: may have suboptimal block division

### Issue: Optimization Made Things Slower

**Possible reasons**:
- Increased register pressure (lower occupancy)
- Increased shared memory usage (lower occupancy)
- Made memory access pattern worse
- Compiler couldn't optimize complex code

**Solution**: Revert and try a different approach, or simplify the optimization.

---

## Quick Optimization Workflow

1. **Baseline**: Implement naive version, verify correctness
2. **Profile**: Identify actual bottleneck
3. **Hypothesize**: What optimization should help?
4. **Implement**: Apply one optimization at a time
5. **Measure**: Did it help? By how much?
6. **Iterate**: Repeat from step 2

**Always**:
- Profile before optimizing
- Apply one optimization at a time
- Verify correctness after each change
- Document what you tried (even if it didn't work)

---

## Additional Resources

- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUDA Optimization Tips](https://developer.nvidia.com/blog/tag/optimization/)
- [Nsight Compute Metrics Guide](https://docs.nvidia.com/nsight-compute/)
- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)

Remember: **Measure, don't guess!**
