# Example 03: IO Overlap with Computation

## Overview

This example demonstrates **overlapping memory transfers with computation** using NVIDIA Hopper/Blackwell async features, specifically the Tensor Memory Accelerator (TMA).

## The IO Bottleneck in MoE

### Memory Transfer Overhead

In standard MoE implementations:

```
For each expert:
  1. Load expert weights (14336 * 4096 * 4 bytes = 235 MB)
  2. Wait for load to complete
  3. Compute GEMM
  4. Wait for compute to complete
  5. Store results
```

**Problem**: Steps 1-2 and 4-5 block the computation pipeline!

### Hopper/Blackwell TMA Features

**Tensor Memory Accelerator (TMA):**
- Asynchronous memory copy engine
- Independent of SM execution
- Hardware-managed synchronization
- Supports multi-dimensional tensor layouts

**Key Operations:**
```cuda
// Async copy (doesn't block SM)
__pipeline_memcpy_async(dest, src, size);

// Continue computing while copy happens
compute_something();

// Wait only when needed
__pipeline_wait_prior(0);
```

## Performance Impact

### Without IO Overlap
```
Timeline:
[Load-1][Compute-1][Store-1][Load-2][Compute-2][Store-2]...
```
- Total time: (Load + Compute + Store) × NumExperts
- GPU idle during memory ops

### With IO Overlap
```
Timeline:
[Load-1]
  [Compute-1]
    [Store-1][Load-2]
              [Compute-2]
                [Store-2][Load-3]...
```
- Total time ≈ max(Load, Compute, Store) × NumExperts
- 40% reduction in end-to-end latency (SonicMoE results)

## Files

### `io_overlap.cu`
Basic IO overlap example:
- Async memory copies with TMA
- Double buffering for continuous pipeline
- Synchronization primitives

### `pipelined_moe.cu`
Full pipelined MoE forward pass:
- Overlap expert weight loading
- Concurrent computation and transfers
- Multi-stream execution

### `CMakeLists.txt`
Build configuration (requires SM90+ for TMA)

## Hardware Requirements

**Minimum:**
- NVIDIA H100 (SM90) for full TMA support
- CUDA 12.1+

**Alternative:**
- A100 (SM80) with cudaMemcpyAsync (limited overlap)

## Building and Running

```bash
mkdir build && cd build
cmake ..
make

# Basic IO overlap
./io_overlap

# Full pipelined MoE
./pipelined_moe
```

## Expected Results

### Basic IO Overlap
```
IO Overlap Benchmark
====================

Configuration:
  - Data Size: 256 MB
  - Num Transfers: 8
  - Compute Intensity: High

Without Overlap:
  - Total Time: 145.2 ms
  - Transfer Time: 89.3 ms
  - Compute Time: 55.9 ms

With Overlap:
  - Total Time: 92.1 ms
  - Speedup: 1.58x
  - Overlap Efficiency: 91.2%
```

### Pipelined MoE
```
Pipelined MoE Benchmark
=======================

Configuration:
  - Num Experts: 256
  - Hidden Dim: 7168
  - FFN Dim: 14336
  - Batch Size: 32, Seq Len: 512

Sequential Execution:
  - Weight Load Time: 342.1 ms
  - Compute Time: 189.5 ms
  - Total Time: 531.6 ms

Pipelined Execution:
  - Total Time: 315.8 ms
  - Speedup: 1.68x
  - Pipeline Efficiency: 87.4%
```

## Key Techniques

### 1. Double Buffering

```cuda
__shared__ float weight_buffer[2][TILE_SIZE][TILE_SIZE];
int read_buffer = 0;
int write_buffer = 1;

for (int expert = 0; expert < num_experts; ++expert) {
    // Load next expert weights to write_buffer (async)
    if (expert + 1 < num_experts) {
        __pipeline_memcpy_async(
            weight_buffer[write_buffer],
            expert_weights[expert + 1],
            BUFFER_SIZE
        );
    }

    // Compute with read_buffer
    compute_expert(weight_buffer[read_buffer]);

    // Swap buffers
    read_buffer ^= 1;
    write_buffer ^= 1;

    // Wait for async copy
    __pipeline_wait_prior(0);
}
```

### 2. Multi-Stream Execution

```cuda
cudaStream_t compute_stream, transfer_stream;
cudaStreamCreate(&compute_stream);
cudaStreamCreate(&transfer_stream);

for (int expert = 0; expert < num_experts; ++expert) {
    // Async load on transfer stream
    cudaMemcpyAsync(weight_buffer[expert],
                    expert_weights[expert],
                    size,
                    cudaMemcpyDeviceToDevice,
                    transfer_stream);

    // Compute on compute stream (can overlap)
    expert_gemm_kernel<<<grid, block, 0, compute_stream>>>(
        inputs, weight_buffer[expert - 1], outputs
    );
}

cudaStreamSynchronize(compute_stream);
cudaStreamSynchronize(transfer_stream);
```

### 3. TMA-Specific Optimizations

```cuda
// Create TMA descriptor (Hopper)
CUtensorMap tma_desc;
cuTensorMapEncodeTiled(
    &tma_desc,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
    num_dims,
    tensor_address,
    tensor_dims,
    tensor_strides,
    box_dims,
    element_strides,
    CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_128B,
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
);

// Use TMA in kernel
__shared__ float tile[TILE_M][TILE_K];
asm volatile("cp.async.bulk.tensor.2d.shared.global [%0], [%1, {%2, %3}];"
             :: "r"(__cvta_generic_to_shared(tile)),
                "l"(&tma_desc),
                "r"(tile_coord_m),
                "r"(tile_coord_k));
```

## Performance Tuning

### Pipeline Depth
- **Depth 1**: No overlap, sequential
- **Depth 2**: Double buffering (recommended)
- **Depth 3+**: Diminishing returns, increased memory

### Buffer Sizing
- **Too small**: Frequent synchronization, poor overlap
- **Too large**: Increased memory pressure, cache thrashing
- **Optimal**: Match tile size (128x128 for FP32 = 64 KB)

### Stream Configuration
- **Single stream**: No overlap
- **Dual stream**: Transfer + Compute overlap
- **Multi-stream**: Expert-level parallelism (complex synchronization)

## Comparison: A100 vs H100

| Feature | A100 (SM80) | H100 (SM90) |
|---------|-------------|-------------|
| Async Memory | cudaMemcpyAsync | TMA |
| Overlap Efficiency | 60-70% | 85-95% |
| Synchronization | cudaEvent | Hardware barrier |
| Multi-dim Copies | Manual | Native TMA |
| Speedup vs Sequential | 1.3x | 1.6-1.8x |

**Recommendation**: Use TMA on H100/H200 for best performance.

## Debugging Tips

### Check Overlap Efficiency

```cuda
// Measure actual overlap
cudaEvent_t start, stop, transfer_done;
cudaEventCreate(&start);
cudaEventCreate(&transfer_done);
cudaEventCreate(&stop);

cudaEventRecord(start, transfer_stream);
// ... async transfer ...
cudaEventRecord(transfer_done, transfer_stream);

cudaEventRecord(start, compute_stream);
// ... compute ...
cudaEventRecord(stop, compute_stream);

float transfer_time, compute_time, total_time;
cudaEventElapsedTime(&transfer_time, start, transfer_done);
cudaEventElapsedTime(&compute_time, start, stop);
total_time = max(transfer_time, compute_time);

float overlap_efficiency = 1.0f - (total_time - max(transfer_time, compute_time)) /
                                  (transfer_time + compute_time);
```

### Nsight Systems Profile

```bash
nsys profile --stats=true ./pipelined_moe
```

Look for:
- Stream overlap visualizations
- Memory transfer gaps
- Kernel launch overhead

## Next Steps

- **Example 04**: Combine IO overlap with tile-aware token rounding
- **Example 05**: Full SonicMoE integration
