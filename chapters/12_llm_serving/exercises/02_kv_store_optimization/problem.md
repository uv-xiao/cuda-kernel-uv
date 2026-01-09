# Exercise: KV Store Optimization

## Difficulty: Hard

## Learning Goal

Optimize the KV cache store kernel by fusing K and V writes and using async copy.

## Problem Statement

The current mini-sglang store kernel writes K and V separately. Your task is to implement an optimized version that:

1. Fuses K and V into a single kernel
2. Uses asynchronous memory copy (`cp.async`) for better pipelining
3. Handles non-contiguous K and V inputs efficiently

## Input Specification

- `k_cache`: K cache buffer, `[num_pages, num_heads, head_dim]` float16
- `v_cache`: V cache buffer, `[num_pages, num_heads, head_dim]` float16
- `indices`: Target page indices, `[seq_len]` int32
- `k`: Input K tensor, `[seq_len, num_heads, head_dim]` float16
- `v`: Input V tensor, `[seq_len, num_heads, head_dim]` float16

## Current Implementation (Baseline)

```cpp
// Current: Two separate warp copies per token
const auto dst_k = pointer::offset(k_cache, pos * stride);
const auto src_k = pointer::offset(k, warp_id * stride);
warp::copy<kSize>(dst_k, src_k);

const auto dst_v = pointer::offset(v_cache, pos * stride);
const auto src_v = pointer::offset(v, warp_id * stride);
warp::copy<kSize>(dst_v, src_v);
```

## Requirements

1. **Fused Copy**: Single kernel handles both K and V
2. **Async Operations**: Use `cp.async` for Ampere+ GPUs
3. **Overlapping**: Pipeline loads and stores where possible

## Starter Code

```cpp
template <int HEAD_DIM, int NUM_HEADS>
__global__ void fused_store_kv_async(
    __half* __restrict__ k_cache,      // [num_pages, num_heads, head_dim]
    __half* __restrict__ v_cache,      // [num_pages, num_heads, head_dim]
    const int* __restrict__ indices,   // [seq_len]
    const __half* __restrict__ k,      // [seq_len, num_heads, head_dim]
    const __half* __restrict__ v,      // [seq_len, num_heads, head_dim]
    int seq_len
) {
    // Shared memory for async staging
    extern __shared__ char smem[];
    __half* k_smem = reinterpret_cast<__half*>(smem);
    __half* v_smem = k_smem + HEAD_DIM * NUM_HEADS;

    const int token_idx = blockIdx.x;
    if (token_idx >= seq_len) return;

    const int page_idx = indices[token_idx];

    // TODO: Implement fused async store
    // 1. Issue cp.async for K data to smem
    // 2. Issue cp.async for V data to smem
    // 3. Wait for async copies
    // 4. Store from smem to global memory

    // Hint: Use __pipeline_memcpy_async for cp.async
}
```

## Hints

<details>
<summary>Hint 1: Async Copy API</summary>

```cpp
// For sm_80+
#include <cuda_pipeline.h>

// Async copy from global to shared
__pipeline_memcpy_async(
    dst_smem_ptr,      // Destination in shared memory
    src_global_ptr,    // Source in global memory
    sizeof(uint4)      // Size (must be 4, 8, or 16 bytes)
);

// Wait for all async copies
__pipeline_commit();
__pipeline_wait_prior(0);
```
</details>

<details>
<summary>Hint 2: Pipelining Strategy</summary>

```cpp
// Pipeline loads and stores:
// 1. Load K[i] -> K_smem
// 2. Load V[i] -> V_smem
// 3. Wait for loads
// 4. Store K_smem -> K_cache[page]
// 5. Store V_smem -> V_cache[page]
```
</details>

<details>
<summary>Hint 3: Memory Layout</summary>

```cpp
// Calculate offsets
const int global_offset = token_idx * NUM_HEADS * HEAD_DIM;
const int cache_offset = page_idx * NUM_HEADS * HEAD_DIM;

// Each thread handles multiple elements
const int tid = threadIdx.x;
const int elements_per_thread = (NUM_HEADS * HEAD_DIM) / blockDim.x;
```
</details>

## Testing

```bash
python test.py
```

Expected output:
```
Correctness: PASSED
Baseline performance: XX us
Optimized performance: YY us
Speedup: 1.XX x
```

## Performance Target

Achieve at least 15% improvement over baseline:
- Baseline: ~0.8 TB/s effective bandwidth
- Target: ~0.95 TB/s effective bandwidth

## Analysis Questions

1. Why does fusing K and V improve performance?
2. What is the benefit of async copy vs regular loads?
3. When would this optimization NOT help?

## Bonus Challenges

1. **PTX Assembly**: Use inline PTX for `cp.async` instead of intrinsics
2. **Multi-Head Parallelism**: Parallelize across heads within a block
3. **Stream Pipelining**: Overlap store with next attention computation
