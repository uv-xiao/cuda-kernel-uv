# Example 1: Mini-SGLang Custom Kernels

Understanding the custom CUDA kernels in mini-sglang's lightweight LLM serving framework.

## Overview

Mini-sglang implements four custom kernels that are essential for efficient LLM serving:

1. **Index Kernel** - Embedding table lookup
2. **Store Kernel** - KV cache scatter-write
3. **PyNCCL Kernel** - NCCL communication wrapper
4. **Radix Kernel** - Prefix matching (CPU)

## Index Kernel (Embedding Lookup)

### Functionality

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      INDEX KERNEL OPERATION                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Embedding Table                    Token Indices                       │
│  ┌────────────────────┐            ┌──────────────────┐                │
│  │ V × D (float16)    │            │ B (int32)        │                │
│  │                    │            │                  │                │
│  │ Row 0: ███████████ │            │ [42, 17, 99, ...]│                │
│  │ Row 1: ███████████ │            │                  │                │
│  │ ...                │            └──────────────────┘                │
│  │ Row V-1: █████████ │                    │                           │
│  └────────────────────┘                    │                           │
│                                            ▼                           │
│                                   ┌──────────────────┐                 │
│                                   │ Gather Operation │                 │
│                                   │ (Warp-level)     │                 │
│                                   └────────┬─────────┘                 │
│                                            │                           │
│                                            ▼                           │
│  Output:                          ┌──────────────────┐                 │
│  ┌────────────────────┐          │ B × D (float16)  │                 │
│  │ Row 42: ██████████ │◀─────────│                  │                 │
│  │ Row 17: ██████████ │          │ Embeddings for   │                 │
│  │ Row 99: ██████████ │          │ input tokens     │                 │
│  │ ...                │          └──────────────────┘                 │
│  └────────────────────┘                                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Code Location

**File:** `code-repos/mini-sglang/python/minisgl/kernel/csrc/jit/index.cu`
**Lines:** 31-96

### Key Implementation

```cpp
// Each warp handles one embedding vector
template <std::size_t kNumThreads, std::size_t kMaxOccupancy, bool kUsePDL,
          std::size_t kElementSize, std::size_t kNumSplits, std::integral T>
__global__ void index_kernel(const __grid_constant__ IndexKernelParams params) {
    constexpr auto kWarpPerBlock = kNumThreads / device::kWarpThreads;
    const auto warp_id = (threadIdx.x / device::kWarpThreads) +
                         blockIdx.x * kWarpPerBlock;

    if (warp_id < num_warps) {
        // Get token index for this warp
        const auto pos = indices[warp_id / kNumSplits];

        // Calculate source and destination pointers
        const auto dst = pointer::offset(output, warp_id * kSizePerWarp);
        const auto src = pointer::offset(weight, pos * kSize,
                                         (warp_id % kNumSplits) * kSizePerWarp);

        // Vectorized warp-level copy (16 bytes per thread)
        warp::copy<kSizePerWarp>(dst, src);
    }
}
```

### Memory Access Pattern

- Each **warp (32 threads)** handles one embedding vector
- Vectorized load: `uint4` (16 bytes) per thread per iteration
- For `D=4096, dtype=fp16`: 8KB per vector, 16 iterations per warp
- **Memory bound**: Achieve ~1.8 TB/s on A100

### Tensor Parallelism Support

With `vocab_range` parameter for vocabulary sharding:

```cpp
// Mask out-of-range indices (set to 0)
if (vocab_range) {
    indices_mask = (indices < start) | (indices >= start + length);
    indices[indices_mask] = 0;  // Point to first row
    // Later: zero out masked outputs
}
```

---

## Store Kernel (KV Cache Scatter)

### Functionality

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      STORE KERNEL OPERATION                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input K, V Tensors               Target Indices                        │
│  ┌──────────────────┐            ┌──────────────────┐                  │
│  │ K: [L × H × D]   │            │ [L] int32        │                  │
│  │ V: [L × H × D]   │            │ [42, 17, 99, ...]│                  │
│  └──────────────────┘            └────────┬─────────┘                  │
│                                           │                             │
│                                           ▼                             │
│                                  ┌────────────────────┐                │
│                                  │  Scatter Write     │                │
│                                  │  (Warp-level)      │                │
│                                  └────────┬───────────┘                │
│                                           │                             │
│                                           ▼                             │
│  KV Cache (Paged):                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ K_cache / V_cache: [num_pages × num_heads × head_dim]           │   │
│  │                                                                  │   │
│  │ Page 0:   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░                   │   │
│  │ Page 17:  ████████████████████████████████████  ← Written      │   │
│  │ Page 42:  ████████████████████████████████████  ← Written      │   │
│  │ ...                                                              │   │
│  │ Page 99:  ████████████████████████████████████  ← Written      │   │
│  │ ...                                                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Code Location

**File:** `code-repos/mini-sglang/python/minisgl/kernel/csrc/jit/store.cu`
**Lines:** 25-53

### Key Implementation

```cpp
template <std::size_t kNumThreads, std::size_t kMaxOccupancy, bool kUsePDL,
          std::size_t kElementSize, std::integral T>
__global__ void store_kv_cache(const __grid_constant__ StoreKernelParams params) {
    const auto warp_id = (threadIdx.x / device::kWarpThreads) +
                         blockIdx.x * kWarpPerBlock;

    if (warp_id < length) {
        // Get target page index
        const auto pos = static_cast<const T *>(indices)[warp_id];

        // Store K
        const auto dst_k = pointer::offset(k_cache, pos * kv_cache_stride);
        const auto src_k = pointer::offset(k, warp_id * kv_input_stride);
        warp::copy<kElementSize>(dst_k, src_k);

        // Store V
        const auto dst_v = pointer::offset(v_cache, pos * kv_cache_stride);
        const auto src_v = pointer::offset(v, warp_id * kv_input_stride);
        warp::copy<kElementSize>(dst_v, src_v);
    }
}
```

### Optimization Opportunities

1. **Fuse K+V**: Single kernel call for both
2. **Async copy**: Use `cp.async` for better pipelining
3. **Coalesced writes**: Already achieved with warp-level copy

---

## Warp Copy Utility

Both kernels use a shared warp-level copy primitive:

**File:** `code-repos/mini-sglang/python/minisgl/kernel/csrc/include/minisgl/warp.cuh`
**Lines:** 40-59

```cpp
template <std::size_t kSizePerWarp>
__always_inline __device__ auto copy(void *dst, const void *src) -> void {
    constexpr auto kChunkSize = 16u;  // uint4 = 16 bytes
    constexpr auto kRounds = kSizePerWarp / kWarpThreads / kChunkSize;

    auto src_vec = static_cast<const uint4 *>(src);
    auto dst_vec = static_cast<uint4 *>(dst);
    const auto lane_id = threadIdx.x % kWarpThreads;

    #pragma unroll
    for (auto i = 0u; i < kRounds; ++i) {
        const auto idx = i * kWarpThreads + lane_id;
        dst_vec[idx] = src_vec[idx];  // 16-byte vectorized copy
    }
}
```

---

## Running the Examples

### Test Index Kernel

```bash
cd /home/uvxiao/mlkb/code-repos/mini-sglang
python tests/kernel/test_index.py
```

### Test Store Kernel

```bash
python tests/kernel/test_store.py
```

### Profile with NCU

```bash
# Index kernel
ncu --set full \
    --kernel-name "index_kernel" \
    -o index_profile \
    python tests/kernel/test_index.py

# Store kernel
ncu --set full \
    --kernel-name "store_kv_cache" \
    -o store_profile \
    python tests/kernel/test_store.py

# View results
ncu-ui index_profile.ncu-rep
```

### Key Metrics to Observe

| Metric | Index Kernel Target | Store Kernel Target |
|--------|---------------------|---------------------|
| Memory Throughput | >1.8 TB/s | >1.5 TB/s |
| SM Occupancy | >50% | >50% |
| Achieved BW % | >80% | >70% |

---

## Summary

| Kernel | Pattern | Key Optimization |
|--------|---------|------------------|
| Index | Gather (random read) | Warp-level vectorized load |
| Store | Scatter (random write) | Warp-level vectorized store |

Both kernels are **memory-bound** - optimization focuses on maximizing memory bandwidth utilization through vectorized access patterns.

## Next: [Example 2 - FlashInfer Attention](../02_flashinfer_attention/)
