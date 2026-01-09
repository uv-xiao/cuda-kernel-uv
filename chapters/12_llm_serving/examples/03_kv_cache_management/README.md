# Example 3: KV Cache Management

Understanding paged KV cache and memory management in LLM serving.

## Overview

Efficient KV cache management is critical for LLM serving:
- **Paged allocation** avoids fragmentation
- **Prefix sharing** (radix cache) reuses common prefixes
- **Scatter-write** kernel stores new KV pairs efficiently

## Paged KV Cache Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PAGED KV CACHE LAYOUT                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Two Layout Options:                                                    │
│                                                                         │
│  LayerFirst:                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ kv_buffer: [2 × num_layers × num_pages × num_heads × head_dim]  │   │
│  │                                                                  │   │
│  │ Layer 0, K:  [Page 0] [Page 1] [Page 2] ... [Page N]            │   │
│  │ Layer 0, V:  [Page 0] [Page 1] [Page 2] ... [Page N]            │   │
│  │ Layer 1, K:  [Page 0] [Page 1] [Page 2] ... [Page N]            │   │
│  │ ...                                                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  PageFirst:                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ kv_buffer: [2 × num_pages × num_layers × num_heads × head_dim]  │   │
│  │                                                                  │   │
│  │ Page 0:  [L0_K][L0_V][L1_K][L1_V]...[LN_K][LN_V]                 │   │
│  │ Page 1:  [L0_K][L0_V][L1_K][L1_V]...[LN_K][LN_V]                 │   │
│  │ ...                                                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Mini-SGLang uses LayerFirst for cache coherence                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Code Location

**File:** `code-repos/mini-sglang/python/minisgl/kvcache/mha_pool.py`
**Lines:** 10-80

```python
class MHAKVCache(BaseKVCache):
    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        dtype: torch.dtype,
        kv_layout: KVCacheLayout,
        device: torch.device,
    ):
        # LayerFirst layout for better cache locality
        if kv_layout == KVCacheLayout.LayerFirst:
            kv_buffer = torch.empty(
                (2, num_layers, num_pages, local_kv_heads, head_dim),
                device=device,
                dtype=dtype,
            )

        # Split into K and V views
        self._k_buffer = kv_buffer[0]  # [num_layers, num_pages, heads, dim]
        self._v_buffer = kv_buffer[1]  # [num_layers, num_pages, heads, dim]
```

---

## Store Kernel Integration

The store kernel writes new K,V pairs to specific pages:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    STORE KERNEL FLOW                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Attention Layer computes K, V                                       │
│     K: [seq_len × num_kv_heads × head_dim]                             │
│     V: [seq_len × num_kv_heads × head_dim]                             │
│                                                                         │
│  2. Scheduler allocates pages → out_loc tensor                         │
│     out_loc: [seq_len] int32 - page indices                            │
│                                                                         │
│  3. Store kernel scatters K, V to cache                                │
│     ┌─────────────────────────────────────────────────────────────┐    │
│     │ for i in range(seq_len):                                    │    │
│     │     page_idx = out_loc[i]                                   │    │
│     │     k_cache[layer_id, page_idx] = K[i]                      │    │
│     │     v_cache[layer_id, page_idx] = V[i]                      │    │
│     └─────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Store Kernel Call

**File:** `code-repos/mini-sglang/python/minisgl/kvcache/mha_pool.py`
**Lines:** 56-67

```python
def store_kv(
    self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
) -> None:
    from minisgl.kernel import store_cache

    store_cache(
        k_cache=self._k_buffer[layer_id].view(self._storage_shape),
        v_cache=self._v_buffer[layer_id].view(self._storage_shape),
        indices=out_loc,
        k=k,
        v=v,
    )
```

---

## Radix Cache (Prefix Sharing)

Radix cache enables KV reuse for shared prefixes:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    RADIX CACHE OPERATION                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Example: Two requests with shared system prompt                        │
│                                                                         │
│  Request 1: "You are a helpful assistant. What is Python?"              │
│  Request 2: "You are a helpful assistant. Explain ML."                  │
│                                                                         │
│  Radix Tree Structure:                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                         ROOT                                     │   │
│  │                           │                                      │   │
│  │                           ▼                                      │   │
│  │              "You are a helpful assistant."                     │   │
│  │              [Pages 0-15 in KV cache]                           │   │
│  │                    ┌─────┴─────┐                                 │   │
│  │                    │           │                                 │   │
│  │                    ▼           ▼                                 │   │
│  │          "What is Python?"  "Explain ML."                       │   │
│  │          [Pages 16-20]      [Pages 21-24]                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Benefits:                                                              │
│  - Pages 0-15 computed once, reused by both requests                   │
│  - Memory saved: 16 pages × num_layers × kv_size                       │
│  - Compute saved: No re-prefill of shared prefix                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Fast Compare Key (CPU)

**File:** `code-repos/mini-sglang/python/minisgl/kernel/csrc/src/radix.cpp`
**Lines:** 19-40

```cpp
auto fast_compare_key(tvm::ffi::TensorView a, tvm::ffi::TensorView b)
    -> std::int64_t {
    // Find first differing position using std::mismatch (SIMD-optimized)
    const auto begin_a = static_cast<const T *>(a.data());
    const auto begin_b = static_cast<const T *>(b.data());
    const auto end_a = begin_a + std::min(len_a, len_b);

    const auto [it_a, it_b] = std::mismatch(begin_a, end_a, begin_b);

    // Return length of common prefix
    return static_cast<std::int64_t>(std::distance(begin_a, it_a));
}
```

---

## Page Allocation Strategy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PAGE ALLOCATION                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Free Pool                  Active Pages                               │
│  ┌─────────────┐           ┌─────────────────────────────────────┐    │
│  │ [50, 51, 52,│           │ Request A: [0, 1, 2, 3]             │    │
│  │  53, ...]   │           │ Request B: [0, 1, 4, 5, 6]  (shared)│    │
│  └─────────────┘           │ Request C: [10, 11, 12]              │    │
│                            └─────────────────────────────────────┘    │
│                                                                         │
│  Operations:                                                            │
│  1. Allocate: Pop from free pool → assign to request                   │
│  2. Free: Request done → return pages to free pool                     │
│  3. Evict: Memory pressure → evict LRU, return to free pool            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Memory Sizing

Calculate KV cache memory requirement:

```python
def calc_kv_cache_memory(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    num_pages: int,
    dtype: torch.dtype = torch.float16,
) -> int:
    """Calculate KV cache memory in bytes."""
    bytes_per_element = torch.finfo(dtype).bits // 8
    # 2 for K and V
    return 2 * num_layers * num_pages * num_kv_heads * head_dim * bytes_per_element

# Example: Llama-3.1-8B with 8192 pages
# 32 layers, 8 KV heads (GQA), 128 head_dim
memory_gb = calc_kv_cache_memory(32, 8, 128, 8192) / (1024**3)
print(f"KV Cache: {memory_gb:.2f} GB")  # ~1 GB
```

---

## Running the Example

### Test Store Kernel

```bash
cd /home/uvxiao/mlkb/code-repos/mini-sglang
python tests/kernel/test_store.py
```

### Profile Memory Access

```bash
ncu --set full \
    --section MemoryWorkloadAnalysis \
    --kernel-name "store_kv_cache" \
    -o store_memory \
    python tests/kernel/test_store.py
```

---

## Summary

| Component | Purpose | Key Optimization |
|-----------|---------|------------------|
| Paged KV Cache | Avoid fragmentation | Fixed-size page allocation |
| Store Kernel | Scatter-write K,V | Warp-level vectorized copy |
| Radix Cache | Prefix sharing | Fast CPU comparison (std::mismatch) |

## Next: [Example 4 - Distributed Inference](../04_distributed_inference/)
