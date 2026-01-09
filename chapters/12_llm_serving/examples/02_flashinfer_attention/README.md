# Example 2: FlashInfer Attention Kernels

Deep dive into FlashInfer's optimized attention kernels for LLM serving.

## Overview

FlashInfer provides two primary attention wrappers used in mini-sglang:

1. **BatchPrefillWithPagedKVCacheWrapper** - For prefill phase
2. **BatchDecodeWithPagedKVCacheWrapper** - For decode phase

Both support **paged KV cache** for efficient memory management.

## Prefill Attention

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PREFILL ATTENTION (FlashAttention-2)                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input:                                                                 │
│    Q: [total_tokens × num_heads × head_dim]                            │
│    K_cache, V_cache: [num_pages × num_heads × head_dim]                │
│    qo_indptr: [batch+1] - cumulative Q lengths                         │
│    paged_kv_indptr: [batch+1] - cumulative KV page counts              │
│                                                                         │
│  FlashAttention-2 Tiling:                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  For each Q block (Br rows):                                    │   │
│  │    Initialize: O = 0, m = -inf, l = 0                           │   │
│  │                                                                  │   │
│  │    For each K block (Bc columns):                               │   │
│  │      ┌─────────────────────────────────────────────────────┐    │   │
│  │      │ 1. Load Q_block, K_block, V_block to SMEM           │    │   │
│  │      │ 2. S = Q_block @ K_block^T / sqrt(d)                │    │   │
│  │      │ 3. Apply causal mask (if causal)                    │    │   │
│  │      │ 4. m_new = max(m, rowmax(S))                        │    │   │
│  │      │ 5. P = exp(S - m_new)                               │    │   │
│  │      │ 6. l_new = exp(m - m_new) * l + rowsum(P)           │    │   │
│  │      │ 7. O = exp(m - m_new) * O + P @ V_block             │    │   │
│  │      │ 8. Update m = m_new, l = l_new                      │    │   │
│  │      └─────────────────────────────────────────────────────┘    │   │
│  │                                                                  │   │
│  │    Final: O = O / l                                             │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Output: [total_tokens × num_heads × head_dim]                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Usage in Mini-SGLang

**File:** `code-repos/mini-sglang/python/minisgl/attention/fi.py`
**Lines:** 86-130

```python
from flashinfer import BatchPrefillWithPagedKVCacheWrapper

# Initialize with 128MB workspace
workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device='cuda')
prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
    workspace,
    kv_layout="NHD",     # num_pages × num_heads × head_dim
    backend="fa2",       # FlashAttention-2 backend
)

# Plan phase (CPU, sets up indices)
prefill_wrapper.plan(
    qo_indptr=qo_indptr.cpu(),
    paged_kv_indptr=kv_indptr.cpu(),
    paged_kv_indices=kv_indices.cuda(),
    paged_kv_last_page_len=last_page_len.cpu(),
    num_qo_heads=32,
    num_kv_heads=8,      # GQA: 4 Q heads per KV head
    head_dim_qk=128,
    page_size=1,
    causal=True,
)

# Run phase (GPU)
output = prefill_wrapper.run(
    q=q,                  # [total_tokens, num_heads, head_dim]
    paged_kv_cache=(k_cache, v_cache)
)
```

---

## Decode Attention

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DECODE ATTENTION (Split-K)                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input:                                                                 │
│    Q: [batch_size × num_heads × head_dim]  (1 token per sequence)      │
│    K_cache, V_cache: [num_pages × num_heads × head_dim]                │
│                                                                         │
│  Split-K Parallelism (for long sequences):                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Sequence: [KV tokens 0...4095]                                 │   │
│  │            Split into 4 chunks of 1024                          │   │
│  │                                                                  │   │
│  │  Thread Block 0: KV[0:1024]                                     │   │
│  │  ┌─────────────────────┐                                        │   │
│  │  │ partial_O[0]       │                                        │   │
│  │  │ partial_lse[0]     │                                        │   │
│  │  └─────────────────────┘                                        │   │
│  │                                                                  │   │
│  │  Thread Block 1: KV[1024:2048]                                  │   │
│  │  ┌─────────────────────┐                                        │   │
│  │  │ partial_O[1]       │                                        │   │
│  │  │ partial_lse[1]     │                                        │   │
│  │  └─────────────────────┘                                        │   │
│  │                                                                  │   │
│  │  ...                                                             │   │
│  │                                                                  │   │
│  │  Reduction:                                                      │   │
│  │  ┌─────────────────────────────────────────────────────────┐    │   │
│  │  │ O_final = merge(partial_O[0:4], partial_lse[0:4])       │    │   │
│  │  │                                                          │    │   │
│  │  │ For each partition i:                                    │    │   │
│  │  │   scale_i = exp(lse_i - lse_max)                        │    │   │
│  │  │   O_final += scale_i * O_i                              │    │   │
│  │  │ O_final /= sum(scale_i)                                  │    │   │
│  │  └─────────────────────────────────────────────────────────┘    │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Tensor Core Optimization:                                              │
│  - Enabled when GQA ratio >= 4 (4+ Q heads per KV head)               │
│  - Uses WMMA/MMA instructions for FP16/BF16                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Usage in Mini-SGLang

```python
from flashinfer import BatchDecodeWithPagedKVCacheWrapper

decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
    workspace,
    kv_layout="NHD",
)

# Plan phase
decode_wrapper.plan(
    indptr=indptr.cpu(),
    indices=indices,
    last_page_len=last_page_len.cpu(),
    num_qo_heads=32,
    num_kv_heads=8,
    head_dim_qk=128,
    page_size=1,
    pos_encoding_mode="NONE",      # RoPE applied separately
    q_data_type=torch.float16,
    kv_data_type=torch.float16,
)

# Run phase
output = decode_wrapper.run(
    q=q,                            # [batch_size, num_heads, head_dim]
    paged_kv_cache=(k_cache, v_cache)
)
```

---

## CUDA Graph Support

For latency-critical decode, use CUDA graphs:

```python
from flashinfer import CUDAGraphBatchDecodeWithPagedKVCacheWrapper

cuda_graph_wrapper = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
    workspace,
    kv_layout="NHD",
    max_batch_size=128,             # Pre-allocate for max batch
    max_num_pages=8192,
)

# First iteration captures the graph
cuda_graph_wrapper.plan(...)
output = cuda_graph_wrapper.run(q, kv_cache)

# Subsequent iterations replay the graph (low latency)
output = cuda_graph_wrapper.run(q, kv_cache)
```

---

## RMSNorm and RoPE Kernels

### RMSNorm

```python
from flashinfer import rmsnorm, fused_add_rmsnorm

# Standard RMSNorm
output = rmsnorm(input, weight, eps=1e-6)

# Fused Add + RMSNorm (saves memory bandwidth)
# residual += input; input = rmsnorm(residual)
fused_add_rmsnorm(input, residual, weight, eps=1e-6)
```

### RoPE (Rotary Position Embedding)

```python
from flashinfer import apply_rope_with_cos_sin_cache_inplace

# Precompute cos/sin cache
cos_sin_cache = precompute_freqs_cis(head_dim, max_seq_len)

# Apply RoPE in-place
apply_rope_with_cos_sin_cache_inplace(
    positions=positions,    # [seq_len]
    query=query,            # [seq_len, num_heads, head_dim]
    key=key,                # [seq_len, num_kv_heads, head_dim]
    head_size=head_dim,
    cos_sin_cache=cos_sin_cache,
)
```

---

## Profiling FlashInfer Kernels

### Full Attention Profile

```bash
ncu --set full \
    --kernel-regex ".*prefill.*|.*decode.*|.*attention.*" \
    -o flashinfer_attn \
    python -c "
import torch
from flashinfer import BatchDecodeWithPagedKVCacheWrapper

batch_size, num_pages = 32, 8192
num_heads, head_dim = 32, 128

workspace = torch.empty(128*1024*1024, dtype=torch.uint8, device='cuda')
kv_cache = torch.randn(2, num_pages, num_heads, head_dim,
                       device='cuda', dtype=torch.float16)

wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout='NHD')

# Setup (simplified)
indptr = torch.arange(0, batch_size + 1, device='cuda', dtype=torch.int32) * 64
indices = torch.arange(batch_size * 64, device='cuda', dtype=torch.int32)
last_page_len = torch.ones(batch_size, device='cuda', dtype=torch.int32)

wrapper.plan(indptr.cpu(), indices, last_page_len.cpu(),
             num_heads, num_heads, head_dim, 1, 'NONE',
             torch.float16, torch.float16, torch.float16)

q = torch.randn(batch_size, num_heads, head_dim, device='cuda', dtype=torch.float16)
for _ in range(10):
    out = wrapper.run(q, paged_kv_cache=(kv_cache[0], kv_cache[1]))
torch.cuda.synchronize()
"
```

### RMSNorm Profile

```bash
ncu --set full \
    --kernel-regex ".*rmsnorm.*" \
    -o rmsnorm_profile \
    python -c "
import torch
from flashinfer import rmsnorm, fused_add_rmsnorm

bs, hidden = 4096, 4096
x = torch.randn(bs, hidden, device='cuda', dtype=torch.float16)
w = torch.randn(hidden, device='cuda', dtype=torch.float16)
r = torch.randn(bs, hidden, device='cuda', dtype=torch.float16)

for _ in range(100):
    rmsnorm(x, w)
    fused_add_rmsnorm(x.clone(), r.clone(), w)
torch.cuda.synchronize()
"
```

---

## Performance Targets

| Kernel | Metric | A100 Target |
|--------|--------|-------------|
| Prefill Attention | Memory BW | >1.5 TB/s |
| Decode Attention | Memory BW | >1.6 TB/s |
| RMSNorm | Memory BW | >1.8 TB/s |
| RoPE | Memory BW | >1.5 TB/s |

All attention/norm kernels are **memory-bound** for typical LLM configurations.

---

## Summary

| Component | FlashInfer API | Key Feature |
|-----------|----------------|-------------|
| Prefill | `BatchPrefillWithPagedKVCacheWrapper` | FA2 tiling, online softmax |
| Decode | `BatchDecodeWithPagedKVCacheWrapper` | Split-K, tensor cores |
| CUDA Graph | `CUDAGraphBatchDecodeWithPagedKVCacheWrapper` | Low-latency decode |
| RMSNorm | `rmsnorm`, `fused_add_rmsnorm` | Memory-efficient normalization |
| RoPE | `apply_rope_with_cos_sin_cache_inplace` | In-place position embedding |

## Next: [Example 3 - KV Cache Management](../03_kv_cache_management/)
