# Example 01: FlashInfer Basics

## Overview

This example demonstrates the core FlashInfer API for attention computation in LLM inference.

## Files

- `prefill_attention.py` - Prefill phase attention with paged KV-cache
- `decode_attention.py` - Decode phase attention with Split-K
- `backend_selection.py` - Selecting and comparing attention backends
- `plan_run_pattern.py` - Demonstrating Plan-Run for CUDAGraph compatibility

## Key Concepts

### 1. Paged KV-Cache

FlashInfer uses a unified block-sparse format for KV-cache:

```python
import flashinfer

# Create wrapper for paged attention
workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")

# Plan phase: compute work distribution
wrapper.plan(
    qo_indptr=qo_indptr,     # [batch+1], cumsum of query lengths
    kv_indptr=kv_indptr,     # [batch+1], cumsum of kv lengths
    kv_indices=kv_indices,   # [total_kv_pages], page indices
    kv_last_page_len=kv_last_page_len,  # [batch], tokens in last page
    num_qo_heads=32,
    num_kv_heads=8,          # GQA: 32 / 8 = 4 queries per KV head
    head_dim=128,
)

# Run phase: execute attention
output = wrapper.run(q, kv_cache)
```

### 2. Plan-Run Pattern

```
PLAN-RUN SEPARATION
===================

CPU (Dynamic)                    GPU (Static)
    |                                |
    v                                |
plan(indptr, indices, ...)          |
    |                                |
    v                                |
work_indptr = balance_work(...)     |
    |                                |
    v                                v
pack_plan_info(...)  ----------> run(q, kv_cache)
                                    |
                                    v
                              kernel<<<grid>>>()
```

**Benefits**:
- Plan can vary per batch (dynamic shapes)
- Run has fixed launch config (CUDAGraph compatible)
- Work is load-balanced across SMs

### 3. Backend Selection

```python
# Automatic backend selection
wrapper = BatchPrefillWithPagedKVCacheWrapper(
    workspace, "NHD", backend="auto"
)

# Or explicit selection
wrapper_fa2 = BatchPrefillWithPagedKVCacheWrapper(
    workspace, "NHD", backend="fa2"
)
wrapper_fa3 = BatchPrefillWithPagedKVCacheWrapper(
    workspace, "NHD", backend="fa3"  # Hopper only
)
```

## Running the Examples

```bash
# Basic prefill attention
python prefill_attention.py

# Decode attention with Split-K
python decode_attention.py

# Compare backends
python backend_selection.py --backends fa2,fa3,auto

# Plan-Run with CUDAGraph
python plan_run_pattern.py --use-cudagraph
```

## Expected Output

```
Prefill Attention (seq_len=2048, batch=4):
  FA2 backend: 0.45 ms
  FA3 backend: 0.38 ms (15.6% faster)

Decode Attention (kv_len=2048, batch=32):
  Latency: 0.12 ms
  HBM bandwidth: 1.8 TB/s (87% of peak)

CUDAGraph Capture:
  First run: 2.3 ms (includes graph capture)
  Subsequent runs: 0.41 ms (graph replay)
```
