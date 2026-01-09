# Exercise 02: Interchangeable Attention Kernels

## Overview

In this exercise, you will implement attention kernels in multiple languages (Triton, CUDA, and TileLang) and make them interchangeable via FlashInfer-Bench's `apply()` mechanism.

**Difficulty**: Advanced
**Estimated Time**: 4-6 hours
**Prerequisites**: Chapter 07 (Triton), Chapter 08 (TileLang), Chapter 09 (Attention)

## Learning Objectives

1. Implement a simplified FlashAttention in Triton
2. Implement the same algorithm in CUDA
3. Implement the same algorithm in TileLang
4. Register all implementations as Solutions in FlashInfer-Bench
5. Use `apply()` to dynamically select the best kernel at runtime
6. Benchmark and compare performance across backends

## Problem Statement

You will implement a **Mini FlashAttention** kernel for the prefill phase that:
- Supports causal masking
- Uses online softmax (memory-efficient)
- Works with GQA (grouped query attention)

### Kernel Specification

```python
Definition:
    name: "mini_flash_attention_prefill"
    op_type: "gqa_ragged"

    axes:
        batch: var
        seq_q: var
        seq_kv: var
        num_heads: const(32)
        num_kv_heads: const(8)
        head_dim: const(128)

    inputs:
        q: [batch, seq_q, num_heads, head_dim] float16
        k: [batch, seq_kv, num_kv_heads, head_dim] float16
        v: [batch, seq_kv, num_kv_heads, head_dim] float16
        causal: scalar bool

    outputs:
        out: [batch, seq_q, num_heads, head_dim] float16
```

### Algorithm (Online Softmax FlashAttention)

```
ONLINE SOFTMAX FLASHATTENTION
=============================

For each Q tile (size BLOCK_Q):
    Initialize: O_acc = 0, m = -inf, l = 0

    For each KV tile (size BLOCK_KV):
        # Load tiles
        Q_tile = Q[q_start:q_end, :]
        K_tile = K[kv_start:kv_end, :]
        V_tile = V[kv_start:kv_end, :]

        # Compute attention scores
        S = Q_tile @ K_tile.T * scale

        # Apply causal mask if needed
        if causal:
            S = mask_future(S, q_start, kv_start)

        # Online softmax update
        m_new = max(m, rowmax(S))
        P = exp(S - m_new)
        l_new = l * exp(m - m_new) + rowsum(P)

        # Update output accumulator
        O_acc = O_acc * exp(m - m_new) + P @ V_tile

        m = m_new
        l = l_new

    # Final normalization
    O = O_acc / l
```

## Tasks

### Task 1: Create the Definition (10 points)

Create a FlashInfer-Bench Definition for the attention kernel.

**File**: `starter/definition.py`

```python
# TODO: Complete the Definition
from flashinfer_bench.data import Definition, AxisConst, AxisVar, TensorSpec, DType

attention_def = Definition(
    name="mini_flash_attention_prefill",
    op_type="gqa_ragged",
    axes={
        # TODO: Define axes
    },
    inputs={
        # TODO: Define inputs
    },
    outputs={
        # TODO: Define outputs
    },
    reference='''
# TODO: Implement reference using PyTorch
''',
    tags=["stage:prefill", "attention:flash"]
)
```

### Task 2: Implement Triton Version (25 points)

Implement the attention kernel in Triton.

**File**: `starter/triton_attention.py`

```python
import torch
import triton
import triton.language as tl

@triton.jit
def flash_attention_kernel(
    Q, K, V, Out,
    stride_qb, stride_qq, stride_qh, stride_qd,
    stride_kb, stride_kk, stride_kh, stride_kd,
    stride_vb, stride_vk, stride_vh, stride_vd,
    stride_ob, stride_oq, stride_oh, stride_od,
    seq_q, seq_kv, scale, causal: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    # TODO: Implement the kernel
    # Hints:
    # 1. Use tl.program_id() for batch, head, and q_tile indices
    # 2. Load Q tile once, iterate over KV tiles
    # 3. Track m (max), l (sum) for online softmax
    # 4. Apply causal mask by comparing indices
    pass

def run(q, k, v, causal=True):
    # TODO: Launch kernel with appropriate grid and block sizes
    pass
```

**Hints**:
- Use `tl.dot()` for matrix multiplication
- Use `tl.where()` for causal masking
- Track `m` and `l` per row for online softmax

### Task 3: Implement CUDA Version (25 points)

Implement the same algorithm in CUDA.

**File**: `starter/cuda_attention/kernel.cu`

```cuda
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <int BLOCK_Q, int BLOCK_KV, int HEAD_DIM>
__global__ void flash_attention_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ Out,
    int seq_q, int seq_kv,
    float scale, bool causal
) {
    // TODO: Implement the kernel
    // Hints:
    // 1. Use shared memory for Q, K, V tiles
    // 2. Use warp-level operations for reductions
    // 3. Use __hfma2 for fused multiply-add on half2
}
```

**Hints**:
- Allocate shared memory for tiles: `extern __shared__ char smem[];`
- Use `__syncthreads()` between loads and computes
- Vectorize with `half2` for 2x throughput

### Task 4: Implement TileLang Version (20 points)

Implement the same algorithm in TileLang.

**File**: `starter/tilelang_attention.py`

```python
import tilelang as tl
from tilelang import Kernel, Tensor, constexpr

@tl.kernel
def flash_attention_kernel(
    Q: Tensor[B, S_Q, H, D, tl.float16],
    K: Tensor[B, S_KV, H_KV, D, tl.float16],
    V: Tensor[B, S_KV, H_KV, D, tl.float16],
    Out: Tensor[B, S_Q, H, D, tl.float16],
    scale: float,
    causal: bool,
    BLOCK_Q: constexpr[int],
    BLOCK_KV: constexpr[int],
):
    # TODO: Implement the kernel
    # TileLang makes tiling explicit and clean
    pass
```

**Hints**:
- Use `T.tile()` for explicit tiling
- Use `T.gemm()` for QK^T and PV
- TileLang auto-handles shared memory

### Task 5: Register Solutions and Benchmark (10 points)

Register all implementations as Solutions and run benchmarks.

**File**: `starter/register_and_benchmark.py`

```python
from flashinfer_bench.data import Solution, TraceSet, Workload
from flashinfer_bench import Benchmark

# TODO: Create TraceSet with definition and all solutions
# TODO: Create workloads for different configs
# TODO: Run benchmarks and generate traces
```

### Task 6: Deploy with Apply (10 points)

Use `apply()` to make kernels interchangeable at runtime.

**File**: `starter/deploy.py`

```python
from flashinfer_bench import enable_apply, apply

# TODO: Enable apply runtime
# TODO: Create decorated function that routes to best kernel
# TODO: Demonstrate switching between backends
```

## Evaluation Criteria

| Criterion | Points |
|-----------|--------|
| Definition correctness | 10 |
| Triton implementation correctness | 15 |
| Triton performance (>50% of FlashInfer) | 10 |
| CUDA implementation correctness | 15 |
| CUDA performance (>60% of FlashInfer) | 10 |
| TileLang implementation correctness | 15 |
| TileLang performance (>40% of FlashInfer) | 5 |
| Solution registration and benchmarking | 10 |
| Apply deployment working | 10 |
| **Total** | **100** |

## Test Configurations

Your implementation must pass correctness tests for:

| Config | batch | seq_q | seq_kv | Causal |
|--------|-------|-------|--------|--------|
| Small | 1 | 128 | 128 | True |
| Medium | 4 | 512 | 512 | True |
| Large | 8 | 2048 | 2048 | True |
| Non-causal | 4 | 512 | 512 | False |
| Asymmetric | 4 | 128 | 2048 | True |

Correctness threshold: `max_relative_error < 1e-2` (FP16 tolerance)

## Performance Targets

Minimum acceptable performance relative to FlashInfer baseline:

| Backend | Minimum | Target |
|---------|---------|--------|
| Triton | 50% | 70% |
| CUDA | 60% | 80% |
| TileLang | 40% | 60% |

## Hints and Resources

### Online Softmax Derivation

```
Standard softmax:
  P = exp(S - max(S)) / sum(exp(S - max(S)))
  O = P @ V

Online (incremental) version:
  After processing KV tiles 0..t:
    m_t = max over all tiles seen
    l_t = sum(exp(S_i - m_t)) for all tiles
    O_t = sum(P_i @ V_i) for all tiles, rescaled by exp(m_i - m_t)

  Update for new tile t+1:
    m_{t+1} = max(m_t, max(S_{t+1}))
    l_{t+1} = l_t * exp(m_t - m_{t+1}) + sum(exp(S_{t+1} - m_{t+1}))
    O_{t+1} = O_t * exp(m_t - m_{t+1}) + P_{t+1} @ V_{t+1}
```

### Causal Mask Logic

```python
# For query position q and key position k:
# Allow attention if k <= q (in absolute positions)
# In tile coordinates:
#   q_abs = q_tile_start + q_local
#   k_abs = kv_tile_start + k_local
#   mask = k_abs <= q_abs
```

### Reference Implementations

- FlashInfer prefill: `flashinfer/include/flashinfer/attention/prefill.cuh`
- Triton Flash Attention: `triton/python/tutorials/06-fused-attention.py`
- TileLang attention: `tilelang/examples/flash_attention.py`

## Submission

Submit a directory containing:
```
submission/
├── definition.py          # Task 1
├── triton_attention.py    # Task 2
├── cuda_attention/        # Task 3
│   ├── kernel.cu
│   └── CMakeLists.txt
├── tilelang_attention.py  # Task 4
├── register_and_benchmark.py  # Task 5
├── deploy.py              # Task 6
├── traces.json            # Benchmark results
└── report.md              # Analysis and findings
```

## Bonus Challenges

1. **GQA Support** (+10 points): Handle num_heads != num_kv_heads
2. **Paged KV Cache** (+15 points): Support non-contiguous KV storage
3. **Sliding Window** (+10 points): Add sliding window attention support
4. **FP8 Support** (+10 points): Add FP8 quantized version
