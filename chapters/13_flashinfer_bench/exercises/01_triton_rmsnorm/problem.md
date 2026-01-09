# Exercise 01: Triton RMSNorm Implementation

## Overview

In this exercise, you will implement an RMSNorm kernel in Triton, register it as a FlashInfer-Bench Solution, and benchmark it against the reference implementation.

**Difficulty**: Intermediate
**Estimated Time**: 2-3 hours
**Prerequisites**: Chapter 07 (Triton), Example 02 (Trace Schema)

## Learning Objectives

1. Implement a memory-bound kernel in Triton
2. Use warp-level reduction for computing variance
3. Register a Solution in FlashInfer-Bench
4. Benchmark and analyze kernel performance

## Problem Statement

RMSNorm (Root Mean Square Normalization) is a simplified version of LayerNorm used in modern LLMs (Llama, Mistral, etc.):

```
RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * weight
```

### Kernel Specification

```python
Definition:
    name: "rmsnorm_d4096"
    op_type: "norm"

    axes:
        batch: var
        hidden: const(4096)

    inputs:
        x: [batch, hidden] float16
        weight: [hidden] float16
        eps: scalar float32

    outputs:
        out: [batch, hidden] float16
```

## Tasks

### Task 1: Implement Triton Kernel (40 points)

**File**: `starter/triton_rmsnorm.py`

Implement the RMSNorm kernel with:
- Block-based processing (one row per program)
- Efficient reduction for computing sum of squares
- Vectorized load/store

```python
@triton.jit
def rmsnorm_kernel(
    x_ptr, weight_ptr, out_ptr,
    eps, hidden,
    BLOCK: tl.constexpr
):
    # TODO: Implement
    pass
```

### Task 2: Create Solution (20 points)

**File**: `starter/create_solution.py`

Create a FlashInfer-Bench Solution for your kernel:
- Set correct BuildSpec (language, entry_point, etc.)
- Include source file
- Add description

### Task 3: Benchmark (20 points)

**File**: `starter/benchmark.py`

Run benchmarks for:
- Batch sizes: 1, 8, 32, 128, 512
- Compare against PyTorch reference
- Generate Traces

### Task 4: Analysis (20 points)

**File**: `starter/analysis.md`

Answer:
1. What is the arithmetic intensity of RMSNorm?
2. Is your kernel compute-bound or memory-bound?
3. What percentage of peak HBM bandwidth do you achieve?
4. How does performance scale with batch size?

## Hints

### Reduction Pattern

```python
# Sum of squares across row
x = tl.load(x_ptr + row * hidden + cols, mask=mask)
x_sq = x * x
sum_sq = tl.sum(x_sq, axis=0)  # Reduction across columns
```

### Memory Efficiency

For hidden_dim=4096 with FP16:
- Input: 4096 * 2 = 8 KB per row
- Weight: 4096 * 2 = 8 KB (shared)
- Output: 4096 * 2 = 8 KB per row

Arithmetic intensity = FLOPs / Bytes ≈ 4 (memory-bound)

## Test Cases

| batch | Expected Latency (A100) |
|-------|------------------------|
| 1 | < 0.01 ms |
| 32 | < 0.02 ms |
| 128 | < 0.05 ms |
| 512 | < 0.15 ms |

Correctness: `max_relative_error < 1e-3`

## Submission

```
submission/
├── triton_rmsnorm.py
├── create_solution.py
├── benchmark.py
├── traces.json
└── analysis.md
```
