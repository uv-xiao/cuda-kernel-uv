# Example 04: Custom Kernel Implementation

## Overview

This example demonstrates how to implement custom kernels in multiple languages and register them with FlashInfer-Bench for use via the `apply()` mechanism.

## Files

- `define_kernel.py` - Create kernel Definition
- `impl_triton.py` - Triton implementation
- `impl_cuda.py` - CUDA implementation
- `impl_tilelang.py` - TileLang implementation
- `register_solutions.py` - Register all solutions
- `test_correctness.py` - Verify implementations
- `benchmark_all.py` - Compare performance

## Workflow

```
CUSTOM KERNEL WORKFLOW
======================

1. Define         2. Implement        3. Register       4. Deploy
   |                  |                   |                |
   v                  v                   v                v
Definition  -->  Solutions     -->   TraceSet   -->   apply()
(schema)      (Triton/CUDA/...)   (benchmark)     (production)
```

## Step 1: Define the Kernel

```python
# define_kernel.py
from flashinfer_bench.data import (
    Definition, AxisConst, AxisVar, TensorSpec, DType
)

# Define fused add + RMSNorm (common in LLM inference)
fused_add_rmsnorm_def = Definition(
    name="fused_add_rmsnorm_d4096",
    op_type="norm",

    axes={
        "batch": AxisVar(),
        "hidden": AxisConst(value=4096),
    },

    inputs={
        "input": TensorSpec(shape=["batch", "hidden"], dtype=DType.FLOAT16),
        "residual": TensorSpec(shape=["batch", "hidden"], dtype=DType.FLOAT16),
        "weight": TensorSpec(shape=["hidden"], dtype=DType.FLOAT16),
        "eps": TensorSpec(shape=None, dtype=DType.FLOAT32),
    },

    outputs={
        # Two outputs: normalized result AND updated residual
        "out": TensorSpec(shape=["batch", "hidden"], dtype=DType.FLOAT16),
        "residual_out": TensorSpec(shape=["batch", "hidden"], dtype=DType.FLOAT16),
    },

    reference='''
import torch

def run(input, residual, weight, eps):
    """Fused add + RMSNorm: saves memory traffic"""
    # Add residual
    hidden_states = input + residual

    # RMSNorm
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    normed = hidden_states * torch.rsqrt(variance + eps) * weight

    # Return both normalized output and updated residual
    return normed, hidden_states
''',

    tags=["stage:inference", "op:fused_norm", "memory:optimized"],
)
```

## Step 2: Implement in Multiple Languages

### Triton Implementation

```python
# impl_triton.py
import torch
import triton
import triton.language as tl

@triton.jit
def fused_add_rmsnorm_kernel(
    input_ptr, residual_ptr, weight_ptr,
    out_ptr, residual_out_ptr,
    eps, hidden,
    BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < hidden

    # Load inputs
    inp = tl.load(input_ptr + row * hidden + cols, mask=mask)
    res = tl.load(residual_ptr + row * hidden + cols, mask=mask)
    w = tl.load(weight_ptr + cols, mask=mask)

    # Fused add
    hidden_states = inp + res

    # RMSNorm
    var = tl.sum(hidden_states * hidden_states, axis=0) / hidden
    rstd = 1.0 / tl.sqrt(var + eps)
    normed = hidden_states * rstd * w

    # Store outputs
    tl.store(out_ptr + row * hidden + cols, normed, mask=mask)
    tl.store(residual_out_ptr + row * hidden + cols, hidden_states, mask=mask)

def run(input, residual, weight, eps):
    batch, hidden = input.shape
    out = torch.empty_like(input)
    residual_out = torch.empty_like(residual)

    BLOCK = triton.next_power_of_2(hidden)
    grid = (batch,)

    fused_add_rmsnorm_kernel[grid](
        input, residual, weight,
        out, residual_out,
        eps, hidden,
        BLOCK=BLOCK
    )
    return out, residual_out
```

### CUDA Implementation

```cuda
// impl_cuda/kernel.cu
#include <cuda_fp16.h>

template <int BLOCK_SIZE>
__global__ void fused_add_rmsnorm_kernel(
    const half* __restrict__ input,
    const half* __restrict__ residual,
    const half* __restrict__ weight,
    half* __restrict__ out,
    half* __restrict__ residual_out,
    float eps,
    int hidden
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float smem[];

    // Phase 1: Fused add + compute sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden; i += BLOCK_SIZE) {
        float inp = __half2float(input[row * hidden + i]);
        float res = __half2float(residual[row * hidden + i]);
        float h = inp + res;

        // Store updated residual
        residual_out[row * hidden + i] = __float2half(h);
        smem[i] = h;  // Cache for phase 2

        sum_sq += h * h;
    }
    __syncthreads();

    // Block-level reduction
    sum_sq = block_reduce_sum<BLOCK_SIZE>(sum_sq);
    float rstd = rsqrtf(sum_sq / hidden + eps);

    // Phase 2: Apply normalization
    for (int i = tid; i < hidden; i += BLOCK_SIZE) {
        float h = smem[i];
        float w = __half2float(weight[i]);
        out[row * hidden + i] = __float2half(h * rstd * w);
    }
}
```

### TileLang Implementation

```python
# impl_tilelang.py
import tilelang as tl
from tilelang import Kernel, Tensor

@tl.kernel
def fused_add_rmsnorm_kernel(
    input: Tensor[M, N, tl.float16],
    residual: Tensor[M, N, tl.float16],
    weight: Tensor[N, tl.float16],
    out: Tensor[M, N, tl.float16],
    residual_out: Tensor[M, N, tl.float16],
    eps: float
):
    m = tl.program_id(0)

    # Load and fuse add
    inp = tl.load(input[m, :])
    res = tl.load(residual[m, :])
    hidden_states = inp + res

    # Store updated residual
    tl.store(residual_out[m, :], hidden_states)

    # RMSNorm
    var = tl.reduce(hidden_states * hidden_states, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    normed = hidden_states * rstd * tl.load(weight[:])

    tl.store(out[m, :], normed)

def run(input, residual, weight, eps):
    batch, hidden = input.shape
    out = torch.empty_like(input)
    residual_out = torch.empty_like(residual)

    fused_add_rmsnorm_kernel[(batch,)](
        input, residual, weight, out, residual_out, eps
    )
    return out, residual_out
```

## Step 3: Register Solutions

```python
# register_solutions.py
from flashinfer_bench.data import (
    Solution, BuildSpec, SourceFile, SupportedLanguages, TraceSet
)

# Create TraceSet
trace_set = TraceSet()

# Add definition
trace_set.add_definition(fused_add_rmsnorm_def)

# Add Triton solution
triton_solution = Solution(
    name="fused_add_rmsnorm_triton_v1",
    definition="fused_add_rmsnorm_d4096",
    author="tutorial",
    spec=BuildSpec(
        language=SupportedLanguages.TRITON,
        entry_point="kernel.py::run",
    ),
    sources=[SourceFile(path="kernel.py", content=triton_source)]
)
trace_set.add_solution(triton_solution)

# Add CUDA solution
cuda_solution = Solution(
    name="fused_add_rmsnorm_cuda_v1",
    definition="fused_add_rmsnorm_d4096",
    author="tutorial",
    spec=BuildSpec(
        language=SupportedLanguages.CUDA,
        entry_point="kernel.cu::run",
    ),
    sources=[SourceFile(path="kernel.cu", content=cuda_source)]
)
trace_set.add_solution(cuda_solution)

# Save
trace_set.save("custom_kernels.json")
```

## Step 4: Test and Benchmark

```python
# test_correctness.py
from flashinfer_bench import TraceSet, Benchmark

trace_set = TraceSet.load("custom_kernels.json")
benchmark = Benchmark(trace_set)

# Run all solutions against all workloads
results = benchmark.run_all()

for trace in results:
    print(f"{trace.solution}: {trace.evaluation.status}")
    if trace.evaluation.status == "PASSED":
        print(f"  Speedup: {trace.evaluation.speedup_factor:.2f}x")
        print(f"  Max error: {trace.evaluation.max_relative_error:.2e}")
```

## Step 5: Deploy with Apply

```python
# deploy.py
from flashinfer_bench import enable_apply, apply, TraceSet

trace_set = TraceSet.load("custom_kernels.json")
enable_apply(trace_set)

@apply("fused_add_rmsnorm_d4096")
def fused_add_rmsnorm(input, residual, weight, eps):
    # Fallback (only used if no solution matches)
    hidden_states = input + residual
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    normed = hidden_states * torch.rsqrt(variance + eps) * weight
    return normed, hidden_states

# Now uses best registered solution automatically
out, updated_residual = fused_add_rmsnorm(x, res, w, 1e-6)
```

## Memory Traffic Analysis

```
MEMORY TRAFFIC COMPARISON
=========================

Separate ops (add + rmsnorm):
  Read:  input (2BD) + residual (2BD) → temp
  Write: temp (2BD)
  Read:  temp (2BD) + weight (2D)
  Write: out (2BD)
  Total: 8BD + 2D bytes

Fused op:
  Read:  input (2BD) + residual (2BD) + weight (2D)
  Write: out (2BD) + residual_out (2BD)
  Total: 6BD + 2D bytes

Savings: 25% memory traffic reduction
```

## Running the Examples

```bash
# Define kernel
python define_kernel.py

# Implement in all languages
python impl_triton.py
python impl_cuda.py
python impl_tilelang.py

# Register solutions
python register_solutions.py

# Test correctness
python test_correctness.py

# Benchmark all
python benchmark_all.py --output results.json
```

## Expected Output

```
Custom Kernel: fused_add_rmsnorm_d4096
======================================

Solutions registered:
  - fused_add_rmsnorm_triton_v1
  - fused_add_rmsnorm_cuda_v1
  - fused_add_rmsnorm_tilelang_v1

Correctness Tests (batch=32):
  Triton:   PASSED (max_rel_error=1.2e-5)
  CUDA:     PASSED (max_rel_error=8.3e-6)
  TileLang: PASSED (max_rel_error=1.5e-5)

Performance (batch=32, hidden=4096):
  Reference: 0.089 ms
  Triton:    0.041 ms (2.17x speedup)
  CUDA:      0.035 ms (2.54x speedup)
  TileLang:  0.045 ms (1.98x speedup)

Memory traffic:
  Separate ops: 1.05 MB
  Fused op:     0.79 MB (25% reduction)
```
