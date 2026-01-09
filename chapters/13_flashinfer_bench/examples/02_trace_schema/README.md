# Example 02: FlashInfer Trace Schema

## Overview

This example demonstrates how to define kernels using the FlashInfer Trace Schema and create solutions in multiple languages.

## Files

- `create_definition.py` - Creating a Definition for RMSNorm
- `create_solutions.py` - Implementing solutions in Triton, CUDA, TileLang
- `create_workload.py` - Creating test workloads
- `run_benchmark.py` - Running benchmarks and generating traces
- `kernels/` - Kernel implementations
  - `triton_rmsnorm.py`
  - `cuda_rmsnorm/kernel.cu`
  - `tilelang_rmsnorm.py`

## Key Concepts

### 1. Definition Schema

A Definition describes **what** a kernel computes:

```python
from flashinfer_bench.data import (
    Definition, AxisConst, AxisVar, TensorSpec, DType
)

# Define RMSNorm operation
rmsnorm_def = Definition(
    name="rmsnorm_d4096",
    op_type="norm",

    # Axes: compile-time constants vs runtime variables
    axes={
        "batch": AxisVar(),              # Determined at runtime
        "hidden": AxisConst(value=4096)  # Fixed at compile time
    },

    # Input specifications
    inputs={
        "x": TensorSpec(shape=["batch", "hidden"], dtype=DType.FLOAT16),
        "weight": TensorSpec(shape=["hidden"], dtype=DType.FLOAT16),
        "eps": TensorSpec(shape=None, dtype=DType.FLOAT32),  # Scalar
    },

    # Output specifications
    outputs={
        "out": TensorSpec(shape=["batch", "hidden"], dtype=DType.FLOAT16),
    },

    # Reference implementation (ground truth)
    reference='''
import torch

def run(x, weight, eps):
    """RMSNorm: x * rsqrt(mean(x^2) + eps) * weight"""
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight
''',

    # Searchable tags
    tags=["stage:inference", "op:norm", "status:verified"],

    # Optional constraints
    constraints=["hidden % 128 == 0"]  # For vectorization
)
```

### 2. Solution Schema

A Solution is a **concrete implementation** of a Definition:

```python
from flashinfer_bench.data import (
    Solution, BuildSpec, SourceFile, SupportedLanguages
)

triton_solution = Solution(
    name="rmsnorm_triton_warp_v1",
    definition="rmsnorm_d4096",
    author="tutorial",

    spec=BuildSpec(
        language=SupportedLanguages.TRITON,
        target_hardware=["NVIDIA_A100", "NVIDIA_H100"],
        entry_point="kernel.py::run",
        dependencies=["triton >= 2.3", "torch >= 2.0"],
        destination_passing_style=False,  # Returns output
    ),

    sources=[
        SourceFile(
            path="kernel.py",
            content=open("kernels/triton_rmsnorm.py").read()
        )
    ],

    description="Triton RMSNorm with warp-level reduction"
)
```

### 3. Workload Schema

A Workload defines a **test case**:

```python
from flashinfer_bench.data import Workload, RandomInput, ScalarInput

workload = Workload(
    # Concrete axis values
    axes={"batch": 32, "hidden": 4096},

    # Input data generation
    inputs={
        "x": RandomInput(dtype="float16"),
        "weight": RandomInput(dtype="float16"),
        "eps": ScalarInput(value=1e-6),
    }
)
```

### 4. Trace Schema

A Trace records **benchmark results**:

```python
from flashinfer_bench.data import Trace, Evaluation

trace = Trace(
    definition="rmsnorm_d4096",
    workload=workload,
    solution="rmsnorm_triton_warp_v1",

    evaluation=Evaluation(
        status="PASSED",
        max_relative_error=1.2e-5,
        max_absolute_error=3.4e-4,
        latency_ms=0.023,
        reference_latency_ms=0.045,
        speedup_factor=1.96,
        hardware="NVIDIA_A100_80GB",
        timestamp="2026-01-09T10:30:00Z"
    )
)
```

## Multi-Language Implementations

### Triton (`kernels/triton_rmsnorm.py`)

```python
import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(
    x_ptr, w_ptr, out_ptr,
    eps,
    hidden,
    BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < hidden

    # Load input row
    x = tl.load(x_ptr + row * hidden + cols, mask=mask, other=0.0)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)

    # Compute variance
    x_sq = x * x
    var = tl.sum(x_sq, axis=0) / hidden

    # RMSNorm
    rstd = 1.0 / tl.sqrt(var + eps)
    out = x * rstd * w

    tl.store(out_ptr + row * hidden + cols, out, mask=mask)

def run(x, weight, eps):
    batch, hidden = x.shape
    out = torch.empty_like(x)

    BLOCK = triton.next_power_of_2(hidden)
    grid = (batch,)

    rmsnorm_kernel[grid](
        x, weight, out,
        eps, hidden,
        BLOCK=BLOCK
    )
    return out
```

### CUDA (`kernels/cuda_rmsnorm/kernel.cu`)

```cuda
#include <cuda_fp16.h>

template <int BLOCK_SIZE>
__device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template <int BLOCK_SIZE>
__global__ void rmsnorm_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ out,
    float eps,
    int hidden
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden; i += BLOCK_SIZE) {
        float val = __half2float(x[row * hidden + i]);
        sum_sq += val * val;
    }

    // Block-level reduction
    __shared__ float shared_sum[32];
    int lane = tid % 32;
    int warp_id = tid / 32;

    sum_sq = warp_reduce_sum<32>(sum_sq);
    if (lane == 0) shared_sum[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (tid < (BLOCK_SIZE / 32)) ? shared_sum[lane] : 0.0f;
        sum_sq = warp_reduce_sum<32>(sum_sq);
    }
    __syncthreads();

    float rstd = rsqrtf(shared_sum[0] / hidden + eps);

    // Apply normalization
    for (int i = tid; i < hidden; i += BLOCK_SIZE) {
        float val = __half2float(x[row * hidden + i]);
        float w = __half2float(weight[i]);
        out[row * hidden + i] = __float2half(val * rstd * w);
    }
}
```

## Running the Examples

```bash
# Create a definition
python create_definition.py

# Create solutions in multiple languages
python create_solutions.py --languages triton,cuda,tilelang

# Create test workloads
python create_workload.py --batch-sizes 1,8,32,128

# Run benchmarks
python run_benchmark.py --output traces.json
```

## Expected Output

```
Definition: rmsnorm_d4096
  Axes: batch (var), hidden (const: 4096)
  Inputs: x [batch, 4096] fp16, weight [4096] fp16, eps scalar
  Outputs: out [batch, 4096] fp16

Solutions:
  1. rmsnorm_triton_warp_v1 (Triton)
  2. rmsnorm_cuda_v1 (CUDA)
  3. rmsnorm_tilelang_v1 (TileLang)

Benchmark Results (batch=32):
  Reference:     0.045 ms
  Triton:        0.023 ms (1.96x speedup)
  CUDA:          0.019 ms (2.37x speedup)
  TileLang:      0.025 ms (1.80x speedup)
```
