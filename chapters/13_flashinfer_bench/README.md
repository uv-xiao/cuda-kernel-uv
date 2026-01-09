# Chapter 13: FlashInfer and FlashInfer-Bench

## Overview

This chapter provides a comprehensive study of FlashInfer and its benchmarking/extensibility framework FlashInfer-Bench. You will learn how FlashInfer achieves high performance through composable kernel design, and how to use FlashInfer-Bench to create, benchmark, and deploy interchangeable kernel implementations.

**Prerequisites**: Chapters 07 (Triton), 08 (TileLang), 09 (Sparse Attention), 12 (LLM Serving)

## Learning Objectives

By the end of this chapter, you will be able to:

1. Understand FlashInfer's architecture and kernel design principles
2. Use the FlashInfer Trace Schema to define kernels and workloads
3. Implement custom kernels in multiple languages (Triton, CUDA, TileLang)
4. Use the `apply()` mechanism for dynamic kernel substitution
5. Benchmark and compare kernel implementations
6. Integrate custom kernels into serving systems (SGLang, vLLM)

---

## Part 1: FlashInfer Architecture

### Key Innovations (from MLSys 2025 Best Paper)

FlashInfer introduces several innovations for LLM inference kernels:

```
FLASHINFER ARCHITECTURE LAYERS
===============================

+-------------------------------------------------------------------------+
|                         PYTHON API LAYER                                 |
|  - BatchDecodeWithPagedKVCacheWrapper                                    |
|  - BatchPrefillWithPagedKVCacheWrapper                                   |
|  - Plan-Run pattern for CUDAGraph compatibility                          |
+-------------------------------------------------------------------------+
                                   |
                                   v
+-------------------------------------------------------------------------+
|                      JIT COMPILATION LAYER                               |
|  - ~1000+ attention variants (GQA, MLA, RoPE, ALiBi, sliding window)     |
|  - On-demand compilation instead of pre-compiling all combinations       |
|  - Two-level caching: memory + disk (~/.cache/flashinfer/)               |
+-------------------------------------------------------------------------+
                                   |
                                   v
+-------------------------------------------------------------------------+
|                   KERNEL LAYER (Header-Only CUDA)                        |
|  - Block-sparse format unifying dense/ragged/paged KV-cache              |
|  - Multiple backends: FA2, FA3 (Hopper), CUTLASS (Blackwell)             |
|  - Load-balanced scheduling across SMs                                   |
+-------------------------------------------------------------------------+
```

### Block-Sparse Format

FlashInfer unifies different KV-cache layouts under a single abstraction:

```cpp
// Unified KV-cache structure (from flashinfer/include/flashinfer/page.cuh)
template <typename DType, typename IdType>
struct paged_kv_t {
    DType* k_data, *v_data;          // Storage pointers
    IdType* indices;                  // Page indices (identity for dense)
    IdType* indptr;                   // CSR-style row pointers
    IdType* last_page_len;            // Tokens in last page per sequence
    uint_fastdiv page_size;           // Fast division for page computations
    uint32_t stride_page, stride_n, stride_h;
};
```

This enables the **same kernel code** to handle:
- Dense attention (contiguous memory)
- Ragged attention (variable-length sequences)
- Paged attention (non-contiguous pages)

### Plan-Run Pattern

```python
class BatchPrefillWithPagedKVCacheWrapper:
    def plan(self, qo_indptr, kv_indptr, ...):
        """CPU scheduling - dynamic, can vary per batch"""
        work_indptr = compute_balanced_partition(...)
        self._plan_info = pack_plan_info(work_indptr, ...)

    def run(self, q, k, v, o, ...):
        """GPU execution - static launch config for CUDAGraph"""
        kernel<<<num_sms, threads>>>(q, k, v, o, self._plan_info)
```

**Benefits**:
- Decouples scheduling from execution
- Enables CUDAGraph capture (static kernel configs)
- Load-balanced work distribution across SMs

### Backend Selection

FlashInfer supports multiple backends:

| Backend | Architecture | Features |
|---------|--------------|----------|
| FA2 | SM80+ (Ampere) | Default, well-optimized |
| FA3 | SM90+ (Hopper) | TMA, warp specialization |
| CUTLASS | SM100+ (Blackwell) | FMHA CUTLASS implementation |
| cuDNN | SM80+ | Integration with cuDNN |

```python
# Backend selection in SGLang
self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
    self.workspace_buffer, "NHD",
    backend="auto"  # or "fa2", "fa3", "cutlass"
)
```

---

## Part 2: FlashInfer-Bench Framework

### Overview

FlashInfer-Bench enables:
1. **Standardized kernel definition** via FlashInfer Trace Schema
2. **Dynamic kernel substitution** via `apply()` mechanism
3. **Multi-backend compilation** (Triton, CUDA, Python, TVM-FFI)
4. **Automated benchmarking** with correctness verification
5. **Integration with serving systems** (SGLang, vLLM)

```
FLASHINFER-BENCH WORKFLOW
=========================

  +-----------+     +------------+     +------------+     +-----------+
  | Define    |---->| Implement  |---->| Benchmark  |---->| Deploy    |
  | Kernel    |     | Solutions  |     | & Verify   |     | via apply |
  +-----------+     +------------+     +------------+     +-----------+
       |                  |                  |                  |
       v                  v                  v                  v
  Definition         Solution(s)          Traces           Production
  (schema)          (CUDA/Triton)      (latency, acc)       System
```

### FlashInfer Trace Schema

The schema consists of four core data types:

#### 1. Definition
Describes what a kernel computes:

```python
from flashinfer_bench.data import Definition, AxisConst, AxisVar, TensorSpec, DType

Definition(
    name="rmsnorm_d4096",
    op_type="norm",
    axes={
        "batch": AxisVar(),           # Runtime-determined
        "hidden": AxisConst(value=4096)  # Compile-time constant
    },
    inputs={
        "x": TensorSpec(shape=["batch", "hidden"], dtype=DType.FLOAT16),
        "weight": TensorSpec(shape=["hidden"], dtype=DType.FLOAT16),
        "eps": TensorSpec(shape=None, dtype=DType.FLOAT32),  # Scalar
    },
    outputs={
        "out": TensorSpec(shape=["batch", "hidden"], dtype=DType.FLOAT16),
    },
    reference='''
import torch
def run(x, weight, eps):
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight
''',
    tags=["stage:inference", "quantization:float16"]
)
```

#### 2. Solution
Concrete implementation of a Definition:

```python
from flashinfer_bench.data import Solution, BuildSpec, SourceFile, SupportedLanguages

Solution(
    name="rmsnorm_triton_v1",
    definition="rmsnorm_d4096",
    author="tutorial",
    spec=BuildSpec(
        language=SupportedLanguages.TRITON,
        target_hardware=["NVIDIA_A100", "NVIDIA_H100"],
        entry_point="kernel.py::run",
        dependencies=["triton >= 2.3", "torch"],
        destination_passing_style=False,  # Returns output
    ),
    sources=[
        SourceFile(
            path="kernel.py",
            content='''
import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, eps, hidden, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < hidden

    x = tl.load(x_ptr + row * hidden + cols, mask=mask)
    w = tl.load(w_ptr + cols, mask=mask)

    var = tl.sum(x * x, axis=0) / hidden
    rstd = 1.0 / tl.sqrt(var + eps)
    out = x * rstd * w

    tl.store(out_ptr + row * hidden + cols, out, mask=mask)

def run(x, weight, eps):
    batch, hidden = x.shape
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(hidden)
    rmsnorm_kernel[(batch,)](x, weight, out, eps, hidden, BLOCK)
    return out
'''
        )
    ]
)
```

#### 3. Workload
Concrete test configuration:

```python
from flashinfer_bench.data import Workload, RandomInput, ScalarInput

Workload(
    axes={"batch": 32, "hidden": 4096},
    inputs={
        "x": RandomInput(dtype="float16"),
        "weight": RandomInput(dtype="float16"),
        "eps": ScalarInput(value=1e-6),
    }
)
```

#### 4. Trace
Benchmark result:

```python
Trace(
    definition="rmsnorm_d4096",
    workload=workload,
    solution="rmsnorm_triton_v1",
    evaluation=Evaluation(
        status="PASSED",
        max_relative_error=1e-5,
        latency_ms=0.023,
        reference_latency_ms=0.045,
        speedup_factor=1.96,
    )
)
```

---

## Part 3: The `apply()` Mechanism

### How It Works

The `apply()` mechanism enables runtime kernel routing without code changes:

```
APPLY DISPATCH FLOW
===================

  User Code            Apply Runtime           Solution Registry
      |                      |                        |
      v                      v                        v
  apply("rmsnorm")  -->  Extract axes  -->  Lookup best solution
         |                   |                       |
         v                   v                       v
  (batch=32, hidden=4096)   ApplyKey         "rmsnorm_triton_v1"
                            |                        |
                            v                        v
                    Build solution  <--  Load from cache/compile
                            |
                            v
                    Execute & return
```

### Usage Modes

#### A. Decorator Mode
```python
from flashinfer_bench import apply

@apply("rmsnorm_d4096")
def rmsnorm(x, weight, eps):
    # Fallback implementation
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight

# When called, automatically routes to best registered solution
output = rmsnorm(x, weight, 1e-6)
```

#### B. Function Mode
```python
from flashinfer_bench import apply

def reference_rmsnorm(x, weight, eps):
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight

# Explicit dispatch
output = apply(
    "rmsnorm_d4096",
    args=(x, weight, 1e-6),
    fallback=reference_rmsnorm
)
```

### Enabling Apply Runtime

```python
from flashinfer_bench import enable_apply, TraceSet

# Load trace set with definitions and solutions
trace_set = TraceSet.load("path/to/traces")

# Enable apply runtime
enable_apply(trace_set)

# Now all @apply decorated functions route through the runtime
```

---

## Part 4: Multi-Language Kernel Implementation

### Supported Languages

| Language | Builder | Use Case |
|----------|---------|----------|
| Triton | TritonBuilder | Rapid prototyping, auto-tuning |
| CUDA | TVMFFIBuilder | Maximum performance |
| Python | PythonBuilder | Reference implementations |
| C++ | TorchBuilder | Integration with PyTorch |

### Example: RMSNorm in Three Languages

#### Triton Implementation
```python
# See examples/02_trace_schema/triton_rmsnorm.py
@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, eps, hidden, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < hidden

    x = tl.load(x_ptr + row * hidden + cols, mask=mask)
    w = tl.load(w_ptr + cols, mask=mask)

    var = tl.sum(x * x, axis=0) / hidden
    rstd = 1.0 / tl.sqrt(var + eps)
    out = x * rstd * w

    tl.store(out_ptr + row * hidden + cols, out, mask=mask)
```

#### CUDA Implementation
```cuda
// See examples/02_trace_schema/cuda_rmsnorm/kernel.cu
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

    // Warp-level reduction for variance
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden; i += BLOCK_SIZE) {
        float val = __half2float(x[row * hidden + i]);
        sum_sq += val * val;
    }
    sum_sq = warp_reduce_sum(sum_sq);

    // Compute rsqrt and apply
    float rstd = rsqrtf(sum_sq / hidden + eps);
    for (int i = tid; i < hidden; i += BLOCK_SIZE) {
        float val = __half2float(x[row * hidden + i]);
        float w = __half2float(weight[i]);
        out[row * hidden + i] = __float2half(val * rstd * w);
    }
}
```

#### TileLang Implementation
```python
# See examples/02_trace_schema/tilelang_rmsnorm.py
import tilelang as tl
from tilelang import Kernel, Tensor

@tl.kernel
def rmsnorm_kernel(
    x: Tensor[M, N, tl.float16],
    w: Tensor[N, tl.float16],
    out: Tensor[M, N, tl.float16],
    eps: float
):
    m = tl.program_id(0)

    # Load row
    x_row = tl.load(x[m, :])

    # Compute variance via reduction
    var = tl.reduce(x_row * x_row, axis=0) / N
    rstd = tl.rsqrt(var + eps)

    # Apply normalization
    result = x_row * rstd * tl.load(w[:])
    tl.store(out[m, :], result)
```

---

## Part 5: Integration with Serving Systems

### Patching FlashInfer Kernels

FlashInfer-Bench can patch FlashInfer functions to use optimized implementations:

```python
from flashinfer_bench.integration.flashinfer import install_flashinfer_integrations

# Install patches for all supported kernels
install_flashinfer_integrations()

# Now FlashInfer calls route through apply()
import flashinfer
out = flashinfer.norm.fused_add_rmsnorm(input, residual, weight, eps)
# ^ Automatically uses best registered solution
```

### Adapter Pattern

Each kernel type has an adapter that handles:
1. Argument extraction from runtime calls
2. Definition name resolution
3. Compatibility checking
4. Fallback to original implementation

```python
# Example adapter structure (from flashinfer_bench/integration/flashinfer/adapters/)
class RMSNormAdapter:
    def targets(self) -> List[PatchSpec]:
        return [
            PatchSpec(
                path="flashinfer.norm.fused_add_rmsnorm",
                kind="function",
                name="fused_add_rmsnorm",
                ctx_key="rmsnorm"
            )
        ]

    def make_wrapper(self, spec, orig):
        def wrapper(*args, **kwargs):
            # Extract definition name from tensor shapes
            def_name = resolve_rmsnorm_def(args)
            return apply(def_name, args=args, fallback=orig)
        return wrapper
```

---

## Examples

| Example | Description |
|---------|-------------|
| `01_flashinfer_basics/` | FlashInfer API usage: prefill, decode, backends |
| `02_trace_schema/` | Creating Definitions, Solutions, Workloads |
| `03_apply_mechanism/` | Using `apply()` for kernel routing |
| `04_custom_kernels/` | Implementing kernels in Triton, CUDA, TileLang |
| `05_integration/` | Integrating with SGLang/vLLM |

## Exercises

| Exercise | Description |
|----------|-------------|
| `01_triton_rmsnorm/` | Implement and benchmark a Triton RMSNorm kernel |
| `02_interchangeable_attention/` | Create interchangeable attention kernels |

---

## Key Files Reference

### FlashInfer Core
- `flashinfer/include/flashinfer/page.cuh` - KV-cache data structures
- `flashinfer/include/flashinfer/attention/variants.cuh` - Attention customization
- `flashinfer/flashinfer/jit/core.py` - JIT compilation system

### FlashInfer-Bench
- `flashinfer_bench/data/definition.py` - Definition schema
- `flashinfer_bench/data/solution.py` - Solution schema
- `flashinfer_bench/apply/apply_api.py` - Apply mechanism
- `flashinfer_bench/apply/runtime.py` - Dispatch engine
- `flashinfer_bench/compile/registry.py` - Builder registry

### Code Repositories
- FlashInfer: `/home/uvxiao/mlkb/code-repos/flashinfer/`
- FlashInfer-Bench: `/home/uvxiao/mlkb/code-repos/flashinfer-bench/`

---

## Reading Materials

### Papers
- [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference](https://arxiv.org/abs/2501.01005) (MLSys 2025 Best Paper)
- [FlashInfer-Bench: A Benchmark Suite for LLM Kernel Optimization](https://arxiv.org/abs/2601.00227)

### Documentation
- [FlashInfer Documentation](https://docs.flashinfer.ai/)
- [FlashInfer-Bench Docs](https://github.com/flashinfer-ai/flashinfer-bench/tree/main/docs)

### Related Chapters
- Chapter 07: Triton Kernel Design
- Chapter 08: TileLang & High-Level DSLs
- Chapter 09: Sparse Attention Kernels
- Chapter 12: LLM Serving Integration
