# Multi-Language LLM Serving Kernel Implementation Plan

## Executive Summary

This plan extends the kernel tutorials to include multi-language kernel implementations that enable interchangeable usage for LLM serving tasks.

**Key Update**: A new **Chapter 13: FlashInfer and FlashInfer-Bench** has been created to teach the FlashInfer-Bench framework for kernel interchangeability. This chapter provides the infrastructure for dynamically substituting kernels at runtime.

Based on analysis of:
- Current 12-chapter tutorial structure
- **FlashInfer paper (arXiv:2501.01005)** and **FlashInfer-Bench paper (arXiv:2601.00227)**
- Hands-on learning insights from Mini-SGLang and FlashInfer
- Existing patterns from Triton (Ch.07), TileLang (Ch.08), and Capstone (Ch.11)

## Goal

**"At the end of the chapters, we should have kernels in multiple languages (DSL, Triton, TileLang) to be used interchangeably for the LLM serving task."**

---

## Chapter 13: FlashInfer and FlashInfer-Bench (NEW)

This new chapter provides the foundation for interchangeable kernels:

### Contents

```
chapters/13_flashinfer_bench/
├── README.md                          # Chapter overview
├── examples/
│   ├── 01_flashinfer_basics/          # FlashInfer API usage
│   ├── 02_trace_schema/               # Defining kernels and solutions
│   ├── 03_apply_mechanism/            # Dynamic kernel substitution
│   ├── 04_custom_kernels/             # Multi-language implementation
│   └── 05_integration/                # SGLang/vLLM integration
├── exercises/
│   ├── 01_triton_rmsnorm/             # Implement and register RMSNorm
│   └── 02_interchangeable_attention/  # Full multi-language attention
└── profiling/
```

### Key Concepts Covered

1. **FlashInfer Architecture**: Block-sparse format, JIT compilation, Plan-Run pattern
2. **FlashInfer Trace Schema**: Definition, Solution, Workload, Trace
3. **Apply Mechanism**: Runtime kernel routing without code changes
4. **Multi-Backend Compilation**: Triton, CUDA, TileLang support
5. **Integration**: Patching serving systems (SGLang, vLLM)

### Main Exercise: Interchangeable Attention

Students implement Mini FlashAttention in:
- **Triton**: ~150 lines
- **CUDA**: ~300 lines
- **TileLang**: ~80 lines

Then register all as Solutions and use `apply()` for dynamic selection.

---

## Key Kernels from Hands-On Learning

| Kernel Category | Kernel | Characteristics | Source |
|----------------|--------|-----------------|--------|
| **Data Movement** | Index (embedding lookup) | Memory-bound, warp-level vectorized copy | Mini-SGLang |
| **Data Movement** | Store (KV cache scatter) | Memory-bound, coalesced writes | Mini-SGLang |
| **Attention** | Prefill (FlashAttention-2) | Compute-bound, online softmax, TC utilization | FlashInfer |
| **Attention** | Decode (Split-K) | Memory-bound (87% HBM), batch parallelism | FlashInfer |
| **Normalization** | RMSNorm | Memory-bound, warp shuffle reduction | FlashInfer |
| **Normalization** | Fused Add+RMSNorm | 40% memory traffic reduction | FlashInfer |
| **Position Encoding** | RoPE | In-place variants, Llama 3.1 scaling | FlashInfer |
| **Communication** | AllReduce | Symmetric memory optimization | Mini-SGLang |

---

## Proposed Structure

### Organization: Extend Chapter 12 with New Example 5

Rather than create a new chapter, extend Chapter 12 with **Example 5: Multi-Language Kernel Implementations**.

### Directory Structure

```
chapters/12_llm_serving/
├── examples/
│   ├── 01_mini_sglang_kernels/     (existing)
│   ├── 02_flashinfer_attention/    (existing)
│   ├── 03_kv_cache_management/     (existing)
│   ├── 04_distributed_inference/   (existing)
│   └── 05_multi_lang_kernels/      (NEW)
│       ├── README.md
│       ├── common/
│       │   ├── interface.py        # Common Python interface
│       │   ├── benchmark.py        # Unified benchmarking
│       │   └── profiling.py        # NCU/nsys helpers
│       ├── kernels/
│       │   ├── embedding/
│       │   │   ├── cuda/           # embedding_cuda.cu
│       │   │   ├── triton/         # embedding_triton.py
│       │   │   ├── tilelang/       # embedding_tilelang.py
│       │   │   └── test_embedding.py
│       │   ├── kv_store/
│       │   │   ├── cuda/
│       │   │   ├── triton/
│       │   │   ├── tilelang/
│       │   │   └── test_kv_store.py
│       │   ├── rmsnorm/
│       │   │   ├── cuda/
│       │   │   ├── triton/
│       │   │   ├── tilelang/
│       │   │   └── test_rmsnorm.py
│       │   ├── rope/
│       │   │   ├── cuda/
│       │   │   ├── triton/
│       │   │   ├── tilelang/
│       │   │   └── test_rope.py
│       │   └── attention/
│       │       ├── cuda/           # Mini FlashAttention
│       │       ├── triton/         # Triton FA implementation
│       │       ├── tilelang/       # TileLang attention
│       │       └── test_attention.py
│       ├── serving_demo/
│       │   ├── kernel_registry.py  # Dynamic kernel selection
│       │   ├── mini_inference.py   # End-to-end demo
│       │   └── config.yaml         # Kernel backend configuration
│       └── analysis/
│           ├── process_analysis.md
│           └── hardware_analysis.md
├── exercises/
│   ├── 01_custom_embedding/        (existing)
│   ├── 02_kv_store_optimization/   (existing)
│   └── 03_interchangeable_kernels/ (NEW)
│       ├── problem.md
│       ├── starter/
│       └── solution/
└── README.md                       (UPDATE)
```

---

## Kernel Implementation Specifications

### 1. Embedding/Index Kernel

**Purpose**: Lookup embeddings from vocabulary table given token IDs.

**Interface** (all languages):
```python
def embedding_lookup(
    weight: Tensor,      # [vocab_size, hidden_dim]
    indices: Tensor,     # [batch_tokens]
    output: Tensor,      # [batch_tokens, hidden_dim]
) -> None
```

**Implementations**:
- **CUDA**: Warp-level vectorized copy (from Mini-SGLang pattern)
- **Triton**: Block-based gather with `tl.load` indirect access
- **TileLang**: Fragment-based with `T.copy()` indirect indexing

**Performance Target**: >1.5 TB/s memory bandwidth on A100

---

### 2. KV Store Kernel

**Purpose**: Scatter-write K/V tensors to paged KV cache.

**Interface**:
```python
def kv_store(
    k_input: Tensor,     # [num_tokens, num_kv_heads, head_dim]
    v_input: Tensor,     # [num_tokens, num_kv_heads, head_dim]
    k_cache: Tensor,     # [num_pages, num_kv_heads, head_dim]
    v_cache: Tensor,     # [num_pages, num_kv_heads, head_dim]
    page_indices: Tensor # [num_tokens]
) -> None
```

**Implementations**:
- **CUDA**: Warp-level scatter with vectorized store
- **Triton**: Block-based store with masking
- **TileLang**: Tile copy with indirect target

**Performance Target**: >1.3 TB/s memory bandwidth

---

### 3. RMSNorm Kernel

**Purpose**: Root Mean Square Layer Normalization.

**Interface**:
```python
def rmsnorm(
    input: Tensor,       # [batch, hidden_dim]
    weight: Tensor,      # [hidden_dim]
    output: Tensor,      # [batch, hidden_dim]
    eps: float = 1e-6
) -> None

def fused_add_rmsnorm(
    input: Tensor,       # [batch, hidden_dim] - modified in-place
    residual: Tensor,    # [batch, hidden_dim] - modified in-place
    weight: Tensor,      # [hidden_dim]
    eps: float = 1e-6
) -> None
```

**Implementations**:
- **CUDA**: Warp shuffle reduction, vectorized load/store
- **Triton**: `tl.sum` reduction, block-based processing
- **TileLang**: Fragment-level reduction with `T.reduce()`

**Performance Target**: >1.8 TB/s memory bandwidth

---

### 4. RoPE Kernel

**Purpose**: Rotary Position Embedding.

**Interface**:
```python
def apply_rope(
    query: Tensor,       # [seq_len, num_q_heads, head_dim]
    key: Tensor,         # [seq_len, num_kv_heads, head_dim]
    cos_cache: Tensor,   # [max_seq_len, head_dim/2]
    sin_cache: Tensor,   # [max_seq_len, head_dim/2]
    positions: Tensor,   # [seq_len]
) -> Tuple[Tensor, Tensor]  # Modified Q, K
```

**Implementations**:
- **CUDA**: In-place rotation with vectorized access
- **Triton**: Element-wise with position indexing
- **TileLang**: Tile-based rotation

**Performance Target**: >1.5 TB/s memory bandwidth

---

### 5. Mini Attention Kernel

**Purpose**: Simplified FlashAttention for educational purposes.

**Interface**:
```python
def attention_forward(
    query: Tensor,       # [batch, seq_q, num_heads, head_dim]
    key: Tensor,         # [batch, seq_kv, num_heads, head_dim]
    value: Tensor,       # [batch, seq_kv, num_heads, head_dim]
    output: Tensor,      # [batch, seq_q, num_heads, head_dim]
    causal: bool = True,
    scale: float = None,
) -> None
```

**Implementations**:
- **CUDA**: Simplified FA-2 with online softmax, fixed tile sizes
- **Triton**: Based on official Triton FA tutorial
- **TileLang**: Clean tile-centric implementation (~80 lines)

**Performance Target**: >70% of FlashInfer/FlashAttention-2 throughput

---

## Common Interface Design

### Kernel Registry (`kernel_registry.py`)

```python
from typing import Protocol, Dict, Callable
from enum import Enum

class KernelBackend(Enum):
    CUDA = "cuda"
    TRITON = "triton"
    TILELANG = "tilelang"
    FLASHINFER = "flashinfer"  # Reference

class KernelRegistry:
    """Dynamic kernel selection for interchangeable usage"""

    _backends: Dict[str, Dict[KernelBackend, Callable]] = {}

    @classmethod
    def register(cls, kernel_name: str, backend: KernelBackend):
        """Decorator to register kernel implementations"""
        def decorator(func):
            cls._backends.setdefault(kernel_name, {})[backend] = func
            return func
        return decorator

    @classmethod
    def get_kernel(cls, kernel_name: str, backend: KernelBackend):
        """Get kernel implementation for specified backend"""
        return cls._backends[kernel_name][backend]

    @classmethod
    def run_kernel(cls, kernel_name: str, backend: KernelBackend, *args, **kwargs):
        """Run kernel with specified backend"""
        kernel_fn = cls.get_kernel(kernel_name, backend)
        return kernel_fn(*args, **kwargs)
```

### Usage Example

```python
from kernel_registry import KernelRegistry, KernelBackend

# Select backend via configuration
backend = KernelBackend.TRITON

# Run embedding lookup
KernelRegistry.run_kernel("embedding", backend, weight, indices, output)

# Run attention
KernelRegistry.run_kernel("attention", backend, q, k, v, output, causal=True)

# Easy switching for benchmarking
for backend in KernelBackend:
    result = KernelRegistry.run_kernel("rmsnorm", backend, input, weight, output)
    benchmark(result)
```

---

## Expected Performance Targets

| Kernel | CUDA Baseline | Triton Target | TileLang Target |
|--------|---------------|---------------|-----------------|
| Embedding | 100% | >90% | >85% |
| KV Store | 100% | >85% | >80% |
| RMSNorm | 100% | >90% | >85% |
| RoPE | 100% | >85% | >80% |
| Attention | 100% | >80% | >75% |

## Expected Code Complexity (Lines of Code)

| Kernel | CUDA | Triton | TileLang | Ratio |
|--------|------|--------|----------|-------|
| Embedding | ~80 | ~40 | ~30 | 2.7:1.3:1 |
| KV Store | ~60 | ~35 | ~25 | 2.4:1.4:1 |
| RMSNorm | ~100 | ~50 | ~35 | 2.9:1.4:1 |
| RoPE | ~70 | ~40 | ~30 | 2.3:1.3:1 |
| Attention | ~300 | ~150 | ~80 | 3.8:1.9:1 |

---

## Implementation Phases

### Phase 1: Infrastructure
- Create directory structure
- Implement common interface and registry
- Set up benchmark framework
- Create test harness

### Phase 2: Memory-Bound Kernels
1. **Embedding kernel**: CUDA -> Triton -> TileLang
2. **KV Store kernel**: CUDA -> Triton -> TileLang
3. **RMSNorm kernel**: CUDA -> Triton -> TileLang
4. **RoPE kernel**: CUDA -> Triton -> TileLang

### Phase 3: Compute Kernel
1. **Mini Attention**: CUDA (simplified FA)
2. **Mini Attention**: Triton (from tutorial)
3. **Mini Attention**: TileLang (~80 lines)

### Phase 4: Integration
- Serving demo with kernel switching
- Complete benchmarks across all configs
- Generate analysis documentation
- Create exercise with solution

### Phase 5: Documentation
- Update Chapter 12 README
- Write Example 5 README
- Complete process/hardware analysis

---

## Key References

- **Mini-SGLang kernel-dev-guide**: Index, Store kernel patterns
- **FlashInfer kernel-dev-guide**: Attention, RMSNorm, RoPE patterns
- **SGLang kernel-dev-guide**: Execution flow, backend selection
- **Chapter 07 (Triton)**: Triton implementation patterns
- **Chapter 08 (TileLang)**: TileLang implementation patterns
- **Chapter 11 (Capstone)**: Benchmark framework pattern
