# Example 03: The Apply Mechanism

## Overview

This example demonstrates how to use FlashInfer-Bench's `apply()` mechanism for dynamic kernel substitution at runtime.

## Files

- `basic_apply.py` - Basic apply usage (decorator and function modes)
- `enable_runtime.py` - Setting up the apply runtime with TraceSet
- `custom_routing.py` - Custom key builders for workload fingerprinting
- `benchmark_apply.py` - Measuring apply overhead

## Key Concepts

### 1. Apply Dispatch Flow

```
APPLY() DISPATCH FLOW
=====================

  @apply("rmsnorm_d4096")          ApplyRuntime
  def rmsnorm(x, w, eps):              |
      return fallback(...)             v
            |                   +--------------+
            |                   | Extract axes |
            v                   | from inputs  |
    rmsnorm(x, w, eps)          +--------------+
            |                          |
            |                          v
            |                   +--------------+
            +-----------------> | Build ApplyKey|
                               | (batch=32,    |
                               |  hidden=4096) |
                               +--------------+
                                       |
                                       v
                               +--------------+
                               | Lookup best  |
                               | solution in  |
                               | ApplyTable   |
                               +--------------+
                                       |
                                       v
                               +--------------+
                               | Build/cache  |
                               | Runnable     |
                               +--------------+
                                       |
                                       v
                               +--------------+
                               | Execute and  |
                               | return result|
                               +--------------+
```

### 2. Basic Usage

#### Decorator Mode

```python
from flashinfer_bench import apply, enable_apply, TraceSet

# Load trace set with definitions and solutions
trace_set = TraceSet.load("path/to/traces")
enable_apply(trace_set)

# Decorate function - routes through apply runtime
@apply("rmsnorm_d4096")
def rmsnorm(x, weight, eps):
    """Fallback implementation"""
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight

# Call normally - automatically uses best solution
output = rmsnorm(x, weight, 1e-6)
```

#### Function Mode

```python
from flashinfer_bench import apply

# Direct dispatch without decorator
def reference_rmsnorm(x, weight, eps):
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight

# Explicit apply call
output = apply(
    "rmsnorm_d4096",
    args=(x, weight, 1e-6),
    fallback=reference_rmsnorm
)
```

### 3. Runtime Configuration

```python
from flashinfer_bench import (
    enable_apply, disable_apply,
    ApplyConfig, ApplyConfigRegistry
)

# Configure behavior per definition
config = ApplyConfig(
    max_atol=1e-3,           # Absolute tolerance for correctness
    max_rtol=1e-3,           # Relative tolerance
    aot_ratio=0.5,           # Pre-compile top 50% of solutions
    on_miss_policy="fallback_only"  # or "use_def_best"
)

# Register config
ApplyConfigRegistry.register("rmsnorm_d4096", config)

# Enable runtime
enable_apply(trace_set)

# ... do work ...

# Disable when done
disable_apply()
```

### 4. ApplyKey and Workload Fingerprinting

The `ApplyKey` identifies which solution to use based on runtime inputs:

```python
from flashinfer_bench.apply import ApplyKey, ApplyKeyBuilder

# Default: extract variable axes from input shapes
class AxesOnlyKeyBuilder(ApplyKeyBuilder):
    def build(self, definition, args, kwargs) -> ApplyKey:
        axes = {}
        for axis_name, axis_spec in definition.axes.items():
            if isinstance(axis_spec, AxisVar):
                # Extract from corresponding input shape
                axes[axis_name] = self._extract_axis_value(axis_name, args)
        return ApplyKey(axes=tuple(sorted(axes.items())))

# Example key for batch=32, hidden=4096
key = ApplyKey(axes=(("batch", 32), ("hidden", 4096)))
```

#### Custom Key Builders

For complex workloads, create custom key builders:

```python
from flashinfer_bench.apply import ApplyKeyBuilder, ApplyKeyFactory

class GQAKeyBuilder(ApplyKeyBuilder):
    """Custom key builder for GQA attention"""

    def build(self, definition, args, kwargs) -> ApplyKey:
        q, k, v = args[:3]
        batch, seq_q, num_heads, head_dim = q.shape
        _, seq_kv, num_kv_heads, _ = k.shape

        return ApplyKey(
            axes=(
                ("batch", batch),
                ("seq_q", seq_q),
                ("seq_kv", seq_kv),
                ("num_heads", num_heads),
                ("num_kv_heads", num_kv_heads),
            ),
            feats=(
                ("head_dim", head_dim),
                ("dtype", str(q.dtype)),
            )
        )

# Register for GQA definitions
ApplyKeyFactory.register("gqa_paged_prefill", GQAKeyBuilder())
```

### 5. ApplyTable: Offline Solution Selection

The `ApplyTable` maps keys to optimal solutions:

```python
from flashinfer_bench.apply import ApplyTable

# Build table from traces
table = ApplyTable.from_traces(trace_set)

# Lookup best solution for a key
key = ApplyKey(axes=(("batch", 32),))
solution_name = table.lookup("rmsnorm_d4096", key)
# Returns: "rmsnorm_triton_warp_v1"

# Get default best for any key
default_best = table.def_best["rmsnorm_d4096"]
# Returns: "rmsnorm_cuda_v1" (overall fastest)
```

### 6. Calling Conventions

Solutions can use two calling conventions:

```python
# Value-returning: solution returns outputs
def run(x, weight, eps):
    out = compute(x, weight, eps)
    return out

# Destination-passing: outputs passed as arguments
def run(x, weight, eps, out):
    compute_into(x, weight, eps, out)
    # Returns None, output written to 'out'
```

The runtime auto-detects based on argument count:
- `len(args) == num_inputs` → value-returning
- `len(args) == num_inputs + num_outputs` → destination-passing

## Running the Examples

```bash
# Basic apply usage
python basic_apply.py

# Enable runtime with TraceSet
python enable_runtime.py --trace-set path/to/traces

# Custom routing
python custom_routing.py

# Measure apply overhead
python benchmark_apply.py --iterations 1000
```

## Expected Output

```
Apply Mechanism Demo
====================

Without apply (reference):
  rmsnorm: 0.045 ms

With apply (Triton solution):
  rmsnorm: 0.023 ms (1.96x speedup)

Apply overhead:
  Key extraction: 0.002 ms
  Table lookup: 0.001 ms
  Total overhead: <1% of kernel time

Dispatch path:
  Definition: rmsnorm_d4096
  ApplyKey: (batch=32, hidden=4096)
  Selected: rmsnorm_triton_warp_v1
  Builder: TritonBuilder
  Cache: HIT (pre-compiled)
```
