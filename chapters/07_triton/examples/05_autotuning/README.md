# Example 05 - Autotuning

Comprehensive guide to using Triton's autotuning capabilities for automatic performance optimization.

## Overview

Autotuning allows Triton to automatically find the best kernel configuration for your specific hardware and input sizes. Instead of manually tuning block sizes, warp counts, and pipeline stages, the `@triton.autotune` decorator benchmarks multiple configurations and caches the fastest one.

## Basic Usage

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    ...
```

## How It Works

1. **First Call**: Triton benchmarks all configurations
   - Runs each config multiple times
   - Measures execution time
   - Selects fastest configuration

2. **Caching**: Best config is cached based on `key`
   - Cache key: tuple of key parameter values
   - Example: `key=['M', 'N']` → cache[(M, N)] = best_config
   - Cache persists across runs (disk-based)

3. **Subsequent Calls**: Use cached configuration
   - Instant lookup (no benchmarking)
   - Much faster than first call

## Configuration Parameters

### Kernel Parameters (dict)

```python
triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32})
```

Any compile-time constant parameter can be autotuned.

### num_warps

Number of warps (groups of 32 threads) per thread block.

- **More warps (8)**: Better occupancy, more parallelism
- **Fewer warps (2)**: Less overhead, faster synchronization
- **Common values**: 2, 4, 8

**Rule of thumb:**
- Small blocks (≤64): 2-4 warps
- Medium blocks (128-256): 4-8 warps
- Large blocks (≥512): 8 warps

### num_stages

Number of pipeline stages for software pipelining.

- **More stages (4-5)**: Better latency hiding, higher throughput
- **Fewer stages (2-3)**: Lower register usage, better occupancy

**Trade-off:** Memory ops overlap with compute vs resource usage

**Rule of thumb:**
- Memory-bound: 4-5 stages
- Compute-bound: 2-3 stages

### num_ctas (Advanced)

Number of cooperative thread array (CTA) blocks.

Usually left at default (1). Used for advanced multi-CTA kernels.

## Key Selection

The `key` parameter determines when to re-autotune:

```python
key=['M', 'N', 'K']  # Autotune per matrix shape
```

**Include in key:**
- Parameters affecting performance (sizes, dimensions)
- Values that vary across calls

**Exclude from key:**
- Pointers (constant effect on performance)
- Strides (usually constant effect)
- Small constants

**Example:**
```python
# Good
key=['n_elements']  # Vector op
key=['M', 'N', 'K']  # Matmul

# Bad
key=['x_ptr', 'y_ptr']  # Pointers don't affect perf
key=['n_elements', 'stride']  # Stride usually constant
```

## Example Configurations

### Vector Operations

```python
configs=[
    triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
]
```

### Matrix Operations

```python
configs=[
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=5),
]
```

### Reduction Operations

```python
configs=[
    triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
]
```

## Debugging Autotuning

### View Autotuning Process

```bash
TRITON_PRINT_AUTOTUNING=1 python your_script.py
```

Shows:
- Configurations being tested
- Benchmark times
- Selected configuration

### Cache Location

```
~/.triton/cache/
```

### Clear Cache

```bash
rm -rf ~/.triton/cache/
```

Forces re-autotuning on next run.

## Performance Impact

Typical improvements from autotuning:

| Scenario | Fixed Config | Autotuned | Improvement |
|----------|--------------|-----------|-------------|
| Vector add (small) | 0.035 ms | 0.028 ms | 25% |
| Matmul (medium) | 1.2 ms | 0.98 ms | 22% |
| Reduction (large) | 0.15 ms | 0.12 ms | 25% |

**Expected speedup:** 10-50% over reasonable fixed configuration

## Best Practices

### 1. Choose Configurations Wisely

**Good:**
```python
# Cover range of sizes
configs=[
    triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
]
```

**Bad:**
```python
# Too many similar configs
configs=[
    triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    triton.Config({'BLOCK_SIZE': 130}, num_warps=4),  # Too similar!
    triton.Config({'BLOCK_SIZE': 132}, num_warps=4),
]
```

### 2. Balance Config Count

- **Too few (1-2)**: May miss optimal config
- **Just right (3-8)**: Good coverage, reasonable overhead
- **Too many (>15)**: First call very slow, diminishing returns

### 3. Incremental Development

```python
# Start without autotuning
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr):
    ...

# Add autotuning once working
@triton.autotune(configs=[...], key=[...])
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr):
    ...
```

### 4. Test Multiple Input Sizes

Autotuning may select different configs for different sizes:

```python
# Small inputs: BLOCK_SIZE=128 might be best
# Large inputs: BLOCK_SIZE=1024 might be best
```

## Advanced Features

### Reset and Restore

```python
@triton.autotune(
    configs=[...],
    key=['M', 'N'],
    reset_to_zero=['output_ptr'],  # Zero before each benchmark
    restore_value=['output_ptr'],   # Restore after benchmark
)
```

Useful when kernel modifies outputs and you want clean benchmarks.

### Warmup Iterations

```python
@triton.autotune(
    configs=[...],
    key=['n'],
    warmup=10,  # Number of warmup iterations
    rep=100,    # Number of benchmark iterations
)
```

More iterations = more accurate benchmarking but slower first call.

## Common Pitfalls

### 1. Autotuning on Every Call

```python
# BAD: Different key values every call
kernel[grid](..., M=np.random.randint(100, 1000))
```

This re-autotunes constantly! Use consistent sizes or round to bins.

### 2. Too Many Key Parameters

```python
# BAD: Too specific, poor cache reuse
key=['M', 'N', 'K', 'batch', 'channels', 'height', 'width']
```

Keep keys minimal for better cache hit rate.

### 3. Forgetting Warmup

```python
# BAD: Benchmark immediately
result = kernel[grid](...)  # First call is slow!
```

Always warmup before benchmarking:
```python
# Good
for _ in range(10):
    kernel[grid](...)  # Warmup + autotune
torch.cuda.synchronize()

# Now benchmark
start = time.time()
for _ in range(100):
    kernel[grid](...)
torch.cuda.synchronize()
time = (time.time() - start) / 100
```

## Comparison with Manual Tuning

| Aspect | Manual | Autotuning |
|--------|--------|------------|
| **Development Time** | Hours-days | Minutes |
| **Portability** | GPU-specific | Adapts to hardware |
| **Maintenance** | Manual retuning | Automatic |
| **Peak Performance** | Potentially higher | 90-100% of manual |
| **First Call** | Fast | Slow (benchmarking) |

## When to Use Autotuning

**Use autotuning when:**
- Performance is critical
- Kernel runs many times (amortize first call cost)
- Input sizes vary
- Deploying to different GPUs

**Skip autotuning when:**
- Quick prototyping
- Single-use kernels
- Minimal performance requirements
- First call latency critical

## Key Takeaways

1. Autotuning finds best config automatically
2. First call is slow, subsequent calls fast
3. Cache based on `key` parameters
4. Typical speedup: 10-50%
5. Balance config count vs first-call overhead
6. Essential for production kernels

## Next Steps

- Review matmul_autotuned.py for comprehensive example
- Experiment with different config sets
- Profile your kernels to identify bottlenecks
- Apply autotuning to your custom kernels

## References

- [Triton Autotuning Tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [Triton API Reference](https://triton-lang.org/main/python-api/triton.html#triton.autotune)
