# Exercise 01: Fused ReLU Matrix Multiplication

## Objective

Implement a fused kernel that computes matrix multiplication followed by ReLU activation:

```
output = ReLU(A @ B)
      = max(0, A @ B)
```

## Background

In neural networks, matrix multiplications are often immediately followed by activation functions like ReLU. By fusing these operations, we can:

1. **Eliminate intermediate storage**: No need to write matmul result to memory
2. **Reduce memory bandwidth**: One fewer read/write pass
3. **Improve performance**: 1.5-2x speedup typical

## Task

Modify the blocked matrix multiplication kernel to apply ReLU before storing results.

### Starting Point

See `starter.py` for the basic matmul kernel structure.

### Requirements

1. Compute `C = A @ B` using blocked tiling
2. Apply ReLU: `C[i,j] = max(0, C[i,j])`
3. Store the result
4. All in one kernel (no intermediate storage)

### Hints

1. ReLU can be applied element-wise to the accumulator before storing
2. Use `tl.maximum(accumulator, 0)` or equivalent
3. You can apply ReLU after all K blocks are accumulated
4. Don't forget to convert dtype before applying ReLU if needed

### Test Cases

Your implementation should pass:
- Correctness test: Match PyTorch's `relu(A @ B)`
- Performance test: Faster than separate matmul + relu
- Edge cases: All negative, all positive, mixed values

### Success Criteria

1. **Correctness**: Output matches PyTorch baseline
2. **Performance**: At least 1.3x faster than unfused version
3. **Code Quality**: Clean, well-commented implementation

## Bonus Challenges

1. **Parameterized Activation**: Support different activations (ReLU, GELU, SiLU) via compile-time parameter
2. **Bias Addition**: Extend to `ReLU(A @ B + bias)`
3. **Autotuning**: Add `@triton.autotune` decorator
4. **Backward Pass**: Implement gradient computation

## Learning Goals

- Understand kernel fusion benefits
- Practice modifying existing kernels
- Learn to apply element-wise operations efficiently
- Measure fusion performance gains

## Resources

- `examples/03_matmul/matmul_blocked.py` - Base matmul implementation
- `examples/04_fused_ops/fused_add_mul.py` - Fusion example
- Solution available in `solution.py` (try first!)

## Estimated Time

30-45 minutes
