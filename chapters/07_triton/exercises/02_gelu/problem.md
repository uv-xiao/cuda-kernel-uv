# Exercise 02: GELU Activation Function

## Objective

Implement the GELU (Gaussian Error Linear Unit) activation function in Triton.

## Background

GELU is a smooth activation function widely used in transformers (BERT, GPT, etc.):

```
GELU(x) = x * Φ(x)
```

where Φ(x) is the cumulative distribution function of the standard normal distribution.

### Approximation

The exact formula involves the error function (erf), which is expensive to compute. We use the tanh approximation:

```
GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

This approximation is accurate and commonly used in practice.

## Task

Implement GELU activation as a Triton kernel.

### Requirements

1. Implement the tanh approximation formula
2. Handle arbitrary tensor sizes
3. Match PyTorch's GELU output (within numerical precision)
4. Achieve competitive performance

### Formula Breakdown

```python
# Constants
sqrt_2_over_pi = 0.7978845608  # sqrt(2/pi)
coeff = 0.044715

# Computation
x_cubed = x * x * x
inner = sqrt_2_over_pi * (x + coeff * x_cubed)
tanh_inner = tanh(inner)
gelu = 0.5 * x * (1 + tanh_inner)
```

### Hints

1. Use `tl.libdevice` for tanh: `tl.libdevice.tanh(x)`
2. Or approximate tanh yourself (bonus challenge)
3. Apply element-wise to each block
4. Remember masking for boundary conditions

### Test Cases

Your implementation should:
- Match PyTorch GELU to <1e-4 relative error
- Handle edge cases: very large/small values
- Work with various tensor shapes
- Be faster than unfused PyTorch operations

## Bonus Challenges

1. **Fast Tanh Approximation**: Implement tanh without libdevice
   ```python
   tanh(x) ≈ x * (27 + x²) / (27 + 9*x²)  # Pade approximation
   ```

2. **Fused GELU**: Combine with another operation (e.g., matmul + GELU)

3. **Backward Pass**: Implement GELU gradient
   ```python
   GELU'(x) = Φ(x) + x * φ(x)
   ```

4. **Exact GELU**: Implement using erf function
   ```python
   GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
   ```

## Learning Goals

- Implement complex mathematical functions
- Use Triton's math library (`tl.libdevice`)
- Optimize for numerical accuracy
- Benchmark against PyTorch

## Resources

- [GELU Paper](https://arxiv.org/abs/1606.08415)
- [PyTorch GELU Source](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/Activation.cu)
- [Triton Math Functions](https://triton-lang.org/main/python-api/triton.language.html#module-triton.language.math)

## Estimated Time

20-30 minutes

## Success Criteria

1. **Correctness**: Max relative error < 1e-4 vs PyTorch
2. **Performance**: Within 20% of PyTorch GELU speed
3. **Code Quality**: Clean, well-commented implementation
