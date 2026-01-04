# Chapter 07 - Quick Start Guide

Get started with Triton kernel development in 15 minutes!

## Prerequisites

```bash
# Install Triton and PyTorch
pip install triton torch

# Verify installation
python -c "import triton; print(f'Triton version: {triton.__version__}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 5-Minute Tour

### 1. Your First Triton Kernel (2 minutes)

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Which block am I?
    pid = tl.program_id(0)

    # Calculate my elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # Load, compute, store
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

# Use the kernel
def add(x, y):
    output = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
    return output

# Test it
x = torch.randn(10000, device='cuda')
y = torch.randn(10000, device='cuda')
result = add(x, y)
assert torch.allclose(result, x + y)
print("Success! Your first Triton kernel works!")
```

**Save this as `first_kernel.py` and run:**
```bash
python first_kernel.py
```

### 2. Key Concepts (3 minutes)

**Block-Based Model:**
- CUDA: Each thread processes 1 element
- Triton: Each program processes BLOCK_SIZE elements

**Essential Operations:**
```python
pid = tl.program_id(0)              # Which block am I?
offsets = tl.arange(0, BLOCK_SIZE)  # Create indices
mask = offsets < n                   # Boundary check
x = tl.load(ptr + offsets, mask=mask)  # Load data
tl.store(ptr + offsets, x, mask=mask)   # Store data
```

**Compile-Time Constants:**
```python
BLOCK_SIZE: tl.constexpr  # Optimized at compile time
```

## 10-Minute Deep Dive

### Run the Examples

```bash
# Example 1: Vector addition
cd examples/01_vector_add
python vector_add.py

# Example 2: Softmax
cd ../02_softmax
python softmax.py

# Example 3: Matrix multiplication
cd ../03_matmul
python matmul_blocked.py
```

### Try a Puzzle

```bash
cd puzzles
# Open puzzle_01_add.py and fill in the blanks
python puzzle_01_add.py
```

### Complete an Exercise

```bash
cd exercises/01_relu_fusion
# Read problem.md
# Edit starter.py
python test.py
```

## Learning Path

### Week 1: Basics
- [ ] Read main README.md
- [ ] Run examples/01_vector_add
- [ ] Run examples/02_softmax
- [ ] Complete puzzle_01

### Week 2: Matrix Operations
- [ ] Run all matmul examples (03)
- [ ] Complete exercise 01 (relu_fusion)
- [ ] Read about kernel fusion (04)

### Week 3: Advanced
- [ ] Study autotuning (05)
- [ ] Complete exercise 02 (gelu)
- [ ] Implement custom kernel for your use case

## Common Tasks

### Task 1: Vector Operation

```python
@triton.jit
def my_kernel(input_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(input_ptr + offsets, mask=mask)
    y = # Your operation here
    tl.store(output_ptr + offsets, y, mask=mask)
```

### Task 2: Reduction

```python
@triton.jit
def reduce_kernel(input_ptr, output_ptr, row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    row_ptr = input_ptr + row * row_stride

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    x = tl.load(row_ptr + offsets, mask=mask, other=0.0)

    result = tl.sum(x, axis=0)  # or tl.max, tl.min
    tl.store(output_ptr + row, result)
```

### Task 3: Matrix Operation

```python
@triton.jit
def matrix_kernel(a_ptr, b_ptr, c_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    a_ptrs = a_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    a = tl.load(a_ptrs, mask=mask)
    # Process a...
    tl.store(c_ptrs, c, mask=mask)
```

## Quick Reference

### Memory Operations
```python
tl.load(ptr + offsets, mask=mask, other=0.0)
tl.store(ptr + offsets, value, mask=mask)
```

### Arithmetic
```python
x + y, x * y, x / y, x ** y
tl.maximum(x, y), tl.minimum(x, y)
tl.exp(x), tl.log(x), tl.sqrt(x)
tl.sin(x), tl.cos(x)
```

### Reductions
```python
tl.sum(x, axis=0)
tl.max(x, axis=0)
tl.min(x, axis=0)
```

### Matrix Operations
```python
c = tl.dot(a, b)  # Matrix multiply
tl.trans(a)        # Transpose
```

## Troubleshooting

### "CUDA not available"
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

### "Triton not installed"
```bash
pip install --upgrade triton
```

### "Kernel launch error"
- Check grid size calculation
- Verify all pointers are valid
- Ensure BLOCK_SIZE is power of 2

### "Wrong results"
- Check mask logic
- Verify pointer arithmetic
- Test with smaller inputs first

## Next Steps

1. **Read the Docs**: Start with main [README.md](README.md)
2. **Run Examples**: Work through [examples/](examples/) in order
3. **Practice**: Complete [puzzles/](puzzles/) and [exercises/](exercises/)
4. **Build**: Create your own kernels!

## Resources

- [Triton Documentation](https://triton-lang.org/)
- [Triton GitHub](https://github.com/openai/triton)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [GPU MODE Lectures](https://github.com/gpu-mode/lectures)

## Help and Support

- Check example READMEs for detailed explanations
- Compare your code with solutions
- Review main README.md for concepts
- Study working examples before implementing

## Success Checklist

After this quick start, you should be able to:
- [x] Write a basic Triton kernel
- [x] Understand block-based programming model
- [x] Use tl.load and tl.store with masks
- [x] Launch kernels from Python
- [x] Test for correctness

Ready to dive deeper? Start with [examples/01_vector_add/](examples/01_vector_add/)!
