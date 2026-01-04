# Getting Started with Chapter 09: Sparse Attention Kernels

## Quick Start

This chapter teaches you to implement efficient attention mechanisms, from dense O(L²) to sparse O(Lk) complexity.

### Prerequisites

- CUDA basics (from previous chapters)
- Understanding of transformers and attention mechanism
- PyTorch installed for testing

### Recommended Learning Path

```
1. Read main README.md
   ↓
2. Study examples/01_attention_basics/
   - Understand naive O(L²) implementation
   - Run Python reference: python naive_attention.py
   ↓
3. Study examples/02_flash_attention_minimal/
   - Learn tiling and online softmax
   - Build and run: cd build && cmake .. && make && ./flash_fwd
   ↓
4. Try Exercise 01: Implement mini FlashAttention
   - Work through exercises/01_mini_flash/
   ↓
5. Study examples/04_sparse_patterns/
   - Local and strided attention
   - Build: cd build && cmake .. && make
   ↓
6. Study examples/05_deepseek_sparse/
   - Lightning indexer concept
   - Run: python sparse_attention.py
   ↓
7. Try Exercise 02: Custom sparse pattern
   - Design pattern for your use case
   ↓
8. Run benchmarks/
   - Compare all implementations
   - Analyze performance vs memory trade-offs
```

## Chapter Structure

### Examples (Progressive Complexity)

1. **01_attention_basics/** - Baseline O(L²) implementation
   - `naive_attention.cu` - CUDA implementation
   - `naive_attention.py` - PyTorch reference

2. **02_flash_attention_minimal/** - Educational FlashAttention
   - ~100 lines, focuses on clarity
   - Tiling + online softmax

3. **03_flash_attention_v2/** - Optimized FlashAttention-2
   - Warp-level primitives
   - Better parallelization

4. **04_sparse_patterns/** - Various sparsity patterns
   - Local (sliding window)
   - Strided (dilated)

5. **05_deepseek_sparse/** - DeepSeek DSA
   - Lightning indexer
   - Dynamic token selection
   - Full sparse attention pipeline

### Benchmarks

- `bench_attention.py` - Performance comparison
- `memory_analysis.py` - Memory usage analysis

### Exercises

1. **01_mini_flash/** - Implement simplified FlashAttention
2. **02_sparse_mask/** - Design custom sparse pattern

## Building Examples

### CUDA Examples

```bash
cd examples/02_flash_attention_minimal
mkdir build && cd build
cmake ..
make
./flash_fwd
```

Repeat for other CUDA examples (03, 04).

### Python Examples

```bash
cd examples/01_attention_basics
python naive_attention.py --all

cd ../05_deepseek_sparse
python sparse_attention.py
```

## Key Concepts Covered

### 1. Complexity Reduction

```
Dense Attention:     O(L² · d)
FlashAttention:      O(L² · d) but memory-efficient
Sparse Attention:    O(L · k · d) where k << L
```

### 2. Memory Efficiency

```
Dense: Stores L×L attention matrix
Flash: Tiling + recomputation (no L×L storage)
Sparse: Only k entries per query
```

### 3. Algorithms

- **Online Softmax**: Incremental max and normalization
- **Tiling**: Block-wise computation in SRAM
- **Token Selection**: Lightning indexer for O(1)-ish selection

## Performance Expectations

On NVIDIA A100, batch=4, heads=8, d=64:

| Implementation | L=2048 Time | L=8192 Time | Memory (L=8192) |
|---------------|-------------|-------------|-----------------|
| Naive | 14.1 ms | 380 ms | 1024 MB |
| FlashAttention | 2.1 ms | 28 ms | 64 MB |
| Flash-v2 | 1.5 ms | 19 ms | 64 MB |
| Sparse (k=512) | 0.5 ms | 2.1 ms | 16 MB |

## Common Issues and Solutions

### Issue 1: Out of Shared Memory

```
Error: too much shared memory required
```

**Solution**: Reduce block sizes (Br, Bc) in kernel definition.

### Issue 2: Wrong Results

```
Output differs from PyTorch reference
```

**Solution**: Check online softmax logic, especially max rescaling.

### Issue 3: Slow Performance

```
Kernel is slower than expected
```

**Solution**:
- Profile with `nsight compute`
- Check occupancy
- Ensure warp-level primitives are used

## Testing Your Implementation

### Correctness Test

```python
import torch
import torch.nn.functional as F

Q = torch.randn(2, 128, 64, device='cuda')
K = torch.randn(2, 128, 64, device='cuda')
V = torch.randn(2, 128, 64, device='cuda')

# Your implementation
output_yours = your_attention(Q, K, V)

# PyTorch reference
Q_m = Q.unsqueeze(1)
K_m = K.unsqueeze(1)
V_m = V.unsqueeze(1)
output_ref = F.scaled_dot_product_attention(Q_m, K_m, V_m).squeeze(1)

# Check
max_diff = (output_yours - output_ref).abs().max()
print(f"Max difference: {max_diff:.2e}")
assert max_diff < 1e-5, "Failed correctness test!"
```

### Performance Test

```bash
# Run benchmarks
cd benchmarks
python bench_attention.py

# Analyze memory
python memory_analysis.py
```

## Resources

### Papers

- [FlashAttention](https://arxiv.org/abs/2205.14135) - Dao et al., 2022
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) - Dao, 2023
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437) - DeepSeek-AI, 2024
- [Sparse Transformers](https://arxiv.org/abs/1904.10509) - Child et al., 2019

### Code References

- [Official FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal)
- [xFormers](https://github.com/facebookresearch/xformers)

## Next Steps

After completing this chapter:

1. **Optimize your kernels**: Use tensor cores, vectorization
2. **Combine techniques**: Flash + Sparse for maximum efficiency
3. **Apply to real models**: Integrate into your transformer
4. **Explore advanced topics**: Multi-GPU attention, quantization

## Questions?

Check the main README.md for detailed explanations of each concept.

Happy coding!
