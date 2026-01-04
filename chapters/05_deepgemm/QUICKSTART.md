# Chapter 05: DeepGEMM - Quick Start Guide

## Prerequisites

- CUDA Toolkit 12.0+ (for FP8 support)
- CMake 3.18+
- Python 3.8+ with PyTorch
- NVIDIA GPU with compute capability 8.0+ (recommended: 9.0+ for FP8)

## Building All Examples

```bash
cd /home/uvxiao/cuda-kernel-tutorial/chapters/05_deepgemm
mkdir build && cd build
cmake ..
make -j
```

This will build all examples and exercises.

## Running Examples

### 1. FP8 Basics

```bash
# From build directory
./examples/01_fp8_basics/fp8_types
./examples/01_fp8_basics/fp8_conversion
```

**What to expect:**
- FP8 format exploration (E4M3, E5M2)
- Conversion accuracy analysis
- Performance benchmarks

### 2. Quantization

```bash
./examples/02_quantization/quantize
./examples/02_quantization/fine_grained_scaling
```

**What to expect:**
- Per-tensor vs per-channel quantization comparison
- Fine-grained scaling demonstration
- Error analysis with outliers

### 3. Grouped GEMM

```bash
./examples/03_grouped_gemm/grouped_gemm
./examples/03_grouped_gemm/variable_sizes
```

**What to expect:**
- Basic grouped GEMM implementation
- Load balancing strategies
- Speedup vs padded approach

### 4. DeepGEMM Usage (Python)

```bash
# Note: Requires DeepGEMM installation
cd ../examples/04_deepgemm_usage

python dense_example.py --size 2048 --compare
python moe_example.py --experts 8 --hidden 2048
```

**What to expect:**
- FP8 vs BF16 performance comparison
- MoE grouped GEMM demonstration

## Running Benchmarks

```bash
cd ../benchmarks

# FP8 benchmark
python bench_fp8.py --sizes 1024,2048,4096

# Grouped GEMM benchmark
python bench_grouped.py --experts 8,16 --hidden 2048
```

## Working on Exercises

### Exercise 1: Simple Grouped GEMM

```bash
cd ../exercises/01_simple_grouped_gemm

# Test reference solution
python test.py

# Start your implementation
cp starter.cu student_solution.cu
# Edit student_solution.cu
# Test your solution
python test.py
```

## Expected Performance

### On NVIDIA H100

| Operation | Size | Precision | Expected Throughput |
|-----------|------|-----------|---------------------|
| Dense GEMM | 4096^3 | FP8 | 800+ TFLOPS |
| Dense GEMM | 4096^3 | BF16 | 400+ TFLOPS |
| Grouped GEMM (8 experts) | 2048^3 avg | FP8 | 700+ TFLOPS |

### On NVIDIA A100

| Operation | Size | Precision | Expected Throughput |
|-----------|------|-----------|---------------------|
| Dense GEMM | 4096^3 | BF16 | 300+ TFLOPS |
| Dense GEMM | 4096^3 | FP16 | 300+ TFLOPS |

Note: A100 does not have FP8 Tensor Cores.

## Troubleshooting

### Compilation Errors

**Issue:** `error: namespace "nvcuda::wmma" has no member "mma_sync"`

**Solution:** Update CUDA Toolkit to 12.0+

**Issue:** `unsupported GPU architecture 'compute_90'`

**Solution:** Remove SM 90 from CMakeLists.txt if you don't have H100:
```cmake
set(CMAKE_CUDA_ARCHITECTURES 80 86 89)  # Remove 90
```

### Runtime Errors

**Issue:** CUDA out of memory

**Solution:** Reduce matrix sizes in benchmarks:
```bash
python bench_fp8.py --sizes 1024,2048  # Skip 4096
```

**Issue:** Numerical errors / NaN values

**Solution:** Check FP8 range limits - values should be within [-448, 448] for E4M3

### Performance Issues

**Issue:** Low TFLOPS (< 50% of expected)

**Checklist:**
- [ ] GPU not throttling (check `nvidia-smi`)
- [ ] Using correct CUDA architecture in CMake
- [ ] Warmup iterations executed before timing
- [ ] cudaDeviceSynchronize() called before stopping timer

## Learning Path

### Beginner (New to FP8/MoE)
1. Read main README.md
2. Run FP8 basics examples (01_fp8_basics)
3. Study quantization techniques (02_quantization)
4. Try exercise 1

### Intermediate (Familiar with GEMM optimization)
1. Study grouped GEMM examples (03_grouped_gemm)
2. Compare with your own implementations
3. Benchmark on your hardware
4. Optimize exercise 1 for performance

### Advanced (Ready for production)
1. Study DeepGEMM source code
2. Implement custom kernels for your workload
3. Integrate with transformer models
4. Profile and optimize end-to-end

## Additional Resources

### Documentation
- [NVIDIA FP8 Guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)
- [DeepSeek-V3 Paper](https://arxiv.org/abs/2412.19437)

### Community
- Open issues for bugs/questions
- Share your benchmark results
- Contribute optimizations

## Next Steps

After mastering this chapter:
- Explore advanced MoE routing strategies
- Implement custom quantization schemes (GPTQ, AWQ)
- Study sparsity patterns in MoE
- Build end-to-end LLM inference system

Happy learning!
