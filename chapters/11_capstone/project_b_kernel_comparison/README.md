# Project B: Kernel Implementation Comparison

Implement the same GEMM kernel across four different frameworks and conduct a comprehensive comparative analysis. This project emphasizes cross-framework proficiency and analytical thinking.

## Project Overview

**Goal**: Implement matrix multiplication (GEMM) in CUDA C++, Triton, TileLang, and CUTLASS, then analyze the tradeoffs between frameworks.

**Duration**: 1-2 weeks

**Difficulty**: Intermediate to Advanced

## Learning Objectives

By completing this project, you will:
- Gain hands-on experience with multiple GPU programming frameworks
- Understand the tradeoffs between low-level control and high-level abstractions
- Develop critical analysis skills for technology evaluation
- Build a mental model for choosing the right tool for different scenarios

## Problem Statement

Matrix multiplication is the fundamental operation in deep learning. Different frameworks offer different approaches:

1. **CUDA C++**: Maximum control, maximum complexity
2. **Triton**: Python-based, automatic optimization
3. **TileLang**: Abstraction for tiled computations
4. **CUTLASS**: NVIDIA's template library for GEMM

Your task is to implement the same GEMM operation in all four frameworks and analyze:
- Development time and complexity
- Performance across different matrix sizes
- Code maintainability and readability
- Debugging difficulty
- Portability and integration

## GEMM Specification

Implement the following operation:

```
C = alpha * A @ B + beta * C
```

Where:
- A: (M, K) matrix
- B: (K, N) matrix
- C: (M, N) matrix
- alpha, beta: scalars

**Requirements**:
- Support FP16 and FP32 data types
- Handle both row-major and column-major layouts
- Optimize for matrix sizes common in ML:
  - Small: 128x128x128
  - Medium: 1024x1024x1024
  - Large: 4096x4096x4096
  - Rectangular: 8192x512x2048

## Required Implementations

### 1. CUDA C++ Implementation (25% of grade)

Implement a tiled GEMM kernel in CUDA C++:

**Features to Include**:
- Shared memory tiling
- Register blocking
- Thread coarsening
- Avoid bank conflicts
- Handle arbitrary matrix sizes

**Files**:
- `gemm_cuda.cu`: Kernel implementation
- `gemm_cuda.h`: Header file
- `test_cuda.cpp`: Correctness tests

**Target Performance**:
- Achieve >60% of cuBLAS throughput for M=N=K=4096

### 2. Triton Implementation (25% of grade)

Implement GEMM using Triton:

**Features to Include**:
- Proper tiling with `tl.dot`
- Block pointer syntax
- Auto-tuning configuration
- Kernel compilation optimization

**Files**:
- `gemm_triton.py`: Kernel implementation
- `test_triton.py`: Correctness tests

**Target Performance**:
- Achieve >70% of cuBLAS throughput for M=N=K=4096

### 3. TileLang Implementation (25% of grade)

Implement GEMM using TileLang:

**Features to Include**:
- Layout specification
- Tiled computation
- Proper memory hierarchy usage

**Files**:
- `gemm_tilelang.py`: Kernel implementation
- `test_tilelang.py`: Correctness tests

**Target Performance**:
- Achieve >65% of cuBLAS throughput for M=N=K=4096

### 4. CUTLASS Implementation (15% of grade)

Implement GEMM using CUTLASS templates:

**Features to Include**:
- Template instantiation
- Tile size configuration
- Instruction selection (Tensor Cores if available)

**Files**:
- `gemm_cutlass.cu`: Implementation
- `test_cutlass.cpp`: Correctness tests

**Target Performance**:
- Achieve >85% of cuBLAS throughput for M=N=K=4096

## Evaluation Criteria

### 1. Correctness (20%)

All implementations must:
- Pass correctness tests against cuBLAS
- Handle edge cases (size 0, size 1, non-square matrices)
- Support both FP16 and FP32
- Numerical error <1e-3 for FP32, <1e-2 for FP16

### 2. Performance (30%)

Each implementation should:
- Achieve target performance (see above)
- Scale well across different matrix sizes
- Utilize GPU resources efficiently

**Performance Grading**:
- Excellent: All implementations >80% of cuBLAS average
- Good: All implementations >60% of cuBLAS average
- Acceptable: All implementations >40% of cuBLAS average

### 3. Code Quality (15%)

- Clean, well-documented code
- Proper error handling
- Consistent style within each framework
- Comprehensive tests

### 4. Analysis Report (35%)

Write a comprehensive technical report (4-6 pages) covering:

#### Development Experience
- Time spent on each implementation
- Difficulty level (subjective)
- Learning curve
- Available documentation quality
- Debugging experience

#### Performance Analysis
- Benchmark results with visualizations
- Profiling insights for each implementation
- Comparison with cuBLAS baseline
- Analysis of performance characteristics

#### Framework Comparison
- Ease of use (1-10 scale)
- Performance potential
- Code complexity
- Maintainability
- Integration with existing codebases

#### Recommendations
- When to use each framework
- Tradeoffs summary
- Future trends
- Personal preferences with justification

## Deliverables

### 1. Code Implementations

```
project_b_kernel_comparison/
├── cuda/
│   ├── gemm_cuda.cu
│   ├── gemm_cuda.h
│   ├── test_cuda.cpp
│   └── Makefile
├── triton/
│   ├── gemm_triton.py
│   ├── test_triton.py
│   └── requirements.txt
├── tilelang/
│   ├── gemm_tilelang.py
│   ├── test_tilelang.py
│   └── requirements.txt
├── cutlass/
│   ├── gemm_cutlass.cu
│   ├── test_cutlass.cpp
│   └── Makefile
├── benchmark.py (unified benchmarking)
└── README.md
```

### 2. Test Results

For each implementation:
- Correctness test output
- Performance benchmarks
- Profiling results (nsys, ncu)

### 3. Technical Report

**Required Sections**:
1. Introduction
2. Implementation Details (one subsection per framework)
3. Performance Results
4. Comparative Analysis
5. Recommendations
6. Conclusion
7. Appendix (code snippets, profiling screenshots)

**Format**: PDF, 4-6 pages (excluding appendix)

### 4. Presentation

10-15 minute presentation covering:
- Overview of each implementation
- Key performance results
- Framework comparison
- Recommendations

## Benchmark Configurations

Test all implementations with these configurations:

### Matrix Sizes
```python
configs = [
    # Small
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),

    # Medium
    (1024, 1024, 1024),
    (2048, 2048, 2048),

    # Large
    (4096, 4096, 4096),
    (8192, 8192, 8192),

    # Rectangular
    (8192, 512, 2048),
    (4096, 1024, 4096),
    (2048, 4096, 1024),
]
```

### Data Types
- FP32
- FP16 (if supported by framework)

### Metrics to Collect
- Execution time (ms)
- Throughput (TFLOPS)
- Memory bandwidth utilization (%)
- Compute utilization (%)
- % of cuBLAS performance

## Expected Results

### H100 GPU

| Framework | M=N=K=4096 (TFLOPS) | % of cuBLAS | Development Time |
|-----------|---------------------|-------------|------------------|
| cuBLAS (baseline) | ~850 | 100% | N/A |
| CUDA C++ | ~510-680 | 60-80% | 8-12 hours |
| Triton | ~600-750 | 70-88% | 4-6 hours |
| TileLang | ~550-720 | 65-85% | 3-5 hours |
| CUTLASS | ~720-850 | 85-100% | 2-4 hours |

### A100 GPU

| Framework | M=N=K=4096 (TFLOPS) | % of cuBLAS | Development Time |
|-----------|---------------------|-------------|------------------|
| cuBLAS (baseline) | ~280 | 100% | N/A |
| CUDA C++ | ~168-224 | 60-80% | 8-12 hours |
| Triton | ~196-246 | 70-88% | 4-6 hours |
| TileLang | ~182-238 | 65-85% | 3-5 hours |
| CUTLASS | ~238-280 | 85-100% | 2-4 hours |

## Optimization Tips

### CUDA C++
1. Use shared memory for A and B tiles
2. Implement register blocking (e.g., 8x8 per thread)
3. Use `__ldg()` for read-only data
4. Avoid shared memory bank conflicts
5. Maximize occupancy

### Triton
1. Tune block sizes (BLOCK_M, BLOCK_N, BLOCK_K)
2. Use `tl.dot()` for matrix multiplication blocks
3. Leverage auto-tuning decorators
4. Use block pointers for efficient memory access
5. Enable persistent mode for small kernels

### TileLang
1. Define appropriate tile layout
2. Specify memory hierarchy explicitly
3. Use built-in GEMM primitives
4. Optimize data movement between memory levels

### CUTLASS
1. Choose appropriate tile sizes
2. Select instruction (Volta, Turing, Ampere, Hopper)
3. Use predefined epilogue operations
4. Leverage threadblock swizzling
5. Enable tensor core operations

## Common Pitfalls

### All Frameworks
- Not validating correctness before optimizing
- Testing only square matrices
- Ignoring warmup iterations
- Not handling non-power-of-2 sizes
- Forgetting to test both FP16 and FP32

### CUDA C++
- Shared memory bank conflicts
- Poor thread block sizing
- Not using constant memory for small data
- Inefficient boundary handling

### Triton
- Suboptimal block sizes
- Not using auto-tuning
- Incorrect pointer arithmetic
- Not leveraging compiler optimizations

### TileLang
- Incorrect layout specification
- Inefficient memory hierarchy usage
- Not using built-in primitives

### CUTLASS
- Template complexity overwhelming
- Not understanding predefined tile sizes
- Incorrect epilogue configuration

## Profiling Checklist

For each implementation, collect:

- [ ] Kernel execution time
- [ ] Memory bandwidth utilization
- [ ] SM utilization
- [ ] Tensor Core utilization (if applicable)
- [ ] L2 cache hit rate
- [ ] Occupancy
- [ ] Register usage per thread
- [ ] Shared memory usage

## Analysis Framework

Use this framework for your comparative analysis:

### Dimension 1: Ease of Use (1-10)
- Learning curve
- Documentation quality
- Error messages
- Debugging tools
- IDE support

### Dimension 2: Performance (1-10)
- Raw performance
- Optimization potential
- Consistency across sizes
- Scalability

### Dimension 3: Productivity (1-10)
- Development time
- Code conciseness
- Abstraction level
- Reusability

### Dimension 4: Ecosystem (1-10)
- Community support
- Integration with frameworks
- Available examples
- Long-term support

### Dimension 5: Flexibility (1-10)
- Customization options
- Constraint handling
- Extensibility

## Report Template

Use the provided [report_template.md](./report_template.md) as a starting point for your technical report.

## Getting Started

### Week 1: Implementations

**Day 1-2: CUDA C++**
- Review CUDA GEMM tutorials
- Implement naive version
- Add tiling and optimization
- Test and validate

**Day 3-4: Triton**
- Review Triton documentation
- Implement with auto-tuning
- Compare with CUDA version

**Day 5: TileLang**
- Study TileLang examples
- Implement GEMM
- Validate correctness

**Day 6: CUTLASS**
- Review CUTLASS examples
- Instantiate templates
- Test performance

**Day 7: Benchmarking**
- Run comprehensive benchmarks
- Collect profiling data
- Generate visualizations

### Week 2: Analysis and Report

**Day 1-3: Analysis**
- Analyze profiling results
- Identify patterns and insights
- Create comparison tables

**Day 4-6: Report Writing**
- Write implementation details
- Document performance results
- Write comparative analysis
- Draft recommendations

**Day 7: Polish**
- Refine report
- Prepare presentation
- Final testing

## Resources

### CUDA
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [How to Optimize GEMM](https://siboehm.com/articles/22/CUDA-MMM)

### Triton
- [Triton Documentation](https://triton-lang.org/)
- [Triton GEMM Tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [Triton GitHub](https://github.com/openai/triton)

### TileLang
- [TileLang Repository](https://github.com/microsoft/BitBLAS)
- TileLang examples in tutorial Chapter 8

### CUTLASS
- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)
- [CUTLASS GEMM Examples](https://github.com/NVIDIA/cutlass/tree/main/examples)
- Tutorial Chapter 4

## FAQ

**Q: Do I need to optimize all implementations equally?**
A: Focus on getting all working correctly first. Optimization depth should reflect typical usage of each framework.

**Q: Can I use libraries within frameworks?**
A: Yes! For example, use `tl.dot()` in Triton, CUTLASS templates, etc. The goal is realistic usage.

**Q: What if one framework doesn't work on my GPU?**
A: Document the limitation and explain why. Focus on frameworks that do work.

**Q: How detailed should profiling be?**
A: At minimum, collect timing and throughput. Deeper analysis (bandwidth, SM utilization) earns more points.

**Q: Can I compare with other operations besides GEMM?**
A: This project focuses on GEMM. For extra credit, you could add a second operation.

## Grading Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| CUDA Implementation | 25 | Correctness (10), Performance (10), Code Quality (5) |
| Triton Implementation | 25 | Correctness (10), Performance (10), Code Quality (5) |
| TileLang Implementation | 25 | Correctness (10), Performance (10), Code Quality (5) |
| CUTLASS Implementation | 15 | Correctness (6), Performance (6), Code Quality (3) |
| Technical Report | 35 | Analysis (15), Insights (10), Writing (10) |

Total: 125 points (scaled to 100)

## Support

If you get stuck:
1. Review the respective framework's tutorial chapter
2. Check official documentation and examples
3. Start with the simplest implementation (Triton or CUTLASS)
4. Validate correctness before optimizing
5. Compare with reference implementations

Good luck with your implementation comparison!
