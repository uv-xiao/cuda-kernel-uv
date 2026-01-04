# Chapter 02: Build and Usage Guide

## Overview

Chapter 02 contains comprehensive examples and exercises covering CUDA basics, memory hierarchy, and synchronization patterns.

## Contents

### Examples (4)
1. **01_2d_indexing** - Thread and block indexing patterns
2. **02_shared_memory** - Shared memory optimization techniques
3. **03_reduction** - Parallel reduction with progressive optimization
4. **04_histogram** - Atomic operations and privatization

### Exercises (2)
1. **01_matrix_transpose** - Implement efficient matrix transpose
2. **02_dot_product** - Parallel dot product with reduction

## Building Examples

### Quick Start

```bash
cd /home/uvxiao/cuda-kernel-tutorial/chapters/02_cuda_basics
mkdir build && cd build
cmake ..
make -j4
```

### Run All Examples

```bash
# From build directory
./examples/01_2d_indexing/indexing_2d
./examples/02_shared_memory/shared_mem
./examples/03_reduction/reduction
./examples/04_histogram/histogram
```

### Individual Example Build

```bash
cd /home/uvxiao/cuda-kernel-tutorial/chapters/02_cuda_basics
mkdir build && cd build
cmake ..
make indexing_2d    # Build specific example
```

## Running Exercises

### Exercise 01: Matrix Transpose

```bash
cd /home/uvxiao/cuda-kernel-tutorial/chapters/02_cuda_basics/exercises/01_matrix_transpose

# Compile your solution
nvcc -O3 -arch=sm_70 starter.cu -o transpose

# Run manually
./transpose

# Or use test script
python test.py
```

### Exercise 02: Dot Product

```bash
cd /home/uvxiao/cuda-kernel-tutorial/chapters/02_cuda_basics/exercises/02_dot_product

# Compile your solution
nvcc -O3 -arch=sm_70 starter.cu -o dotproduct

# Run manually
./dotproduct

# Or use test script
python test.py
```

## System Requirements

- CUDA Toolkit 11.0 or higher
- CMake 3.18 or higher
- C++14 compatible compiler
- GPU with compute capability 7.0+ (Volta or newer)

## Supported GPU Architectures

This chapter is compiled for:
- Volta (sm_70, sm_75)
- Ampere (sm_80, sm_86)
- Ada (sm_89)
- Hopper (sm_90)

## File Structure

```
02_cuda_basics/
├── README.md                    # Chapter overview and concepts
├── BUILD_GUIDE.md              # This file
├── CMakeLists.txt              # Main build configuration
├── examples/
│   ├── 01_2d_indexing/
│   │   ├── indexing_2d.cu      # Source code
│   │   ├── CMakeLists.txt      # Build config
│   │   └── README.md           # Example documentation
│   ├── 02_shared_memory/
│   │   ├── shared_mem.cu
│   │   ├── CMakeLists.txt
│   │   └── README.md
│   ├── 03_reduction/
│   │   ├── reduction.cu
│   │   ├── CMakeLists.txt
│   │   └── README.md
│   └── 04_histogram/
│       ├── histogram.cu
│       ├── CMakeLists.txt
│       └── README.md
└── exercises/
    ├── 01_matrix_transpose/
    │   ├── problem.md          # Exercise description
    │   ├── starter.cu          # Template code
    │   ├── solution.cu         # Reference solution
    │   └── test.py             # Automated testing
    └── 02_dot_product/
        ├── problem.md
        ├── starter.cu
        ├── solution.cu
        └── test.py
```

## Troubleshooting

### CMake Can't Find CUDA

```bash
# Set CUDA path manually
export CUDA_PATH=/usr/local/cuda
cmake -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda ..
```

### Compile Errors for Older GPUs

Edit `CMakeLists.txt` and adjust:
```cmake
set(CMAKE_CUDA_ARCHITECTURES "70;75;80")  # Remove newer architectures
```

### Out of Memory Errors

Reduce array sizes in examples:
- `indexing_2d.cu`: Reduce matrix size from 1024x1024
- `reduction.cu`: Reduce array size from 16M elements
- `histogram.cu`: Reduce data size from 64M elements

### Permission Denied for test.py

```bash
chmod +x exercises/*/test.py
```

## Performance Notes

### Expected Example Runtimes (RTX 3080)

- **2D Indexing**: ~1-2 ms
- **Shared Memory**: ~0.5-2 ms per kernel
- **Reduction**: ~0.1-0.8 ms per version
- **Histogram**: ~1-8 ms per version

### Expected Exercise Performance

**Matrix Transpose (4096x4096)**:
- Naive: ~100-150 GB/s
- Shared: ~300-400 GB/s
- Optimized: ~400-550 GB/s

**Dot Product (64M elements)**:
- Naive: ~5-10 ms
- Shared: ~1-2 ms
- Optimized: ~0.5-1 ms

## Learning Path

1. **Start with README.md** - Understand concepts
2. **Run examples** - See techniques in action
3. **Study example code** - Learn implementation details
4. **Attempt exercises** - Practice what you learned
5. **Compare with solutions** - Improve your approach

## Additional Resources

- Chapter README.md - Comprehensive concept explanations
- Individual example READMEs - Detailed technique breakdowns
- Exercise problem.md files - Step-by-step guidance
- CUDA Programming Guide - Official documentation

## Getting Help

If you encounter issues:
1. Check this build guide
2. Review example READMEs for detailed explanations
3. Examine solution code for reference implementations
4. Consult CUDA Programming Guide for API details

## Next Steps

After completing this chapter:
- Move to Chapter 03: Advanced Memory Patterns
- Explore warp-level primitives
- Study memory coalescing in depth
- Learn about texture and constant memory
