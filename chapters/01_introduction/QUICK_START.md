# Chapter 01: Quick Start Guide

## Building All Examples and Exercises

```bash
cd /home/uvxiao/cuda-kernel-tutorial/chapters/01_introduction
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Running Examples

```bash
# From build directory
./examples/01_hello_cuda/hello_cuda
./examples/02_vector_add/vector_add
./examples/03_device_query/device_query
```

## Working on Exercises

### Exercise 01: Vector Subtraction

```bash
cd exercises/01_vector_subtract

# Edit starter.cu to complete the exercise
vim starter.cu  # or your favorite editor

# Compile and test
nvcc -o vector_subtract starter.cu
./vector_subtract

# Or use the test script
python3 test.py

# Check the solution if needed
nvcc -o vector_subtract_sol solution.cu
./vector_subtract_sol
```

### Exercise 02: SAXPY

```bash
cd exercises/02_saxpy

# Edit starter.cu
vim starter.cu

# Compile and test
nvcc -o saxpy starter.cu
./saxpy

# Test with different parameters
./saxpy 100000 2.5  # size=100k, alpha=2.5

# Run test suite
python3 test.py
```

## File Structure

```
01_introduction/
├── README.md                          # Chapter overview
├── CMakeLists.txt                     # Build configuration
├── QUICK_START.md                     # This file
│
├── examples/
│   ├── 01_hello_cuda/
│   │   ├── hello.cu                   # Basic kernel example
│   │   ├── CMakeLists.txt
│   │   └── README.md
│   │
│   ├── 02_vector_add/
│   │   ├── vector_add.cu              # Complete CUDA workflow
│   │   ├── CMakeLists.txt
│   │   └── README.md
│   │
│   └── 03_device_query/
│       ├── device_query.cu            # Query GPU properties
│       ├── CMakeLists.txt
│       └── README.md
│
└── exercises/
    ├── 01_vector_subtract/
    │   ├── problem.md                 # Exercise description
    │   ├── starter.cu                 # Template to fill in
    │   ├── solution.cu                # Reference solution
    │   └── test.py                    # Automated testing
    │
    └── 02_saxpy/
        ├── problem.md                 # Exercise description
        ├── starter.cu                 # Template to fill in
        ├── solution.cu                # Reference solution
        └── test.py                    # Automated testing
```

## Learning Path

1. **Start**: Read main `README.md`
2. **Example 01**: Hello CUDA - understand kernels and threads
3. **Example 02**: Vector Add - learn complete workflow
4. **Example 03**: Device Query - know your hardware
5. **Exercise 01**: Vector Subtract - apply what you learned
6. **Exercise 02**: SAXPY - master scalar operations

## Common Commands

```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Compile single file
nvcc -o output_name source.cu

# Compile with architecture
nvcc -arch=sm_86 -o output_name source.cu

# Run with arguments
./vector_add 1000000        # 1M elements
./saxpy 100000 2.5          # size=100k, alpha=2.5

# Build with CMake (recommended)
mkdir build && cd build
cmake ..
make
```

## Troubleshooting

**Problem**: "nvcc: command not found"
```bash
# Add CUDA to PATH (adjust version)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Problem**: "no kernel image available"
```bash
# Find your GPU's compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Compile for your architecture (example for 8.6)
nvcc -arch=sm_86 -o program source.cu
```

**Problem**: Compilation errors
```bash
# Check CUDA version compatibility
nvcc --version
gcc --version  # Should be compatible with CUDA version
```

## Next Steps

After completing this chapter:
- Move to Chapter 02: Memory Optimization
- Explore CUDA samples: `/usr/local/cuda/samples/`
- Read NVIDIA CUDA Programming Guide

## Resources

- Chapter README: Detailed concepts and theory
- Example READMEs: Specific explanations for each example
- Problem.md files: Exercise requirements and hints
- NVIDIA CUDA Guide: https://docs.nvidia.com/cuda/

---

Happy learning!
