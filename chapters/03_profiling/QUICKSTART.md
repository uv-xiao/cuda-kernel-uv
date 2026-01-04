# Chapter 03: Quick Start Guide

## 5-Minute Setup

```bash
# Navigate to chapter
cd /home/uvxiao/cuda-kernel-tutorial/chapters/03_profiling

# Build all examples
mkdir build && cd build
cmake ..
make -j

# Run all examples
./examples/01_matmul_naive/matmul_naive
./examples/02_matmul_coalesced/matmul_coalesced
./examples/03_matmul_tiled/matmul_tiled
./examples/04_matmul_optimized/matmul_optimized
```

## Expected Output

You should see performance progression:

```
Example 01 (Naive):        ~300 GFLOPS    (baseline)
Example 02 (Coalesced):    ~1,200 GFLOPS  (4x faster)
Example 03 (Tiled):        ~5,000 GFLOPS  (17x faster)
Example 04 (Optimized):    ~18,000 GFLOPS (60x faster!)
```

## Profile an Example

```bash
# Basic profile
ncu --set basic ./examples/01_matmul_naive/matmul_naive

# Full metrics
ncu --set full -o profile ./examples/01_matmul_naive/matmul_naive

# View in GUI
ncu-ui profile.ncu-rep
```

## Compare Two Versions

```bash
# Profile both
ncu --set full -o naive ./examples/01_matmul_naive/matmul_naive
ncu --set full -o optimized ./examples/04_matmul_optimized/matmul_optimized

# Compare
ncu --import naive.ncu-rep optimized.ncu-rep
```

## Try the Exercise

```bash
cd exercises/01_optimize_transpose

# Read the problem
cat problem.md

# Implement solution in starter.cu, then:
nvcc -O3 -o transpose starter.cu
./transpose

# Test your solution
python test.py
```

## Learning Path

1. **Start**: Read `README.md` for overview
2. **Examples**: Work through examples 01-04 in order
3. **Profiling**: Read `profiling/profiling_guide.md`
4. **Metrics**: Study `profiling/metrics_explained.md`
5. **Practice**: Complete the transpose exercise
6. **Reference**: Use `profiling/sample_commands.sh` for common commands

## Troubleshooting

### Build fails?
```bash
# Check CUDA installation
nvcc --version

# Check CMake version
cmake --version  # Need 3.18+
```

### Can't run ncu?
```bash
# Check if installed
ncu --version

# If permission error:
sudo ncu ./program
# Or disable restrictions (see profiling_guide.md)
```

### Performance much lower than expected?
```bash
# Check GPU
nvidia-smi

# Ensure nothing else is running on GPU
# Verify matrix size is large enough (N >= 2048)
```

## Next Steps

- Experiment with different matrix sizes
- Try different tile sizes in example 03
- Tune parameters in example 04 for your GPU
- Read the detailed READMEs in each example
- Complete bonus challenges in the exercise

## Getting Help

- Check individual README.md files in each directory
- Review SUMMARY.md for comprehensive overview
- Consult profiling_guide.md for tool usage
- See metrics_explained.md for metric interpretation

Happy optimizing!
