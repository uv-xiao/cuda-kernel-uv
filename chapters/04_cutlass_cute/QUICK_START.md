# Quick Start Guide - Chapter 04: CUTLASS and CuTe

## Prerequisites

1. **CUDA Toolkit** 11.0+
   ```bash
   nvcc --version
   ```

2. **CMake** 3.18+
   ```bash
   cmake --version
   ```

3. **NVIDIA GPU** with compute capability 7.0+ (Volta or newer)
   ```bash
   nvidia-smi
   ```

## Installation

### Step 1: Install CUTLASS

```bash
# Clone CUTLASS repository
cd ~
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
git checkout v3.4.0  # Use stable version

# Set environment variable
export CUTLASS_DIR=~/cutlass
echo "export CUTLASS_DIR=~/cutlass" >> ~/.bashrc
```

### Step 2: Build Chapter Examples

```bash
cd /home/uvxiao/cuda-kernel-tutorial/chapters/04_cutlass_cute

# Create build directory
mkdir build && cd build

# Configure
cmake .. -DCUTLASS_DIR=$CUTLASS_DIR

# Or specify GPU architecture explicitly:
# cmake .. -DCUTLASS_DIR=$CUTLASS_DIR -DCMAKE_CUDA_ARCHITECTURES=80

# Build all
make -j$(nproc)

# Or build specific targets
make test_cutlass
make gemm_tiled
make wmma_gemm
```

## Running Examples

### Example 01: CUTLASS Setup

```bash
cd build
./test_cutlass
```

Expected output: Verification of CUTLASS installation and basic CuTe operations.

### Example 02: CuTe Layouts

```bash
./layout_basics
./layout_operations
```

Learn about layout algebra and transformations.

### Example 03: GEMM with CuTe

```bash
# Simple GEMM (no tiling)
./gemm_simple

# Tiled GEMM with shared memory
./gemm_tiled
```

Compare performance with cuBLAS.

### Example 04: Tensor Cores

```bash
# WMMA GEMM (works on SM70+)
./wmma_gemm

# MMA GEMM (placeholder, requires SM80+)
./mma_gemm
```

Achieve >90% cuBLAS performance with FP16.

### Example 05: CuteDSL

```bash
# Python DSL example (conceptual)
python3 ../examples/05_cutedsl/gemm_python.py
```

## Running Exercises

### Exercise 01: Batched GEMM

```bash
cd build/exercises

# Test starter code (incomplete, may fail)
./batched_gemm_starter

# Test solution
./batched_gemm_solution

# Run test script
cd ../exercises/01_batched_gemm
python3 test.py
```

### Exercise 02: FP16 GEMM

```bash
cd build/exercises

# Test starter code
./fp16_gemm_starter

# Test solution
./fp16_gemm_solution

# Run test script
cd ../exercises/02_fp16_gemm
python3 test.py
```

## Profiling

### NSight Compute

```bash
# Profile kernel
ncu --set full -o profile ./gemm_tiled

# View specific metrics
ncu --metrics sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
./gemm_tiled

# For Tensor Cores
ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active \
./wmma_gemm
```

### NSight Systems

```bash
# Timeline profiling
nsys profile -o timeline ./gemm_tiled

# View in GUI
nsys-ui timeline.nsys-rep
```

## Common Issues

### Issue: "CUTLASS_DIR not set"

**Solution:**
```bash
export CUTLASS_DIR=/path/to/cutlass
cmake .. -DCUTLASS_DIR=$CUTLASS_DIR
```

### Issue: "cute/tensor.hpp: No such file"

**Solution:** Ensure CUTLASS is v3.x (not 2.x):
```bash
cd $CUTLASS_DIR
git checkout v3.4.0
```

### Issue: "Illegal memory access"

**Cause:** Misaligned memory for WMMA

**Solution:** Verify 16-byte alignment and correct layouts

### Issue: Wrong results

**Check:**
1. Layout (row-major vs column-major)
2. Leading dimensions
3. Synchronization (`__syncthreads()`)
4. Boundary conditions

## Performance Targets

| Example | Implementation | Target Performance |
|---------|----------------|-------------------|
| 03 | gemm_simple | 5-15% cuBLAS |
| 03 | gemm_tiled | 60-80% cuBLAS |
| 04 | wmma_gemm | 90%+ cuBLAS (FP16) |

## Learning Path

1. **Start:** Example 01 (setup verification)
2. **Foundations:** Example 02 (layout algebra)
3. **Implementation:** Example 03 (GEMM kernels)
4. **Optimization:** Example 04 (Tensor Cores)
5. **Practice:** Exercises 01-02

## Resources

- **README.md**: Comprehensive chapter overview
- **Each example's README.md**: Detailed explanations
- **CUTLASS Docs**: https://github.com/NVIDIA/cutlass/tree/main/media/docs
- **CuTe Tutorial**: https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/

## Next Steps

After completing this chapter:

1. **Chapter 05**: DeepGEMM - state-of-the-art implementations
2. **Study CUTLASS source**: Production-quality code
3. **Experiment**: Modify tile sizes, layouts, precision

## Getting Help

If you encounter issues:

1. Check example READMEs for specific guidance
2. Review CUTLASS documentation
3. Use NSight tools for debugging
4. Compare with solution code in exercises

---

**Ready to begin?** Start with `./test_cutlass` to verify your setup!
