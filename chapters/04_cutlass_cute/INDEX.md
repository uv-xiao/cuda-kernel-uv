# Chapter 04: CUTLASS and CuTe - Complete Index

## 📚 Getting Started

| Document | Purpose | Read Time |
|----------|---------|-----------|
| [README.md](README.md) | Chapter overview, learning goals, key concepts | 15 min |
| [QUICK_START.md](QUICK_START.md) | Installation and running guide | 10 min |
| [CONTENT_SUMMARY.md](CONTENT_SUMMARY.md) | File structure and content details | 5 min |

**Start here:** [QUICK_START.md](QUICK_START.md) → [README.md](README.md)

---

## 🔧 Examples

### Example 01: CUTLASS Setup
**Verify installation and learn CuTe basics**

- 📖 [README](examples/01_cutlass_setup/README.md) - Installation guide
- 💻 [test_cutlass.cu](examples/01_cutlass_setup/test_cutlass.cu) - Basic CuTe operations
- ⚙️ [CMakeLists.txt](examples/01_cutlass_setup/CMakeLists.txt)

**Run:**
```bash
./build/test_cutlass
```

---

### Example 02: CuTe Layout
**Master layout algebra and transformations**

- 📖 [README](examples/02_cute_layout/README.md) - Layout concepts and algebra
- 💻 [layout_basics.cu](examples/02_cute_layout/layout_basics.cu) - Row/column-major, hierarchical
- 💻 [layout_operations.cu](examples/02_cute_layout/layout_operations.cu) - Composition, divide, swizzle
- ⚙️ [CMakeLists.txt](examples/02_cute_layout/CMakeLists.txt)

**Run:**
```bash
./build/layout_basics
./build/layout_operations
```

---

### Example 03: CuTe GEMM
**Implement high-performance GEMM**

- 📖 [README](examples/03_cute_gemm/README.md) - GEMM optimization strategies
- 💻 [gemm_simple.cu](examples/03_cute_gemm/gemm_simple.cu) - Naive implementation (5-15% cuBLAS)
- 💻 [gemm_tiled.cu](examples/03_cute_gemm/gemm_tiled.cu) - Tiled with shared memory (60-80% cuBLAS)
- ⚙️ [CMakeLists.txt](examples/03_cute_gemm/CMakeLists.txt)

**Run:**
```bash
./build/gemm_simple
./build/gemm_tiled
```

**Profile:**
```bash
ncu --set full -o gemm_profile ./build/gemm_tiled
```

---

### Example 04: Tensor Cores
**Leverage Tensor Cores for 10x speedup**

- 📖 [README](examples/04_tensor_cores/README.md) - WMMA/MMA guide
- 💻 [wmma_gemm.cu](examples/04_tensor_cores/wmma_gemm.cu) - FP16 WMMA (90%+ cuBLAS)
- 💻 [mma_gemm.cu](examples/04_tensor_cores/mma_gemm.cu) - MMA PTX placeholder
- ⚙️ [CMakeLists.txt](examples/04_tensor_cores/CMakeLists.txt)

**Run:**
```bash
./build/wmma_gemm  # Requires SM70+
```

**Profile Tensor Cores:**
```bash
ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active ./build/wmma_gemm
```

---

### Example 05: CuteDSL
**Explore Python DSL interface**

- 📖 [README](examples/05_cutedsl/README.md) - CuteDSL overview
- 🐍 [gemm_python.py](examples/05_cutedsl/gemm_python.py) - Conceptual Python example

**Run:**
```bash
python3 examples/05_cutedsl/gemm_python.py
```

---

## 📝 Exercises

### Exercise 01: Batched GEMM
**Implement batched matrix multiplication**

- 📋 [problem.md](exercises/01_batched_gemm/problem.md) - Exercise description
- 💻 [starter.cu](exercises/01_batched_gemm/starter.cu) - Template with TODOs
- ✅ [solution.cu](exercises/01_batched_gemm/solution.cu) - Reference implementation
- 🧪 [test.py](exercises/01_batched_gemm/test.py) - Automated testing
- ⚙️ [CMakeLists.txt](exercises/01_batched_gemm/CMakeLists.txt)

**Difficulty:** Intermediate
**Target:** 70%+ cuBLAS for batched operations

**Run:**
```bash
./build/exercises/batched_gemm_starter  # Your implementation
./build/exercises/batched_gemm_solution # Reference
python3 exercises/01_batched_gemm/test.py
```

---

### Exercise 02: FP16 GEMM
**Achieve >90% cuBLAS with Tensor Cores**

- 📋 [problem.md](exercises/02_fp16_gemm/problem.md) - Exercise description
- 💻 [starter.cu](exercises/02_fp16_gemm/starter.cu) - Template with TODOs
- ✅ [solution.cu](exercises/02_fp16_gemm/solution.cu) - Reference implementation
- 🧪 [test.py](exercises/02_fp16_gemm/test.py) - Automated testing
- ⚙️ [CMakeLists.txt](exercises/02_fp16_gemm/CMakeLists.txt)

**Difficulty:** Advanced
**Target:** 90%+ cuBLAS on SM70+ GPUs

**Run:**
```bash
./build/exercises/fp16_gemm_starter    # Your implementation
./build/exercises/fp16_gemm_solution   # Reference
python3 exercises/02_fp16_gemm/test.py
```

---

## 🗺️ Recommended Learning Path

### Path 1: Quick Overview (2 hours)
1. Read [QUICK_START.md](QUICK_START.md)
2. Run Example 01: `./build/test_cutlass`
3. Run Example 03: `./build/gemm_tiled`
4. Run Example 04: `./build/wmma_gemm`

### Path 2: Deep Understanding (8-10 hours)
1. **Setup** (30 min)
   - Read [README.md](README.md)
   - Install CUTLASS via [QUICK_START.md](QUICK_START.md)
   - Run Example 01

2. **Foundations** (2 hours)
   - Study [02_cute_layout/README.md](examples/02_cute_layout/README.md)
   - Run and modify `layout_basics.cu`
   - Run and modify `layout_operations.cu`

3. **Implementation** (3 hours)
   - Study [03_cute_gemm/README.md](examples/03_cute_gemm/README.md)
   - Understand `gemm_simple.cu`
   - Study `gemm_tiled.cu` in detail
   - Profile with NSight Compute

4. **Optimization** (2 hours)
   - Study [04_tensor_cores/README.md](examples/04_tensor_cores/README.md)
   - Understand WMMA API
   - Run `wmma_gemm.cu`
   - Profile Tensor Core utilization

5. **Practice** (3-4 hours)
   - Complete Exercise 01: Batched GEMM
   - Complete Exercise 02: FP16 GEMM
   - Compare with solutions

### Path 3: Research Focus (Selective)
- **Layout algebra**: Examples 01-02
- **High performance**: Examples 03-04
- **Python interface**: Example 05
- **Specific use case**: Relevant exercise

---

## 📊 Performance Reference

| Implementation | File | Precision | Target | Notes |
|----------------|------|-----------|--------|-------|
| Simple GEMM | gemm_simple.cu | FP32 | 5-15% | Educational baseline |
| Tiled GEMM | gemm_tiled.cu | FP32 | 60-80% | Production FP32 |
| WMMA GEMM | wmma_gemm.cu | FP16 | 90-95% | Tensor Cores |
| Batched GEMM | Exercise 01 | FP32 | 70%+ | Multiple matrices |
| FP16 GEMM | Exercise 02 | FP16 | 90%+ | Advanced Tensor Cores |

*Percentages relative to cuBLAS performance*

---

## 🔍 Quick Reference

### Essential Commands

```bash
# Build all
cd build && cmake .. -DCUTLASS_DIR=$CUTLASS_DIR && make -j$(nproc)

# Run specific example
./build/<executable_name>

# Profile with NSight Compute
ncu --set full -o profile ./build/<executable>

# Profile Tensor Cores
ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active ./build/wmma_gemm

# Test exercises
python3 exercises/01_batched_gemm/test.py
python3 exercises/02_fp16_gemm/test.py
```

### File Naming Convention

- `README.md` - Documentation and theory
- `*.cu` - CUDA/CuTe implementation
- `*.py` - Python scripts (testing or DSL)
- `CMakeLists.txt` - Build configuration
- `problem.md` - Exercise description
- `starter.cu` - Exercise template
- `solution.cu` - Reference implementation

---

## 🎯 Learning Objectives Checklist

After completing this chapter, you should be able to:

- [ ] Install and configure CUTLASS
- [ ] Create and manipulate CuTe layouts
- [ ] Apply layout transformations (composition, divide, swizzle)
- [ ] Implement tiled GEMM with shared memory
- [ ] Understand memory coalescing and bank conflicts
- [ ] Use WMMA API for Tensor Cores
- [ ] Achieve 60%+ cuBLAS performance (FP32)
- [ ] Achieve 90%+ cuBLAS performance (FP16)
- [ ] Profile kernels with NSight Compute
- [ ] Implement batched operations
- [ ] Debug and optimize CUDA kernels

---

## 📚 External Resources

### Official Documentation
- [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
- [CuTe Tutorial](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md)
- [CUDA Programming Guide - WMMA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)

### Additional Learning
- [CUTLASS 3.0 Blog](https://developer.nvidia.com/blog/cutlass-3-0-faster-ai-software-for-nvidia-hopper/)
- [How to Optimize GEMM](https://siboehm.com/articles/22/CUDA-MMM)
- [NSight Compute Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)

---

## 💡 Tips for Success

1. **Start simple**: Run Example 01 before attempting exercises
2. **Read READMEs**: Each example has detailed explanations
3. **Profile early**: Use NSight Compute to understand bottlenecks
4. **Compare with cuBLAS**: Always benchmark against the gold standard
5. **Experiment**: Modify tile sizes and layouts to see effects
6. **Ask questions**: Use CUTLASS GitHub discussions for help

---

## 📞 Getting Help

If you encounter issues:

1. Check the specific README for that example/exercise
2. Review [QUICK_START.md](QUICK_START.md) troubleshooting section
3. Verify CUTLASS version (v3.4.0 recommended)
4. Check GPU compute capability (SM70+ for Tensor Cores)
5. Use NSight tools for debugging
6. Compare with solution code

---

**Navigation:**
- 🏠 [Chapter README](README.md)
- 🚀 [Quick Start](QUICK_START.md)
- 📋 [Content Summary](CONTENT_SUMMARY.md)

**Status:** ✅ Complete - Ready to use
**Last updated:** 2026-01-02
