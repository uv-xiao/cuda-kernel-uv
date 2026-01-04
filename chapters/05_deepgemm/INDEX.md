# Chapter 05: DeepGEMM & Advanced GEMM Patterns - Complete Index

## Quick Navigation

### Getting Started
1. **First time here?** → Start with [README.md](README.md)
2. **Want to run code immediately?** → Check [QUICKSTART.md](QUICKSTART.md)
3. **Need complete overview?** → Read [SUMMARY.md](SUMMARY.md)
4. **Looking for specific file?** → See [FILES.md](FILES.md)

### By Topic

#### FP8 Fundamentals
- **Introduction:** [README.md](README.md) - "Why DeepGEMM?" section
- **Examples:** [examples/01_fp8_basics/](examples/01_fp8_basics/)
  - [fp8_types.cu](examples/01_fp8_basics/fp8_types.cu) - Format exploration
  - [fp8_conversion.cu](examples/01_fp8_basics/fp8_conversion.cu) - Conversion kernels
- **Guide:** [examples/01_fp8_basics/README.md](examples/01_fp8_basics/README.md)

#### Quantization
- **Theory:** [examples/02_quantization/README.md](examples/02_quantization/README.md)
- **Examples:**
  - [quantize.cu](examples/02_quantization/quantize.cu) - Basic quantization
  - [fine_grained_scaling.cu](examples/02_quantization/fine_grained_scaling.cu) - Advanced scaling
- **Best Practices:** [examples/02_quantization/README.md](examples/02_quantization/README.md) - "Best Practices" section

#### Grouped GEMM & MoE
- **Concept:** [examples/03_grouped_gemm/README.md](examples/03_grouped_gemm/README.md)
- **Examples:**
  - [grouped_gemm.cu](examples/03_grouped_gemm/grouped_gemm.cu) - Basic implementation
  - [variable_sizes.cu](examples/03_grouped_gemm/variable_sizes.cu) - Load balancing
- **Architecture:** [examples/03_grouped_gemm/README.md](examples/03_grouped_gemm/README.md) - "MoE Architecture Background"

#### DeepGEMM Integration
- **API Reference:** [examples/04_deepgemm_usage/README.md](examples/04_deepgemm_usage/README.md)
- **Examples:**
  - [dense_example.py](examples/04_deepgemm_usage/dense_example.py) - FP8 GEMM
  - [moe_example.py](examples/04_deepgemm_usage/moe_example.py) - MoE layer
- **Integration Guide:** [examples/04_deepgemm_usage/README.md](examples/04_deepgemm_usage/README.md) - "Integration with Transformer Models"

### By Activity

#### Building & Running
1. **Build Instructions:** [QUICKSTART.md](QUICKSTART.md) - "Building All Examples"
2. **Run Examples:** [QUICKSTART.md](QUICKSTART.md) - "Running Examples"
3. **Troubleshooting:** [QUICKSTART.md](QUICKSTART.md) - "Troubleshooting"

#### Benchmarking
1. **FP8 Performance:** [benchmarks/bench_fp8.py](benchmarks/bench_fp8.py)
2. **Grouped GEMM:** [benchmarks/bench_grouped.py](benchmarks/bench_grouped.py)
3. **Expected Results:** [benchmarks/README.md](benchmarks/README.md)

#### Learning & Practice
1. **Exercise 1:** [exercises/01_simple_grouped_gemm/problem.md](exercises/01_simple_grouped_gemm/problem.md)
2. **Starter Code:** [exercises/01_simple_grouped_gemm/starter.cu](exercises/01_simple_grouped_gemm/starter.cu)
3. **Solution:** [exercises/01_simple_grouped_gemm/solution.cu](exercises/01_simple_grouped_gemm/solution.cu)
4. **Testing:** [exercises/01_simple_grouped_gemm/test.py](exercises/01_simple_grouped_gemm/test.py)

### By Difficulty

#### Beginner (New to FP8)
1. Read: [README.md](README.md)
2. Run: [examples/01_fp8_basics/fp8_types.cu](examples/01_fp8_basics/fp8_types.cu)
3. Study: [examples/01_fp8_basics/README.md](examples/01_fp8_basics/README.md)

#### Intermediate (Familiar with GEMM)
1. Read: [examples/02_quantization/README.md](examples/02_quantization/README.md)
2. Run: [examples/02_quantization/fine_grained_scaling.cu](examples/02_quantization/fine_grained_scaling.cu)
3. Try: [exercises/01_simple_grouped_gemm/](exercises/01_simple_grouped_gemm/)

#### Advanced (Production Ready)
1. Study: [examples/03_grouped_gemm/variable_sizes.cu](examples/03_grouped_gemm/variable_sizes.cu)
2. Integrate: [examples/04_deepgemm_usage/moe_example.py](examples/04_deepgemm_usage/moe_example.py)
3. Benchmark: [benchmarks/bench_grouped.py](benchmarks/bench_grouped.py)

## Document Map

```
Legend:
📚 Documentation
💻 Code (CUDA)
🐍 Code (Python)
🔨 Build
✏️ Exercise

Root Level:
├── 📚 README.md (START HERE)
├── 📚 QUICKSTART.md
├── 📚 SUMMARY.md
├── 📚 FILES.md
├── 📚 INDEX.md (this file)
└── 🔨 CMakeLists.txt

examples/
├── 01_fp8_basics/
│   ├── 📚 README.md
│   ├── 💻 fp8_types.cu
│   ├── 💻 fp8_conversion.cu
│   └── 🔨 CMakeLists.txt
│
├── 02_quantization/
│   ├── 📚 README.md
│   ├── 💻 quantize.cu
│   ├── 💻 fine_grained_scaling.cu
│   └── 🔨 CMakeLists.txt
│
├── 03_grouped_gemm/
│   ├── 📚 README.md
│   ├── 💻 grouped_gemm.cu
│   ├── 💻 variable_sizes.cu
│   └── 🔨 CMakeLists.txt
│
└── 04_deepgemm_usage/
    ├── 📚 README.md
    ├── 🐍 dense_example.py
    └── 🐍 moe_example.py

benchmarks/
├── 📚 README.md
├── 🐍 bench_fp8.py
└── 🐍 bench_grouped.py

exercises/
└── 01_simple_grouped_gemm/
    ├── 📚 problem.md
    ├── ✏️ starter.cu
    ├── 💻 solution.cu
    └── 🐍 test.py
```

## Key Concepts Index

### A-F
- **Activation Functions:** examples/04_deepgemm_usage/moe_example.py
- **Batched GEMM:** examples/03_grouped_gemm/README.md
- **Block Size:** examples/02_quantization/README.md
- **Calibration:** examples/02_quantization/README.md
- **Compute Capability:** QUICKSTART.md
- **Conversion Kernels:** examples/01_fp8_basics/fp8_conversion.cu
- **DeepGEMM API:** examples/04_deepgemm_usage/README.md
- **Dequantization:** examples/02_quantization/quantize.cu
- **Dynamic Quantization:** examples/02_quantization/README.md
- **E4M3 Format:** examples/01_fp8_basics/README.md
- **E5M2 Format:** examples/01_fp8_basics/README.md
- **Epilogue Fusion:** README.md
- **Expert Parallelism:** examples/03_grouped_gemm/README.md
- **Fine-Grained Scaling:** examples/02_quantization/fine_grained_scaling.cu
- **FP8 Hardware Support:** examples/01_fp8_basics/README.md

### G-M
- **GEMM Performance:** benchmarks/bench_fp8.py
- **Grouped GEMM:** examples/03_grouped_gemm/grouped_gemm.cu
- **Load Balancing:** examples/03_grouped_gemm/variable_sizes.cu
- **Masked GEMM:** README.md
- **Mixed Precision:** examples/02_quantization/README.md
- **MoE Architecture:** examples/03_grouped_gemm/README.md
- **MoE Layer:** examples/04_deepgemm_usage/moe_example.py

### N-Z
- **Numerical Accuracy:** examples/02_quantization/README.md
- **Outlier Handling:** examples/02_quantization/fine_grained_scaling.cu
- **Padding Waste:** examples/03_grouped_gemm/README.md
- **Per-Channel Quantization:** examples/02_quantization/quantize.cu
- **Per-Tensor Quantization:** examples/02_quantization/quantize.cu
- **Persistent Kernels:** examples/03_grouped_gemm/README.md
- **Quantization Error:** benchmarks/README.md
- **Router:** examples/04_deepgemm_usage/moe_example.py
- **Scale Factors:** examples/02_quantization/README.md
- **Shared Memory Tiling:** exercises/01_simple_grouped_gemm/solution.cu
- **Static Quantization:** examples/02_quantization/README.md
- **Tensor Cores:** README.md
- **TFLOPS:** benchmarks/bench_fp8.py
- **Token Routing:** examples/03_grouped_gemm/README.md
- **Variable Sizes:** examples/03_grouped_gemm/variable_sizes.cu
- **Work Stealing:** examples/03_grouped_gemm/variable_sizes.cu

## Function Reference

### CUDA Kernels
- `float_to_fp8_e4m3()` - [fp8_conversion.cu](examples/01_fp8_basics/fp8_conversion.cu)
- `fp8_e4m3_to_float()` - [fp8_conversion.cu](examples/01_fp8_basics/fp8_conversion.cu)
- `quantize_per_tensor_kernel()` - [quantize.cu](examples/02_quantization/quantize.cu)
- `quantize_per_channel_kernel()` - [quantize.cu](examples/02_quantization/quantize.cu)
- `quantize_fine_grained_fused_kernel()` - [fine_grained_scaling.cu](examples/02_quantization/fine_grained_scaling.cu)
- `grouped_gemm_tiled_kernel()` - [grouped_gemm.cu](examples/03_grouped_gemm/grouped_gemm.cu)
- `grouped_gemm_work_stealing()` - [variable_sizes.cu](examples/03_grouped_gemm/variable_sizes.cu)

### Python Functions
- `run_dense_gemm_example()` - [dense_example.py](examples/04_deepgemm_usage/dense_example.py)
- `compare_fp8_vs_bf16()` - [dense_example.py](examples/04_deepgemm_usage/dense_example.py)
- `benchmark_moe()` - [moe_example.py](examples/04_deepgemm_usage/moe_example.py)
- `benchmark_gemm()` - [bench_fp8.py](benchmarks/bench_fp8.py)
- `simulate_token_distribution()` - [bench_grouped.py](benchmarks/bench_grouped.py)

## Performance Metrics Index

### Target Performance (H100)
- Dense FP8 GEMM: 800+ TFLOPS - [benchmarks/README.md](benchmarks/README.md)
- Dense BF16 GEMM: 400+ TFLOPS - [benchmarks/README.md](benchmarks/README.md)
- Grouped GEMM: 700+ TFLOPS - [benchmarks/README.md](benchmarks/README.md)

### Accuracy Metrics
- Per-Tensor MAE: 0.015 - [SUMMARY.md](SUMMARY.md)
- Per-Channel MAE: 0.008 - [SUMMARY.md](SUMMARY.md)
- Fine-Grained MAE: 0.003 - [SUMMARY.md](SUMMARY.md)

### Efficiency
- Grouped vs Padded: 1.3-1.6x - [examples/03_grouped_gemm/README.md](examples/03_grouped_gemm/README.md)
- FP8 vs BF16: 1.8-2.0x - [benchmarks/README.md](benchmarks/README.md)

## FAQ Quick Links

**Q: How do I get started?**
→ [QUICKSTART.md](QUICKSTART.md)

**Q: What is FP8?**
→ [examples/01_fp8_basics/README.md](examples/01_fp8_basics/README.md)

**Q: What is grouped GEMM?**
→ [examples/03_grouped_gemm/README.md](examples/03_grouped_gemm/README.md)

**Q: How do I benchmark my GPU?**
→ [benchmarks/README.md](benchmarks/README.md)

**Q: What is fine-grained scaling?**
→ [examples/02_quantization/README.md](examples/02_quantization/README.md)

**Q: How do I use DeepGEMM?**
→ [examples/04_deepgemm_usage/README.md](examples/04_deepgemm_usage/README.md)

**Q: Where are the exercises?**
→ [exercises/01_simple_grouped_gemm/](exercises/01_simple_grouped_gemm/)

**Q: Compilation error?**
→ [QUICKSTART.md](QUICKSTART.md) - "Troubleshooting"

## External Resources

### DeepGEMM Repository
https://github.com/deepseek-ai/DeepSeek-V3/tree/main/inference/kernels/DeepGEMM

### Papers
- FP8 Formats: See [README.md](README.md) - "Additional Resources"
- DeepSeek-V3: See [README.md](README.md) - "Additional Resources"

### NVIDIA Documentation
- FP8 Guide: See [examples/01_fp8_basics/README.md](examples/01_fp8_basics/README.md)
- CUTLASS: See [README.md](README.md) - "Prerequisites"

## Version Information

- **Chapter Version:** 1.0
- **Last Updated:** January 2, 2025
- **Total Files:** 26
- **Total Lines:** ~6,270 (code + docs)

---

**Navigation Tip:** Use your editor's "Go to Definition" or search functionality to quickly jump between files!
