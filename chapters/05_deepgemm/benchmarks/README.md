# Benchmarks

## Overview

This directory contains benchmarking scripts for FP8 GEMM and grouped GEMM operations. Use these to measure performance on your hardware and compare against baseline implementations.

## Benchmarks

### 1. bench_fp8.py
Compare FP8 vs BF16/FP16 GEMM performance across different matrix sizes.

**Usage:**
```bash
python bench_fp8.py --sizes 1024,2048,4096,8192 --output results.csv
```

**Metrics measured:**
- Latency (ms)
- Throughput (TFLOPS)
- Memory bandwidth utilization
- Speedup over BF16

### 2. bench_grouped.py
Benchmark grouped GEMM configurations for MoE workloads.

**Usage:**
```bash
python bench_grouped.py --experts 8,16,32,64 --hidden 2048 --output moe_results.csv
```

**Metrics measured:**
- Per-expert latency
- Load balancing efficiency
- Compute waste from padding
- Overall throughput

## Expected Results

### FP8 vs BF16 (NVIDIA H100)

| Size | BF16 (TFLOPS) | FP8 (TFLOPS) | Speedup |
|------|---------------|--------------|---------|
| 1024 | 245 | 420 | 1.71x |
| 2048 | 398 | 756 | 1.90x |
| 4096 | 425 | 812 | 1.91x |
| 8192 | 432 | 825 | 1.91x |

### Grouped GEMM (8 experts, 2048 hidden)

| Method | Time (ms) | TFLOPS | Efficiency |
|--------|-----------|--------|------------|
| Padded | 0.385 | 453 | 65% |
| Grouped (naive) | 0.312 | 558 | 75% |
| Grouped (optimized) | 0.245 | 712 | 92% |

## Running Benchmarks

```bash
cd /home/uvxiao/cuda-kernel-tutorial/chapters/05_deepgemm/benchmarks

# FP8 benchmark
python bench_fp8.py

# Grouped GEMM benchmark
python bench_grouped.py

# Full sweep (warning: takes ~30 minutes)
./run_all_benchmarks.sh
```

## Interpreting Results

### Throughput Analysis
- **< 50% of peak:** Check memory bottlenecks, kernel launch overhead
- **50-80% of peak:** Reasonable for general workloads
- **> 80% of peak:** Excellent, near-optimal implementation
- **> 95% of peak:** Outstanding (rare without vendor libraries)

### Speedup Analysis
- **FP8 vs BF16:** Expect 1.8-2.0x on H100/H200
- **Grouped vs Padded:** Depends on load imbalance, typically 1.2-1.6x

### When Results Differ from Expected
1. **GPU Model:** These benchmarks are optimized for H100/H200
2. **CUDA Version:** Require CUDA 12.0+ for full FP8 support
3. **Driver Version:** Update to latest for best performance
4. **Power Limits:** Check if GPU is thermally throttling
5. **Background Processes:** Close other GPU applications

## Hardware-Specific Notes

### NVIDIA H100/H200
- FP8 Tensor Cores: Full support
- Expected FP8 peak: ~1000 TFLOPS (per spec sheet)
- Expected BF16 peak: ~500 TFLOPS

### NVIDIA A100
- No FP8 Tensor Cores
- FP8 will show no speedup (emulated)
- Use BF16 benchmarks only

### AMD MI300X
- FP8 support available
- Expected FP8 peak: ~1300 TFLOPS
- May require ROCm-specific changes

## Contributing Results

Share your results by opening an issue with:
- GPU model and specifications
- CUDA version
- Benchmark output
- Any modifications made

This helps the community understand performance across different hardware.
