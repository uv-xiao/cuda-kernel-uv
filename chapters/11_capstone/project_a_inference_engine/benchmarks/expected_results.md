# Expected Benchmark Results

This document provides expected performance targets for different GPU architectures. Use these as a guide for evaluating your implementation.

## Hardware Specifications

### NVIDIA H100 (80GB)
- Compute Capability: 9.0
- Memory Bandwidth: 3.35 TB/s
- FP16 Peak Performance: 989 TFLOPS
- L2 Cache: 50 MB

### NVIDIA A100 (40GB/80GB)
- Compute Capability: 8.0
- Memory Bandwidth: 1.55 TB/s (40GB) / 2.0 TB/s (80GB)
- FP16 Peak Performance: 312 TFLOPS
- L2 Cache: 40 MB

### NVIDIA A10
- Compute Capability: 8.6
- Memory Bandwidth: 600 GB/s
- FP16 Peak Performance: 125 TFLOPS
- L2 Cache: 6 MB

---

## Flash Attention Benchmarks

### Configuration: batch=16, seq_len=2048, heads=16, head_dim=64

| Implementation | H100 (ms) | A100 (ms) | A10 (ms) | Notes |
|----------------|-----------|-----------|----------|-------|
| PyTorch Native | 12.5 | 25.0 | 65.0 | Baseline (slow) |
| Reference Flash Attention | 3.2 | 6.8 | 18.5 | Official implementation |
| **Target (Your Implementation)** | **<4.5** | **<9.5** | **<26** | >70% of reference |
| Excellent | <3.8 | <8.0 | <22 | >80% of reference |

### Configuration: batch=32, seq_len=4096, heads=32, head_dim=128

| Implementation | H100 (ms) | A100 (ms) | A10 (ms) |
|----------------|-----------|-----------|----------|
| PyTorch Native | 95.0 | 180.0 | 450.0 |
| Reference Flash Attention | 18.5 | 42.0 | 115.0 |
| **Target** | **<26** | **<60** | **<165** |
| Excellent | <22 | <50 | <138 |

### Memory Usage

Flash Attention should achieve O(N) memory complexity:

| Sequence Length | Standard Attention Memory | Flash Attention Memory | Reduction |
|-----------------|---------------------------|------------------------|-----------|
| 2048 | 2.1 GB | 0.6 GB | 3.5x |
| 4096 | 8.4 GB | 1.2 GB | 7x |
| 8192 | 33.6 GB | 2.4 GB | 14x |

---

## MoE Layer Benchmarks

### Configuration: hidden=4096, intermediate=11008, experts=32, top_k=2

| Batch | Seq Len | H100 (ms) | A100 (ms) | A10 (ms) | Throughput (H100) |
|-------|---------|-----------|-----------|----------|-------------------|
| 4 | 512 | 3.8 | 8.5 | 22.0 | 538K tokens/s |
| 16 | 2048 | 22.0 | 48.0 | 125.0 | 1.49M tokens/s |
| 32 | 2048 | 38.0 | 85.0 | 220.0 | 1.72M tokens/s |

### Expert Utilization Targets

Good load balancing is critical for MoE performance:

| Metric | Target | Excellent |
|--------|--------|-----------|
| Load balance loss | <0.01 | <0.001 |
| Expert usage variance | <10% | <5% |
| Router overhead | <5% of total time | <2% |

### GEMM Performance

Expert GEMMs should approach cuBLAS performance:

| Configuration | cuBLAS (TFLOPS) | Target (TFLOPS) | % of Peak |
|---------------|-----------------|-----------------|-----------|
| H100, M=8192, N=11008, K=4096 | 850 | >640 | >75% |
| A100, M=8192, N=11008, K=4096 | 280 | >210 | >75% |

---

## End-to-End Inference Benchmarks

### Small Model (12 layers, 768 hidden, 12 heads)

| Batch | Seq Len | H100 (ms) | A100 (ms) | A10 (ms) | Tokens/s (H100) |
|-------|---------|-----------|-----------|----------|-----------------|
| 1 | 128 | 0.8 | 1.8 | 4.5 | 160K |
| 4 | 512 | 4.2 | 9.5 | 24.0 | 488K |
| 16 | 2048 | 28.0 | 62.0 | 155.0 | 1.17M |

### Medium Model (24 layers, 2048 hidden, 16 heads)

| Batch | Seq Len | H100 (ms) | A100 (ms) | A10 (ms) | Tokens/s (H100) |
|-------|---------|-----------|-----------|----------|-----------------|
| 1 | 128 | 2.5 | 5.8 | 14.5 | 51K |
| 4 | 512 | 15.0 | 34.0 | 85.0 | 136K |
| 8 | 1024 | 45.0 | 98.0 | 245.0 | 182K |

### Large Model (32 layers, 4096 hidden, 32 heads)

| Batch | Seq Len | H100 (ms) | A100 (ms) | A10 (ms) | Tokens/s (H100) |
|-------|---------|-----------|-----------|----------|-----------------|
| 1 | 128 | 6.2 | 14.5 | 36.0 | 20.6K |
| 4 | 512 | 38.0 | 86.0 | 215.0 | 53.9K |
| 8 | 2048 | 185.0 | 410.0 | 1025.0 | 88.6K |

---

## Profiling Metrics

### Compute Utilization

Target GPU utilization for different operations:

| Operation | Target SM Utilization | Target Tensor Core Utilization |
|-----------|----------------------|--------------------------------|
| Flash Attention | >60% | >70% (if using TCs) |
| MoE GEMM | >80% | >85% |
| Router | >40% | N/A |
| Layer Norm | >50% | N/A |

### Memory Bandwidth

Target memory bandwidth utilization:

| Operation | Target BW Utilization | Notes |
|-----------|----------------------|-------|
| Flash Attention | >70% | Memory-bound |
| Large GEMM | >80% | Compute-bound when large |
| Small GEMM | >60% | More memory-bound |
| Element-wise ops | >75% | Bandwidth-bound |

### Kernel Launch Overhead

| Metric | Target |
|--------|--------|
| Average kernel launch overhead | <50 μs |
| Memory copy overhead | <1% of total time |
| Kernel fusion benefit | >10% speedup |

---

## Performance Optimization Checklist

Use this checklist to verify you've applied key optimizations:

### Attention Optimizations
- [ ] Tiling for L2 cache reuse
- [ ] Online softmax to avoid materializing full attention matrix
- [ ] Shared memory usage optimized (no bank conflicts)
- [ ] Warp-level optimizations (shuffle, reduce)
- [ ] Proper handling of non-power-of-2 sequence lengths
- [ ] Causal masking implemented efficiently

### MoE Optimizations
- [ ] Efficient TopK implementation (parallel reduction)
- [ ] Token batching by expert (minimize launch overhead)
- [ ] CUTLASS or cuBLAS for expert GEMMs
- [ ] Load balancing loss implemented
- [ ] Minimal memory copies
- [ ] Fused combining kernel

### System-Level Optimizations
- [ ] Kernel fusion where beneficial
- [ ] Memory pool/buffer reuse
- [ ] CUDA streams for overlap (if applicable)
- [ ] Proper synchronization (no unnecessary syncs)
- [ ] Mixed precision (FP16/BF16)
- [ ] Persistent kernels for small operations

---

## Common Performance Issues

### Issue: Low Attention Performance

**Symptoms**: <40% of reference Flash Attention

**Possible Causes**:
- Not using tiling (materializing full O(N²) attention matrix)
- Poor memory access patterns
- Low occupancy
- Not using online softmax
- Unnecessary synchronization

**Solutions**:
- Profile with `ncu` to identify bottleneck
- Check L2 cache hit rate
- Verify shared memory usage
- Check occupancy with `--metrics achieved_occupancy`

### Issue: MoE Slower than Expected

**Symptoms**: <50% of cuBLAS throughput

**Possible Causes**:
- Not batching tokens by expert
- Poor load balancing (some experts idle)
- Inefficient routing (TopK is slow)
- Too many small GEMMs
- Memory copy overhead

**Solutions**:
- Profile router separately
- Check expert usage distribution
- Use CUTLASS for small-batch GEMMs
- Verify batching strategy
- Minimize memory copies

### Issue: End-to-End Slower than Components

**Symptoms**: Individual kernels are fast but full model is slow

**Possible Causes**:
- Kernel launch overhead
- Memory allocation in hot path
- Unnecessary data copies
- Poor memory layout
- Lack of kernel fusion

**Solutions**:
- Profile with `nsys` to see timeline
- Identify kernel launch overhead
- Fuse small kernels
- Pre-allocate buffers
- Use memory pools

---

## Validation Criteria

### Numerical Correctness

| Metric | Acceptable | Excellent |
|--------|-----------|-----------|
| Max absolute error | <1e-3 | <1e-4 |
| Mean absolute error | <1e-4 | <1e-5 |
| Max relative error | <1e-2 | <1e-3 |

### Performance Gates

Your implementation should meet these minimum requirements:

| Component | Minimum Performance |
|-----------|---------------------|
| Flash Attention | >60% of reference |
| MoE Layer | >70% of cuBLAS |
| End-to-End | >40% speedup vs PyTorch naive |

**Grade Boundaries**:
- A: >80% of reference for both attention and MoE
- B: >70% of reference for both
- C: >60% of reference for both
- D: >40% of reference for both

---

## Reporting Requirements

Your benchmark report should include:

1. **Hardware Used**: GPU model, driver version, CUDA version
2. **Results Table**: All configurations tested with timings
3. **Speedup Graphs**: Visual comparison with baselines
4. **Profiling Data**: Screenshots from nsys/ncu showing:
   - Timeline view
   - Top kernels by time
   - Memory bandwidth utilization
   - Compute utilization
5. **Analysis**: Discussion of results, bottlenecks, optimization attempts

---

## Example Report Section

```
## Flash Attention Results (A100 GPU)

Configuration: batch=16, seq_len=2048, heads=16, head_dim=64

Results:
- PyTorch Native: 25.2 ms
- Reference Flash Attention: 6.8 ms
- Our Implementation: 8.4 ms
- Speedup vs PyTorch: 3.0x
- % of Reference: 81%

Profiling Insights:
- SM Utilization: 68%
- Memory Bandwidth: 72% of peak
- L2 Cache Hit Rate: 85%

Bottleneck: Slightly lower SM utilization than reference, likely due to
suboptimal block sizing for this configuration.

Optimization Attempts:
1. Tried block size 128x128 (current)
2. Tried block size 64x64 (slower - 9.2ms)
3. Tried block size 256x256 (OOM for large sequences)

Future Work:
- Adaptive block sizing based on sequence length
- Better tuning of shared memory usage
```

This level of detail demonstrates thorough understanding and analysis.
