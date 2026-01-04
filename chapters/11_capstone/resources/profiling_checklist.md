# Profiling Checklist for Capstone Projects

Use this checklist to ensure comprehensive profiling of your kernels.

## Pre-Profiling Setup

- [ ] Code compiles with `-lineinfo` flag for source correlation
- [ ] Optimizations enabled (`-O3` or equivalent)
- [ ] Debug assertions disabled in profiling builds
- [ ] Profiling tools installed (nsys, ncu)
- [ ] Test inputs prepared (representative workloads)
- [ ] Warmup iterations before profiling runs

## Timeline Profiling with Nsight Systems (nsys)

### Basic Profiling

- [ ] Generate nsys report:
  ```bash
  nsys profile -o profile_name --stats=true python your_script.py
  ```

### Metrics to Collect

- [ ] **Total runtime**
  - Wall clock time
  - GPU active time
  - CPU time

- [ ] **Kernel execution time**
  - Time per kernel launch
  - Number of kernel launches
  - Kernel launch overhead

- [ ] **Memory operations**
  - H2D (Host to Device) transfers
  - D2H (Device to Host) transfers
  - D2D (Device to Device) transfers
  - Time spent in memcpy

- [ ] **CUDA API overhead**
  - cudaMalloc time
  - cudaFree time
  - Synchronization time

### Analysis Tasks

- [ ] Identify top 5 kernels by time
- [ ] Check for CPU-GPU idle time (gaps)
- [ ] Verify kernel launch overhead <5% of total time
- [ ] Check for unnecessary synchronizations
- [ ] Look for opportunities to use CUDA streams
- [ ] Identify memory copy bottlenecks

### Timeline Analysis

- [ ] Open .nsys-rep file in nsys-ui
- [ ] Examine CUDA HW timeline
- [ ] Look for:
  - Gaps between kernel launches
  - Memory copy/compute overlap opportunities
  - Small kernels that could be fused
  - Warp stalls

### Export Data

- [ ] Export statistics to CSV
- [ ] Screenshot timeline for report
- [ ] Document key findings

---

## Kernel Profiling with Nsight Compute (ncu)

### Basic Profiling

- [ ] Profile specific kernel:
  ```bash
  ncu -o profile_kernel --set full python your_script.py
  ```

- [ ] Profile with specific metrics:
  ```bash
  ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
                  dram__throughput.avg.pct_of_peak_sustained_elapsed \
      python your_script.py
  ```

### Compute Utilization Metrics

- [ ] **SM (Streaming Multiprocessor) Utilization**
  - `sm__throughput.avg.pct_of_peak_sustained_elapsed`
  - Target: >70% for compute-bound kernels

- [ ] **Warp Execution Efficiency**
  - `smsp__warp_issue_efficiency`
  - Target: >80%

- [ ] **Occupancy**
  - `sm__warps_active.avg.pct_of_peak_sustained_active`
  - Target: >50% (not always critical)

- [ ] **Instruction Throughput**
  - FP32: `sm__sass_thread_inst_executed_op_fadd_pred_on.sum`
  - FP16: `sm__sass_thread_inst_executed_op_hadd_pred_on.sum`
  - Tensor Core: Check if TC instructions are used

### Memory Utilization Metrics

- [ ] **DRAM Bandwidth Utilization**
  - `dram__throughput.avg.pct_of_peak_sustained_elapsed`
  - Target: >70% for memory-bound kernels

- [ ] **L2 Cache Hit Rate**
  - `lts__t_sector_hit_rate.pct`
  - Target: >80% for kernels with data reuse

- [ ] **L1 Cache Hit Rate**
  - `l1tex__t_sector_hit_rate.pct`

- [ ] **Shared Memory Bandwidth**
  - `l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed`

### Memory Access Pattern Analysis

- [ ] **Global Memory Efficiency**
  - `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct`
  - Target: >80% (indicates good coalescing)

- [ ] **Bank Conflicts (Shared Memory)**
  - `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`
  - Target: 0 or minimal

- [ ] **Uncoalesced Global Access**
  - Check transaction efficiency
  - Look for scattered access patterns

### Warp Analysis

- [ ] **Warp Stall Reasons**
  - Memory throttle
  - Execution dependency
  - Synchronization
  - Memory dependency

- [ ] **Divergence**
  - Branch divergence
  - Thread divergence within warps

### Roofline Analysis

- [ ] Compute achieved FLOPS
- [ ] Compute achieved bandwidth (GB/s)
- [ ] Determine if kernel is:
  - Compute-bound
  - Memory-bound
  - Latency-bound

- [ ] Compare with theoretical roofline
- [ ] Identify optimization direction

### Export and Documentation

- [ ] Save ncu report (.ncu-rep)
- [ ] Export metrics to CSV
- [ ] Screenshot key sections:
  - Summary page
  - Memory workload analysis
  - Compute workload analysis
  - Source code hotspots

---

## Performance Analysis

### Theoretical Performance

- [ ] Calculate theoretical FLOPS for operation
- [ ] Calculate arithmetic intensity
- [ ] Determine theoretical occupancy
- [ ] Calculate theoretical bandwidth requirement

### Achieved Performance

- [ ] Measure actual FLOPS
- [ ] Measure actual bandwidth
- [ ] Calculate % of peak FLOPS
- [ ] Calculate % of peak bandwidth

### Comparison

- [ ] Compare with baseline (cuBLAS, cuDNN, PyTorch)
- [ ] Calculate speedup
- [ ] Identify performance gap
- [ ] List optimization opportunities

---

## Optimization Checklist

Based on profiling, check if you've applied:

### Memory Optimizations

- [ ] Coalesced global memory access
- [ ] Shared memory for data reuse
- [ ] Appropriate use of constant memory
- [ ] Texture memory for read-only data (if beneficial)
- [ ] Avoid bank conflicts
- [ ] Optimal tile size for cache utilization

### Compute Optimizations

- [ ] Sufficient occupancy (if needed)
- [ ] Minimize warp divergence
- [ ] Use of fast math operations (when appropriate)
- [ ] Tensor Core utilization (for GEMM/conv)
- [ ] Loop unrolling
- [ ] Register blocking

### Kernel Launch Configuration

- [ ] Optimal block size (profiled, not guessed)
- [ ] Grid size appropriate for GPU
- [ ] Enough blocks to saturate GPU
- [ ] Not too many blocks (launch overhead)

### System-Level Optimizations

- [ ] Kernel fusion (where beneficial)
- [ ] Async memory copies
- [ ] CUDA streams for overlap
- [ ] Persistent kernels (if applicable)
- [ ] Minimize host-device synchronization

---

## Common Issues and Solutions

### Issue: Low SM Utilization (<40%)

**Possible Causes**:
- Insufficient parallelism
- Too many resources per thread (low occupancy)
- Kernel too small

**Solutions**:
- Increase grid/block size
- Reduce register/shared memory usage
- Combine multiple small kernels

### Issue: Low Memory Bandwidth (<50% of peak)

**Possible Causes**:
- Uncoalesced memory access
- Small transfers
- Cache hit rate too high (not utilizing DRAM)

**Solutions**:
- Improve memory access patterns
- Increase tile size
- Check if kernel is actually compute-bound

### Issue: Low L2 Cache Hit Rate (<50%)

**Possible Causes**:
- Working set too large for cache
- Poor data reuse
- Strided access patterns

**Solutions**:
- Optimize tile sizes for cache
- Improve data reuse
- Use shared memory explicitly

### Issue: Shared Memory Bank Conflicts

**Possible Causes**:
- Multiple threads accessing same bank
- Strided shared memory access

**Solutions**:
- Pad shared memory arrays
- Change access pattern
- Use warp shuffle instead

### Issue: Warp Divergence

**Possible Causes**:
- Branches within warps take different paths
- Non-uniform data distribution

**Solutions**:
- Restructure algorithm to avoid divergence
- Sort/partition data
- Use predication instead of branches

---

## Profiling Report Template

Include in your capstone report:

### 1. Profiling Setup
- GPU used
- Profiling tools and versions
- Test configuration

### 2. Timeline Analysis (nsys)
- Screenshot of timeline
- Top kernels by time
- Memory transfer analysis
- Identified bottlenecks

### 3. Kernel Analysis (ncu)
- Per-kernel metrics table
- Roofline analysis
- Memory access patterns
- Optimization opportunities

### 4. Performance Comparison
- vs. baseline implementation
- vs. theoretical peak
- Speedup achieved
- Remaining performance gap

### 5. Optimization Journey
- Initial performance
- Optimizations applied
- Performance after each optimization
- Final performance

---

## Quick Reference: Key Metrics

| Metric | Tool | Good Value | Interpretation |
|--------|------|------------|----------------|
| SM Utilization | ncu | >70% | GPU is busy computing |
| DRAM Bandwidth | ncu | >70% | Memory system utilized |
| L2 Hit Rate | ncu | >80% | Good data reuse |
| Occupancy | ncu | >50% | Enough active warps |
| Memory Efficiency | ncu | >80% | Good coalescing |
| Warp Efficiency | ncu | >80% | Minimal divergence |
| Kernel Time | nsys | - | Compare with baseline |
| % of cuBLAS | custom | >70% | Good GEMM performance |

---

## Automation Script

Save this script to automate profiling:

```bash
#!/bin/bash
# profile.sh - Automated profiling script

KERNEL_NAME=$1
SCRIPT=$2

# Timeline profiling
echo "Running Nsight Systems profiling..."
nsys profile -o ${KERNEL_NAME}_nsys \
    --stats=true \
    -f true \
    ${SCRIPT}

# Kernel profiling
echo "Running Nsight Compute profiling..."
ncu -o ${KERNEL_NAME}_ncu \
    --set full \
    -f \
    ${SCRIPT}

# Generate report
echo "Profiling complete!"
echo "View timeline: nsys-ui ${KERNEL_NAME}_nsys.nsys-rep"
echo "View kernel details: ncu-ui ${KERNEL_NAME}_ncu.ncu-rep"
```

Usage:
```bash
./profile.sh my_kernel "python benchmark.py"
```

---

## Final Checklist Before Submission

- [ ] Profiled on target GPU architecture
- [ ] Collected both timeline and kernel metrics
- [ ] Compared with baselines
- [ ] Documented all profiling results
- [ ] Included screenshots in report
- [ ] Explained performance characteristics
- [ ] Listed attempted optimizations
- [ ] Identified remaining bottlenecks
- [ ] Provided profiling data files

---

Remember: **Profile, don't guess!** Every optimization should be motivated by profiling data.
