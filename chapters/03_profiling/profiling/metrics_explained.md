# CUDA Performance Metrics Explained

## Introduction

This guide explains the most important CUDA performance metrics, how to interpret them, and what to do when they indicate problems.

## Table of Contents

1. [Memory Metrics](#memory-metrics)
2. [Compute Metrics](#compute-metrics)
3. [Occupancy Metrics](#occupancy-metrics)
4. [Warp Metrics](#warp-metrics)
5. [Cache Metrics](#cache-metrics)
6. [Diagnostic Workflow](#diagnostic-workflow)

---

## Memory Metrics

### DRAM Throughput

**Metric**: `dram__throughput.avg.pct_of_peak_sustained_elapsed`

**What it means**: Percentage of peak DRAM bandwidth utilized

**How to read**:
- **>70%**: Excellent memory bandwidth utilization
- **40-70%**: Good, may have room for improvement
- **<40%**: Poor utilization, investigate access patterns

**Common causes of low throughput**:
- Uncoalesced memory accesses
- Small data transfers
- Cache hits (not bad! means data reuse is good)
- Compute-bound kernel (expected)

**How to improve**:
- Ensure coalesced access patterns
- Use shared memory for data reuse
- Increase problem size per thread
- Vectorize loads/stores (`float4`)

**ncu command**:
```bash
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./program
```

---

### Global Memory Load/Store Efficiency

**Metrics**:
- `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct`
- `smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct`

**What it means**: Percentage of loaded/stored data that is actually used

**How to read**:
- **100%**: Perfect - all loaded data is used
- **50%**: Half the loaded data is wasted (common for strided access)
- **25%**: Severe inefficiency (e.g., stride-4 access on warp)

**Example**:
```cuda
// Good: 100% efficiency
// Warp threads access consecutive addresses
float val = data[threadIdx.x];

// Bad: ~3% efficiency
// Warp threads access every 32nd element
float val = data[threadIdx.x * 32];
```

**How to improve**:
- Transpose data layouts
- Use shared memory to coalesce accesses
- Reorganize data structures for sequential access

**ncu command**:
```bash
ncu --section MemoryWorkloadAnalysis ./program
```

---

### Global Memory Transactions

**Metrics**:
- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` (loads)
- `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` (stores)

**What it means**: Number of 32-byte memory transactions

**How to read**:
- **Minimum**: Num_threads / 32 * sizeof(element) / 32 (fully coalesced)
- **Actual > Minimum**: Some inefficiency
- **Actual >> Minimum**: Severe inefficiency

**Example for 1024 threads loading 4-byte floats**:
- **Ideal**: 1024 / 32 * 4 / 32 = 4 transactions
- **Actual = 128**: Each thread causes its own transaction (32× overhead!)

**How to improve**:
- Fix coalescing issues
- Use shared memory
- Pad arrays to avoid conflicts

**ncu command**:
```bash
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum ./program
```

---

### Shared Memory Bank Conflicts

**Metric**: `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`

**What it means**: Number of times multiple threads in a warp accessed the same shared memory bank

**How to read**:
- **0**: Perfect - no bank conflicts
- **1-100**: Minor conflicts, usually acceptable
- **>1000**: Significant conflicts, serializing accesses

**Why it matters**: Bank conflicts serialize accesses, reducing effective bandwidth by up to 32×

**Example**:
```cuda
__shared__ float data[32][32];

// Bad: Bank conflict (column access)
float val = data[0][threadIdx.x];  // All threads hit bank threadIdx.x % 32

// Good: No conflict (row access)
float val = data[threadIdx.x][0];  // Consecutive threads → different banks

// Good: Padded array
__shared__ float data[32][33];  // +1 padding avoids conflicts
```

**How to improve**:
- Access shared memory with stride-1 pattern
- Add padding to arrays: `[N][M+1]` instead of `[N][M]`
- Use `__shfl_sync` for inter-thread communication instead

**ncu command**:
```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum ./program
```

---

## Compute Metrics

### SM Throughput

**Metric**: `sm__throughput.avg.pct_of_peak_sustained_elapsed`

**What it means**: Percentage of peak computational throughput achieved

**How to read**:
- **>60%**: Excellent compute utilization
- **30-60%**: Good, kernel is reasonably compute-bound
- **<30%**: Low, likely memory-bound or latency-bound

**Common causes of low throughput**:
- Memory-bound kernel (most common)
- Insufficient parallelism
- Warp stalls due to dependencies
- Low occupancy

**How to improve**:
- Increase arithmetic intensity (more FLOPs per byte)
- Use shared memory tiling
- Increase work per thread
- Reduce memory access frequency

**ncu command**:
```bash
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ./program
```

---

### Instructions Per Cycle (IPC)

**Metric**: `smsp__average_inst_executed_per_cycle_active.ratio`

**What it means**: Average number of instructions executed per clock cycle

**How to read**:
- **>3**: Excellent instruction-level parallelism
- **1-3**: Moderate ILP, room for improvement
- **<1**: Poor ILP, significant stalls or dependencies

**How to improve**:
- Increase independent operations (register blocking)
- Loop unrolling
- Reduce dependencies between instructions
- Use multiple accumulators

**Example**:
```cuda
// Low ILP: Sequential dependencies
for (int i = 0; i < N; i++) {
    sum = sum + data[i];  // Each iteration depends on previous
}

// High ILP: Multiple independent accumulators
float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
for (int i = 0; i < N; i += 4) {
    sum0 += data[i];
    sum1 += data[i+1];
    sum2 += data[i+2];
    sum3 += data[i+3];
}
```

**ncu command**:
```bash
ncu --section ComputeWorkloadAnalysis ./program
```

---

### Floating-Point Operations

**Metrics**:
- `sm__sass_thread_inst_executed_op_fadd_pred_on.sum` (FP adds)
- `sm__sass_thread_inst_executed_op_fmul_pred_on.sum` (FP multiplies)
- `sm__sass_thread_inst_executed_op_ffma_pred_on.sum` (FP fused multiply-add)

**What it means**: Count of floating-point operations executed

**How to use**:
- Calculate GFLOPS: `(fadd + fmul + 2*ffma) / time_seconds / 1e9`
- Verify your kernel does expected number of operations
- Compare with theoretical peak

**ncu command**:
```bash
ncu --metrics sm__sass_thread_inst_executed_op_fadd_pred_on.sum,sm__sass_thread_inst_executed_op_fmul_pred_on.sum,sm__sass_thread_inst_executed_op_ffma_pred_on.sum ./program
```

---

## Occupancy Metrics

### Achieved Occupancy

**Metric**: `sm__warps_active.avg.pct_of_peak_sustained_active`

**What it means**: Average percentage of active warps vs. maximum possible

**How to read**:
- **50-100%**: Good, sufficient parallelism to hide latency
- **25-50%**: Moderate, may benefit from optimization
- **<25%**: Low, likely hurting performance

**Common causes of low occupancy**:
- Too many registers per thread
- Too much shared memory per block
- Too few threads per block
- Large thread block dimensions

**How to improve**:
- Reduce registers: simplify code, reduce local variables
- Reduce shared memory usage
- Increase block size (e.g., 256 → 512 threads)
- Tune launch configuration

**Important note**: High occupancy doesn't guarantee high performance! A kernel can have 100% occupancy and still be slow if it's memory-bound.

**ncu command**:
```bash
ncu --section Occupancy ./program
```

---

### Theoretical Occupancy

**Metric**: `sm__maximum_warps_per_active_cycle_pct`

**What it means**: Maximum possible occupancy given resource constraints

**How to read**:
- **100%**: No resource constraints
- **<100%**: Limited by registers, shared memory, or block size

**Limiting factors**:
1. **Registers**: Check `Function Cache Configuration` for register usage
2. **Shared Memory**: Check `Shared Memory Configuration`
3. **Block Size**: Too small blocks waste resources

**Example**:
```
Registers per thread: 64
Shared memory per block: 48 KB
Block size: 256 threads

GPU has:
- 65536 registers per SM
- 164 KB shared memory per SM
- Max 2048 threads per SM

Limit by registers: 65536 / (64 * 256) = 4 blocks → 1024 threads → 50% occupancy
Limit by shared mem: 164 / 48 = 3 blocks → 768 threads → 37.5% occupancy
Limit by threads: 2048 / 256 = 8 blocks → 2048 threads → 100% occupancy

Theoretical occupancy: min(50%, 37.5%, 100%) = 37.5% (limited by shared memory)
```

**How to improve**:
- Reduce resource usage
- Increase threads per block
- Use smaller data types where possible

**ncu command**:
```bash
ncu --section LaunchStats ./program
```

---

## Warp Metrics

### Warp Execution Efficiency

**Metric**: `smsp__sass_average_branch_targets_threads_uniform.pct`

**What it means**: Percentage of threads in a warp that follow the same execution path

**How to read**:
- **100%**: No divergence, all threads execute same instructions
- **<100%**: Warp divergence present, some threads are masked off

**Why it matters**: Divergent warps execute both branches serially, wasting cycles

**Example**:
```cuda
// Bad: High divergence
if (threadIdx.x % 2 == 0) {
    // Half of threads execute this
} else {
    // Other half execute this
}
// Warp executes both branches, 50% efficiency

// Good: No divergence (entire warp takes same path)
if (blockIdx.x % 2 == 0) {
    // All threads in block execute this together
}
```

**How to improve**:
- Minimize divergent branches
- Reorganize data to group similar threads
- Use warp-level primitives when possible

**ncu command**:
```bash
ncu --section WarpStateStats ./program
```

---

### Warp Stall Reasons

**Metrics**:
- `smsp__average_warps_issue_stalled_barrier.pct` (waiting at `__syncthreads()`)
- `smsp__average_warps_issue_stalled_long_scoreboard.pct` (waiting for memory)
- `smsp__average_warps_issue_stalled_short_scoreboard.pct` (waiting for compute)
- `smsp__average_warps_issue_stalled_not_selected.pct` (scheduler didn't pick it)

**What it means**: Why warps are not executing instructions

**How to read**: Look for the dominant stall reason

**Interpretations**:

1. **Barrier stalls high**: Too much synchronization
   - Reduce `__syncthreads()` calls
   - Consider using warp-level primitives
   - Increase work between syncs

2. **Long scoreboard high**: Memory-bound
   - Improve memory access patterns
   - Increase arithmetic intensity
   - Use shared memory

3. **Short scoreboard high**: Instruction dependencies
   - Increase ILP (multiple accumulators)
   - Loop unrolling
   - Reduce data dependencies

4. **Not selected high**: Low occupancy or insufficient work
   - Increase occupancy
   - Increase problem size

**ncu command**:
```bash
ncu --section WarpStateStats ./program
```

---

## Cache Metrics

### L1 Cache Hit Rate

**Metric**: `l1tex__t_sector_hit_rate.pct`

**What it means**: Percentage of memory requests served by L1 cache

**How to read**:
- **>80%**: Excellent data reuse
- **40-80%**: Moderate reuse
- **<40%**: Little reuse or working set too large

**Why it matters**: L1 cache is ~20× faster than global memory

**How to improve**:
- Use shared memory for explicit caching
- Improve temporal locality (reuse data quickly)
- Improve spatial locality (access nearby data)
- Tile algorithms to fit in cache

**Note**: For many kernels, low L1 hit rate is expected (e.g., streaming workloads)

**ncu command**:
```bash
ncu --metrics l1tex__t_sector_hit_rate.pct ./program
```

---

### L2 Cache Hit Rate

**Metric**: `lts__t_sector_hit_rate.pct`

**What it means**: Percentage of memory requests served by L2 cache

**How to read**:
- **>60%**: Good data reuse across blocks
- **30-60%**: Moderate reuse
- **<30%**: Little reuse or working set exceeds L2

**Why it matters**: L2 cache is shared across all SMs, enabling data sharing

**How to improve**:
- Organize work to improve locality across blocks
- Process data in smaller chunks that fit in L2
- Use persistent kernels to keep data in cache

**ncu command**:
```bash
ncu --metrics lts__t_sector_hit_rate.pct ./program
```

---

## Diagnostic Workflow

### Step 1: Identify Bottleneck Type

Run basic profiling:

```bash
ncu --set basic ./program
```

Look at **Speed of Light** section:

| SM Throughput | Memory Throughput | Bottleneck |
|---------------|-------------------|------------|
| High (>60%) | Low (<40%) | Compute-bound ✓ |
| Low (<30%) | High (>60%) | Memory-bound ✗ |
| Low | Low | Latency-bound or inefficient |
| High | High | Well-optimized! |

### Step 2: Drill Into Memory (if memory-bound)

```bash
ncu --section MemoryWorkloadAnalysis ./program
```

Check:
- **Global Load/Store Efficiency**: Should be >80%
- **L1/L2 Hit Rates**: Higher is better
- **Shared Memory Bank Conflicts**: Should be 0

### Step 3: Drill Into Compute (if compute-bound)

```bash
ncu --section ComputeWorkloadAnalysis ./program
```

Check:
- **IPC**: Should be >1, ideally >3
- **Warp Execution Efficiency**: Should be >95%
- **Instruction Mix**: Is it doing what you expect?

### Step 4: Check Occupancy

```bash
ncu --section Occupancy ./program
```

Check:
- **Achieved Occupancy**: >50% generally good
- **Theoretical Occupancy**: What's limiting it?
- **Launch Configuration**: Could you use more threads?

### Step 5: Investigate Stalls

```bash
ncu --section WarpStateStats ./program
```

Look at stall reasons to understand what's blocking progress.

### Step 6: Optimize and Re-profile

Apply targeted optimization based on findings, then re-profile to verify improvement.

---

## Quick Reference Table

| Metric | Good Value | Indicates Problem If... | How to Fix |
|--------|------------|------------------------|------------|
| DRAM Throughput | >70% | <40% | Coalescing, tiling, reuse |
| SM Throughput | >60% | <30% | Increase arithmetic intensity |
| Global Load Efficiency | >80% | <60% | Fix access patterns |
| Shared Bank Conflicts | 0 | >100 | Padding, stride-1 access |
| Occupancy | 50-75% | <25% | Reduce resources, increase threads |
| Warp Efficiency | >95% | <80% | Reduce divergence |
| IPC | >3 | <1 | Increase ILP, unroll loops |
| L1 Hit Rate | - | - | Not always important |
| L2 Hit Rate | >30% | - | Improve locality |

---

## Resources

- [Nsight Compute Metrics Reference](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [GPU Architecture Whitepapers](https://www.nvidia.com/en-us/data-center/resources/gpu-architecture/)
