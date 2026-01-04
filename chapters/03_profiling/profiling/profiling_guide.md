# CUDA Profiling Guide: Nsight Compute & Nsight Systems

## Introduction

Profiling is essential for understanding and optimizing CUDA applications. NVIDIA provides two main profiling tools:

- **Nsight Compute (ncu)**: Detailed kernel-level profiling
- **Nsight Systems (nsys)**: System-wide timeline and trace analysis

This guide teaches you how to use both tools effectively.

## Nsight Compute (ncu)

### What is Nsight Compute?

Nsight Compute is an interactive kernel profiler for CUDA applications. It provides:
- Detailed performance metrics for each kernel
- Memory access pattern analysis
- Roofline analysis
- Optimization suggestions
- Source-level correlation

### Basic Usage

#### Quick Profile

Get a summary of kernel performance:

```bash
ncu ./your_program

# With arguments
ncu ./your_program arg1 arg2
```

**Output**:
- Kernel name and launch configuration
- Duration and throughput
- High-level warnings and recommendations

#### Metric Sets

Nsight Compute organizes metrics into sets:

```bash
# Basic metrics (fastest)
ncu --set basic ./your_program

# Full metrics (comprehensive, slower)
ncu --set full ./your_program

# Specific sections
ncu --section MemoryWorkloadAnalysis ./your_program
ncu --section ComputeWorkloadAnalysis ./your_program
ncu --section LaunchStats ./your_program
ncu --section Occupancy ./your_program
```

**Available sections**:
- `LaunchStats`: Grid/block dimensions, registers, shared memory
- `Occupancy`: Theoretical and achieved occupancy
- `MemoryWorkloadAnalysis`: Memory access patterns and efficiency
- `ComputeWorkloadAnalysis`: Instruction mix and throughput
- `SpeedOfLight`: Percentage of peak performance
- `WarpStateStats`: Warp stall reasons
- `Scheduler`: Warp scheduling efficiency

#### Selecting Specific Metrics

Profile only the metrics you need:

```bash
# Single metric
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ./your_program

# Multiple metrics (comma-separated)
ncu --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct \
./your_program
```

**Common metrics**:
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` - SM compute utilization
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` - Memory bandwidth utilization
- `smsp__warps_active.avg.pct_of_peak_sustained_active` - Occupancy
- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` - Global load transactions
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` - Shared memory bank conflicts

See `ncu --query-metrics` for a full list.

#### Saving Reports

Save profiling results for later analysis:

```bash
# Save to file
ncu --set full -o report ./your_program

# Creates: report.ncu-rep

# View later in GUI
ncu-ui report.ncu-rep

# View in terminal
ncu --import report.ncu-rep
```

#### Profiling Specific Kernels

For applications with multiple kernels:

```bash
# Profile only kernels matching a pattern
ncu --kernel-name matmul ./your_program

# Profile kernel by index (0-based)
ncu --kernel-id 2 ./your_program

# Profile specific kernel launch (kernel may be launched multiple times)
ncu --launch-skip 5 --launch-count 1 ./your_program
```

### Advanced Usage

#### Comparing Kernels

Compare performance across different versions:

```bash
# Profile multiple versions
ncu --set full -o naive ./matmul_naive
ncu --set full -o optimized ./matmul_optimized

# Compare in GUI
ncu-ui naive.ncu-rep optimized.ncu-rep

# Compare in terminal
ncu --import naive.ncu-rep optimized.ncu-rep
```

#### Roofline Analysis

Visualize performance vs. arithmetic intensity:

```bash
# Generate roofline data
ncu --set roofline -o roofline ./your_program

# View in GUI (shows where you are vs. theoretical limits)
ncu-ui roofline.ncu-rep
```

**Interpreting the roofline**:
- **On compute roofline**: Compute-bound, optimize ALU usage
- **On memory roofline**: Memory-bound, optimize bandwidth
- **Below both**: Multiple bottlenecks or inefficiencies

#### Source Correlation

See which source lines consume time:

```bash
# Compile with line info
nvcc -lineinfo -o program program.cu

# Profile with source correlation
ncu --set full --source-folders /path/to/source -o report ./program

# View in GUI to see hotspots highlighted in source
ncu-ui report.ncu-rep
```

#### Replay Mode

For accurate metrics, ncu may need to replay kernels:

```bash
# Application replay (reruns entire program)
ncu --replay-mode application ./your_program

# Kernel replay (reruns only the profiled kernel)
ncu --replay-mode kernel ./your_program

# Disable replay (faster but less accurate)
ncu --replay-mode none ./your_program
```

**Note**: Replay mode affects runtime. Use `application` for most accurate results.

### GUI Workflow

The Nsight Compute GUI provides interactive analysis:

```bash
# Open GUI with report
ncu-ui report.ncu-rep
```

**Key features**:
1. **Details** page: All metrics organized by section
2. **Source** page: Source code with metrics overlay
3. **Speed of Light**: Quick overview of utilization
4. **Memory Workload Analysis**: Charts showing memory patterns
5. **Baseline comparison**: Compare with other runs

**Workflow**:
1. Start with "Speed of Light" to identify bottleneck
2. Drill into relevant section (Memory/Compute)
3. Check "Details" for specific metrics
4. Use "Source" to find hotspots
5. Iterate after optimization

## Nsight Systems (nsys)

### What is Nsight Systems?

Nsight Systems provides system-wide performance analysis:
- Timeline visualization of CPU and GPU activity
- CUDA API call traces
- Memory transfers
- Kernel execution timeline
- Multi-GPU synchronization
- OS runtime libraries (pthreads, etc.)

Use it to find:
- CPU-GPU synchronization issues
- Memory transfer bottlenecks
- Kernel launch overhead
- Load imbalance across GPUs

### Basic Usage

#### Generate Timeline

```bash
# Basic profile
nsys profile -o timeline ./your_program

# With statistics
nsys profile --stats=true -o timeline ./your_program

# Specify what to trace
nsys profile --trace=cuda,nvtx --stats=true ./your_program
```

**Output**: `timeline.nsys-rep` and `timeline.sqlite` files

#### View Results

```bash
# Open in GUI (recommended)
nsys-ui timeline.nsys-rep

# View statistics in terminal
nsys stats timeline.nsys-rep
```

#### Trace Options

Control what gets profiled:

```bash
# Trace CUDA API and kernel execution
nsys profile --trace=cuda ./your_program

# Add NVTX markers (see below)
nsys profile --trace=cuda,nvtx ./your_program

# Add OpenACC, OpenMP
nsys profile --trace=cuda,openacc,openmp ./your_program

# Trace everything (slower)
nsys profile --trace=cuda,nvtx,osrt,opengl ./your_program
```

#### Sampling Options

```bash
# Sample CPU usage
nsys profile --sample=cpu ./your_program

# Sample at different frequency (Hz)
nsys profile --sample=cpu --sampling-frequency=1000 ./your_program
```

### NVTX Markers

Annotate your code for better profiling:

```cpp
#include <nvToolsExt.h>

void my_function() {
    nvtxRangePush("My Function");

    // Your code here

    nvtxRangePop();
}

// Named ranges with color
void another_function() {
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = 0xFF00FF00;  // Green
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = "My Custom Range";

    nvtxRangePushEx(&eventAttrib);
    // Code
    nvtxRangePop();
}
```

**Compile with**:
```bash
nvcc -lnvToolsExt -o program program.cu
```

**Profile**:
```bash
nsys profile --trace=cuda,nvtx ./program
```

Markers appear in the timeline, making it easy to identify regions.

### GUI Analysis

The Nsight Systems GUI shows:

1. **Timeline View**:
   - CPU threads
   - CUDA streams
   - Kernel executions
   - Memory transfers
   - NVTX ranges

2. **Analysis View**:
   - Kernel statistics (count, duration, etc.)
   - Memory operation statistics
   - CUDA API call statistics

**Workflow**:
1. Load report in GUI: `nsys-ui timeline.nsys-rep`
2. Examine timeline for gaps and inefficiencies
3. Look for:
   - GPU idle periods (CPU bottleneck)
   - CPU idle periods (waiting for GPU)
   - Overlapping memory transfers and compute
   - Kernel launch overhead
4. Use statistics to identify expensive operations
5. Zoom in to problematic regions

### Common Patterns to Look For

#### Pattern 1: GPU Idle Time

**Symptom**: Gaps between kernel executions

**Cause**: CPU can't submit work fast enough

**Solution**: Reduce CPU overhead, use CUDA graphs, batch operations

#### Pattern 2: CPU Waiting

**Symptom**: CPU thread blocked on `cudaDeviceSynchronize()`

**Cause**: Insufficient async work, unnecessary synchronization

**Solution**: Use streams, async operations, remove unnecessary syncs

#### Pattern 3: Memory Transfer Bottleneck

**Symptom**: Long memory copy operations, GPU idle during transfers

**Cause**: Transfers not overlapped with compute

**Solution**: Use pinned memory, async copies, multiple streams

#### Pattern 4: Small Kernels

**Symptom**: Many short kernel executions

**Cause**: Launch overhead dominates runtime

**Solution**: Kernel fusion, increase work per kernel, use CUDA graphs

## Profiling Workflow

### Step-by-Step Process

#### 1. System-Level Overview (nsys)

Start with Nsight Systems to understand overall behavior:

```bash
nsys profile --trace=cuda,nvtx --stats=true -o overview ./program
nsys-ui overview.nsys-rep
```

**Look for**:
- Where is time spent? (CPU vs. GPU)
- Are there obvious bottlenecks?
- Is the GPU well-utilized?

#### 2. Kernel-Level Analysis (ncu)

Identify slow kernels and analyze them:

```bash
# Quick scan
ncu --set basic ./program

# Deep dive on specific kernel
ncu --set full --kernel-name slow_kernel -o slow_kernel ./program
ncu-ui slow_kernel.ncu-rep
```

**Look for**:
- Low SM or memory throughput
- Memory access inefficiencies
- Low occupancy
- Warp stalls

#### 3. Optimize

Based on findings, apply targeted optimizations:
- **Memory-bound**: Coalescing, tiling, caching
- **Compute-bound**: ILP, loop unrolling, reduce operations
- **Latency-bound**: Increase occupancy
- **Launch-bound**: Kernel fusion, CUDA graphs

#### 4. Measure and Iterate

Re-profile to verify improvements:

```bash
# Compare before and after
ncu --set full -o before ./program_v1
ncu --set full -o after ./program_v2
ncu --import before.ncu-rep after.ncu-rep
```

Repeat until performance goals are met.

## Quick Reference

### Nsight Compute Cheat Sheet

```bash
# Quick profile
ncu ./program

# Full metrics
ncu --set full ./program

# Save report
ncu --set full -o report ./program

# Specific kernel
ncu --kernel-name matmul ./program

# Key metrics only
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./program

# Roofline
ncu --set roofline -o roofline ./program

# Compare
ncu --import v1.ncu-rep v2.ncu-rep

# GUI
ncu-ui report.ncu-rep
```

### Nsight Systems Cheat Sheet

```bash
# Basic profile
nsys profile -o timeline ./program

# With stats
nsys profile --stats=true -o timeline ./program

# CUDA + NVTX
nsys profile --trace=cuda,nvtx --stats=true ./program

# View timeline
nsys-ui timeline.nsys-rep

# View stats
nsys stats timeline.nsys-rep
```

## Best Practices

### Do's

1. **Profile early and often**: Don't wait until the end
2. **Start broad, then narrow**: System-level first, then kernel-level
3. **Use appropriate tools**: nsys for system, ncu for kernels
4. **Save reports**: Compare before/after optimizations
5. **Profile realistic workloads**: Representative data sizes and inputs
6. **Check multiple metrics**: One metric can be misleading
7. **Use NVTX markers**: Annotate your code for clarity

### Don'ts

1. **Don't profile debug builds**: Always use release builds with optimizations
2. **Don't profile on a busy system**: Other processes affect results
3. **Don't over-optimize**: Know when to stop (diminishing returns)
4. **Don't trust one run**: Variation is normal, run multiple times
5. **Don't ignore the low-hanging fruit**: Fix obvious issues first
6. **Don't profile and develop simultaneously**: Profile, then optimize, then profile again

## Troubleshooting

### ncu: Permission denied

**Problem**: Unable to profile due to permissions

**Solution**:
```bash
# Run as root (not recommended for regular use)
sudo ncu ./program

# Or, adjust permissions (permanent, more secure):
echo 'options nvidia "NVreg_RestrictProfilingToAdminUsers=0"' | sudo tee /etc/modprobe.d/nvidia-prof.conf
# Reboot required
```

### ncu: Kernel replays many times

**Problem**: Profiling is very slow

**Explanation**: ncu replays kernels to collect different metrics

**Solution**:
```bash
# Use --set basic for faster profiling
ncu --set basic ./program

# Or, select only needed metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ./program

# Or, disable replay (less accurate)
ncu --replay-mode none ./program
```

### nsys: Report file is huge

**Problem**: .nsys-rep file is multiple GB

**Solution**:
```bash
# Limit trace duration
nsys profile --duration=10 ./program

# Reduce sampling rate
nsys profile --sampling-frequency=100 ./program

# Trace less
nsys profile --trace=cuda ./program  # Don't trace everything
```

### Can't open GUI (remote server)

**Problem**: No X11 forwarding or no GUI available

**Solution**:
```bash
# For ncu: Use CLI mode
ncu --import report.ncu-rep

# For nsys: Export to CSV
nsys stats --report cuda_kern_sum report.nsys-rep

# Or, copy files to local machine and open there
scp remote:report.ncu-rep .
ncu-ui report.ncu-rep
```

## Resources

### Official Documentation

- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [CUDA Profiler User's Guide](https://docs.nvidia.com/cuda/profiler-users-guide/)
- [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)

### Video Tutorials

- [Nsight Compute Tutorial](https://www.youtube.com/watch?v=GvQj79o5kkw)
- [Nsight Systems Tutorial](https://www.youtube.com/watch?v=2FNDxPfVEXU)
- [GTC Profiling Talks](https://www.nvidia.com/en-us/on-demand/search/?facet.mimetype[]=event%20session&page=1&q=nsight)

### Cheat Sheets

- [Nsight Compute Metrics Reference](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference)
- [CUDA Optimization Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

---

**Next Steps**: Review [metrics_explained.md](./metrics_explained.md) for in-depth explanation of key metrics, or try the [sample_commands.sh](./sample_commands.sh) for ready-to-run profiling commands.
