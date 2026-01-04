#!/bin/bash

# Sample Profiling Commands for CUDA Kernel Tutorial
# Chapter 03: Profiling & Optimization

# This script contains ready-to-run profiling commands for analyzing CUDA kernels
# Uncomment and run the sections you need

# ==============================================================================
# SETUP
# ==============================================================================

# Set these variables to match your environment
PROGRAM="./matmul_naive"  # Change to your program
OUTPUT_DIR="./profile_results"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ==============================================================================
# BASIC PROFILING
# ==============================================================================

echo "=== Basic Profiling Commands ==="

# Quick overview of all kernels
# ncu $PROGRAM

# Basic metrics (fast, good starting point)
# ncu --set basic $PROGRAM

# Full metrics (comprehensive, slower)
# ncu --set full $PROGRAM

# ==============================================================================
# SAVE REPORTS FOR LATER ANALYSIS
# ==============================================================================

echo "=== Saving Reports ==="

# Save basic profile
# ncu --set basic -o $OUTPUT_DIR/basic_profile $PROGRAM

# Save full profile
# ncu --set full -o $OUTPUT_DIR/full_profile $PROGRAM

# View saved report in GUI
# ncu-ui $OUTPUT_DIR/full_profile.ncu-rep

# View saved report in terminal
# ncu --import $OUTPUT_DIR/full_profile.ncu-rep

# ==============================================================================
# MEMORY ANALYSIS
# ==============================================================================

echo "=== Memory Analysis Commands ==="

# Detailed memory analysis
# ncu --section MemoryWorkloadAnalysis $PROGRAM

# Key memory metrics
# ncu --metrics \
# dram__throughput.avg.pct_of_peak_sustained_elapsed,\
# smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
# smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,\
# l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
# l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum \
# $PROGRAM

# Check for bank conflicts in shared memory
# ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum $PROGRAM

# Cache hit rates
# ncu --metrics \
# l1tex__t_sector_hit_rate.pct,\
# lts__t_sector_hit_rate.pct \
# $PROGRAM

# ==============================================================================
# COMPUTE ANALYSIS
# ==============================================================================

echo "=== Compute Analysis Commands ==="

# Detailed compute analysis
# ncu --section ComputeWorkloadAnalysis $PROGRAM

# SM throughput and utilization
# ncu --metrics \
# sm__throughput.avg.pct_of_peak_sustained_elapsed,\
# smsp__average_inst_executed_per_cycle_active.ratio \
# $PROGRAM

# Count floating-point operations
# ncu --metrics \
# sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
# sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
# sm__sass_thread_inst_executed_op_ffma_pred_on.sum \
# $PROGRAM

# ==============================================================================
# OCCUPANCY ANALYSIS
# ==============================================================================

echo "=== Occupancy Analysis Commands ==="

# Detailed occupancy info
# ncu --section Occupancy $PROGRAM

# Launch configuration details
# ncu --section LaunchStats $PROGRAM

# Achieved occupancy
# ncu --metrics smsp__warps_active.avg.pct_of_peak_sustained_active $PROGRAM

# ==============================================================================
# WARP ANALYSIS
# ==============================================================================

echo "=== Warp Analysis Commands ==="

# Warp state and stall reasons
# ncu --section WarpStateStats $PROGRAM

# Warp execution efficiency (divergence)
# ncu --metrics smsp__sass_average_branch_targets_threads_uniform.pct $PROGRAM

# Detailed stall reasons
# ncu --metrics \
# smsp__average_warps_issue_stalled_barrier.pct,\
# smsp__average_warps_issue_stalled_long_scoreboard.pct,\
# smsp__average_warps_issue_stalled_short_scoreboard.pct,\
# smsp__average_warps_issue_stalled_not_selected.pct \
# $PROGRAM

# ==============================================================================
# ROOFLINE ANALYSIS
# ==============================================================================

echo "=== Roofline Analysis Commands ==="

# Generate roofline model data
# ncu --set roofline -o $OUTPUT_DIR/roofline $PROGRAM

# View in GUI (shows performance vs. arithmetic intensity)
# ncu-ui $OUTPUT_DIR/roofline.ncu-rep

# ==============================================================================
# KERNEL-SPECIFIC PROFILING
# ==============================================================================

echo "=== Kernel-Specific Profiling Commands ==="

# Profile specific kernel by name
# ncu --kernel-name matmul_naive $PROGRAM

# Profile by kernel ID (0-indexed)
# ncu --kernel-id 0 $PROGRAM

# Profile specific launch (if kernel is called multiple times)
# ncu --launch-skip 5 --launch-count 1 $PROGRAM

# Profile first N kernel launches
# ncu --launch-count 3 $PROGRAM

# ==============================================================================
# COMPARISON
# ==============================================================================

echo "=== Comparison Commands ==="

# Profile multiple versions
# ncu --set full -o $OUTPUT_DIR/version1 ./matmul_v1
# ncu --set full -o $OUTPUT_DIR/version2 ./matmul_v2

# Compare in GUI
# ncu-ui $OUTPUT_DIR/version1.ncu-rep $OUTPUT_DIR/version2.ncu-rep

# Compare in terminal
# ncu --import $OUTPUT_DIR/version1.ncu-rep $OUTPUT_DIR/version2.ncu-rep

# ==============================================================================
# SOURCE CODE CORRELATION
# ==============================================================================

echo "=== Source Code Correlation Commands ==="

# Compile with line info for source correlation
# nvcc -lineinfo -o program program.cu

# Profile with source info
# ncu --set full --source-folders $(pwd) -o $OUTPUT_DIR/with_source $PROGRAM

# View in GUI to see hotspots in source code
# ncu-ui $OUTPUT_DIR/with_source.ncu-rep

# ==============================================================================
# NSIGHT SYSTEMS (SYSTEM-LEVEL PROFILING)
# ==============================================================================

echo "=== Nsight Systems Commands ==="

# Basic timeline profile
# nsys profile -o $OUTPUT_DIR/timeline $PROGRAM

# With statistics summary
# nsys profile --stats=true -o $OUTPUT_DIR/timeline $PROGRAM

# Specify trace types
# nsys profile --trace=cuda,nvtx --stats=true -o $OUTPUT_DIR/timeline $PROGRAM

# Limited duration (useful for long-running programs)
# nsys profile --duration=10 -o $OUTPUT_DIR/timeline $PROGRAM

# CPU sampling
# nsys profile --sample=cpu --trace=cuda -o $OUTPUT_DIR/timeline $PROGRAM

# View timeline in GUI
# nsys-ui $OUTPUT_DIR/timeline.nsys-rep

# View statistics in terminal
# nsys stats $OUTPUT_DIR/timeline.nsys-rep

# Specific statistics reports
# nsys stats --report cuda_kern_sum $OUTPUT_DIR/timeline.nsys-rep
# nsys stats --report cuda_mem_sum $OUTPUT_DIR/timeline.nsys-rep
# nsys stats --report cuda_api_sum $OUTPUT_DIR/timeline.nsys-rep

# ==============================================================================
# BATCH PROFILING SCRIPT
# ==============================================================================

echo "=== Batch Profiling Example ==="

# Example: Profile all matrix multiplication versions
profile_all_versions() {
    local versions=("naive" "coalesced" "tiled" "optimized")

    for version in "${versions[@]}"; do
        echo "Profiling $version version..."
        ncu --set full -o "$OUTPUT_DIR/matmul_$version" "./matmul_$version"
    done

    echo "Comparing all versions..."
    ncu --import "$OUTPUT_DIR"/matmul_*.ncu-rep
}

# Uncomment to run:
# profile_all_versions

# ==============================================================================
# CUSTOM METRIC SETS
# ==============================================================================

echo "=== Custom Metric Sets ==="

# Memory-focused metrics
MEMORY_METRICS="dram__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,\
l1tex__t_sector_hit_rate.pct,\
lts__t_sector_hit_rate.pct"

# ncu --metrics $MEMORY_METRICS $PROGRAM

# Compute-focused metrics
COMPUTE_METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__average_inst_executed_per_cycle_active.ratio,\
sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum"

# ncu --metrics $COMPUTE_METRICS $PROGRAM

# Occupancy-focused metrics
OCCUPANCY_METRICS="smsp__warps_active.avg.pct_of_peak_sustained_active,\
sm__maximum_warps_per_active_cycle_pct"

# ncu --metrics $OCCUPANCY_METRICS $PROGRAM

# All-in-one quick check
QUICK_METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__warps_active.avg.pct_of_peak_sustained_active"

# ncu --metrics $QUICK_METRICS $PROGRAM

# ==============================================================================
# PERFORMANCE REGRESSION TESTING
# ==============================================================================

echo "=== Performance Regression Testing ==="

regression_test() {
    local baseline="$1"
    local current="$2"

    echo "Running baseline..."
    ncu --set full -o "$OUTPUT_DIR/baseline" "$baseline"

    echo "Running current version..."
    ncu --set full -o "$OUTPUT_DIR/current" "$current"

    echo "Comparing..."
    ncu --import "$OUTPUT_DIR/baseline.ncu-rep" "$OUTPUT_DIR/current.ncu-rep"
}

# Example usage:
# regression_test ./matmul_v1 ./matmul_v2

# ==============================================================================
# AUTOMATED PERFORMANCE EXTRACTION
# ==============================================================================

echo "=== Automated Performance Extraction ==="

# Extract specific metrics to CSV (for scripting)
extract_metrics() {
    local program="$1"
    local output="$2"

    ncu --csv --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed \
    "$program" > "$output"
}

# Example usage:
# extract_metrics ./matmul_naive metrics.csv

# ==============================================================================
# TROUBLESHOOTING
# ==============================================================================

echo "=== Troubleshooting Commands ==="

# If you get permission errors, try:
# sudo ncu $PROGRAM
# Or permanently disable profiling restrictions (requires reboot):
# echo 'options nvidia "NVreg_RestrictProfilingToAdminUsers=0"' | sudo tee /etc/modprobe.d/nvidia-prof.conf

# If profiling is very slow:
# - Use --set basic instead of --set full
# - Use --replay-mode none (less accurate)
# - Profile specific kernels only with --kernel-name

# If kernel is replayed many times:
# - This is normal for --set full
# - Use --set basic or select specific metrics
# - Use --replay-mode application or kernel

# ==============================================================================
# LIST AVAILABLE METRICS
# ==============================================================================

echo "=== Listing Available Metrics ==="

# List all available metrics
# ncu --query-metrics

# List metrics with descriptions
# ncu --query-metrics-brief

# Search for specific metrics (example: memory-related)
# ncu --query-metrics | grep -i memory

# ==============================================================================
# USEFUL ALIASES
# ==============================================================================

echo "=== Useful Aliases (add to ~/.bashrc) ==="

# alias ncu-quick='ncu --set basic'
# alias ncu-full='ncu --set full'
# alias ncu-mem='ncu --section MemoryWorkloadAnalysis'
# alias ncu-compute='ncu --section ComputeWorkloadAnalysis'
# alias ncu-occ='ncu --section Occupancy'
# alias ncu-save='ncu --set full -o profile'

echo ""
echo "To use these commands:"
echo "1. Set PROGRAM variable to your executable"
echo "2. Uncomment the desired command(s)"
echo "3. Run this script or copy commands to terminal"
echo ""
echo "For more info, see:"
echo "  - profiling_guide.md"
echo "  - metrics_explained.md"
echo "  - https://docs.nvidia.com/nsight-compute/"
