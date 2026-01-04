#!/usr/bin/env python3
"""
Test script for matrix transpose optimization exercise.
Checks correctness and performance of student implementation.
"""

import subprocess
import re
import sys
import os

class Color:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Color.BOLD}{Color.BLUE}{'='*60}{Color.RESET}")
    print(f"{Color.BOLD}{Color.BLUE}{text:^60}{Color.RESET}")
    print(f"{Color.BOLD}{Color.BLUE}{'='*60}{Color.RESET}\n")

def print_success(text):
    print(f"{Color.GREEN}✓ {text}{Color.RESET}")

def print_error(text):
    print(f"{Color.RED}✗ {text}{Color.RESET}")

def print_warning(text):
    print(f"{Color.YELLOW}⚠ {text}{Color.RESET}")

def run_command(cmd):
    """Run a shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print_error("Command timed out!")
        return -1, "", "Timeout"

def build_program(source_file, output_name):
    """Build CUDA program"""
    print(f"Building {source_file}...")
    cmd = f"nvcc -O3 -o {output_name} {source_file}"
    returncode, stdout, stderr = run_command(cmd)

    if returncode != 0:
        print_error(f"Build failed!")
        print(stderr)
        return False

    print_success(f"Built {output_name}")
    return True

def extract_bandwidth(output):
    """Extract bandwidth from program output"""
    # Look for lines like "Bandwidth: 123.45 GB/s"
    match = re.search(r'Bandwidth:\s*([\d.]+)\s*GB/s', output)
    if match:
        return float(match.group(1))
    return None

def extract_utilization(output):
    """Extract bandwidth utilization percentage"""
    match = re.search(r'Utilization:\s*([\d.]+)%', output)
    if match:
        return float(match.group(1))
    return None

def check_correctness(output):
    """Check if correctness test passed"""
    return "Correctness: PASSED" in output

def profile_bank_conflicts(program):
    """Use ncu to check for shared memory bank conflicts"""
    print("Profiling for bank conflicts...")
    cmd = f"ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum --csv {program} 2>/dev/null"
    returncode, stdout, stderr = run_command(cmd)

    if returncode != 0:
        print_warning("Could not run ncu (Nsight Compute may not be available)")
        return None

    # Parse CSV output to extract bank conflicts
    # Look for numeric value in the CSV
    numbers = re.findall(r'\d+\.?\d*', stdout)
    if numbers:
        conflicts = float(numbers[-1])  # Last number is usually the metric value
        return conflicts

    return None

def test_solution():
    """Test the student's solution"""
    print_header("Testing Matrix Transpose Optimization")

    # Check if solution file exists
    if not os.path.exists("solution.cu"):
        print_error("solution.cu not found!")
        print("Please implement your solution in solution.cu")
        return False

    # Build solution
    if not build_program("solution.cu", "transpose_solution"):
        return False

    # Run solution
    print("\nRunning solution...")
    returncode, stdout, stderr = run_command("./transpose_solution")

    if returncode != 0:
        print_error("Program crashed!")
        print(stderr)
        return False

    print(stdout)

    # Check correctness
    print_header("Correctness Test")
    if check_correctness(stdout):
        print_success("Correctness: PASSED")
    else:
        print_error("Correctness: FAILED")
        return False

    # Check performance
    print_header("Performance Test")

    bandwidth = extract_bandwidth(stdout)
    utilization = extract_utilization(stdout)

    if bandwidth is None or utilization is None:
        print_error("Could not extract performance metrics")
        return False

    print(f"Bandwidth: {bandwidth:.2f} GB/s")
    print(f"Utilization: {utilization:.1f}%")

    # Performance threshold: should achieve >80% utilization
    if utilization >= 80.0:
        print_success(f"Performance: PASSED (utilization {utilization:.1f}% >= 80%)")
        perf_pass = True
    else:
        print_error(f"Performance: FAILED (utilization {utilization:.1f}% < 80%)")
        print("Hint: Make sure you've added padding to eliminate bank conflicts")
        perf_pass = False

    # Check bank conflicts
    print_header("Bank Conflict Test")

    conflicts = profile_bank_conflicts("./transpose_solution")

    if conflicts is not None:
        print(f"Shared memory bank conflicts: {int(conflicts)}")
        if conflicts == 0:
            print_success("Bank conflicts: PASSED (zero conflicts)")
            bank_pass = True
        else:
            print_error(f"Bank conflicts: FAILED ({int(conflicts)} conflicts detected)")
            print("Hint: Add +1 padding to shared memory array: __shared__ float tile[32][33];")
            bank_pass = False
    else:
        print_warning("Could not profile bank conflicts (ncu not available)")
        print("To manually check: ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum ./transpose_solution")
        bank_pass = True  # Don't fail if profiling tools not available

    # Overall result
    print_header("Overall Result")

    if perf_pass and bank_pass:
        print_success("All tests PASSED!")
        print("\nCongratulations! Your implementation:")
        print(f"  ✓ Is correct")
        print(f"  ✓ Achieves {utilization:.1f}% memory bandwidth utilization")
        if conflicts is not None:
            print(f"  ✓ Has zero shared memory bank conflicts")
        print("\nYou've successfully optimized matrix transpose!")
        return True
    else:
        print_error("Some tests FAILED")
        print("\nYour implementation needs improvement:")
        if not perf_pass:
            print(f"  ✗ Performance is below target ({utilization:.1f}% < 80%)")
        if not bank_pass:
            print(f"  ✗ Has {int(conflicts)} bank conflicts (should be 0)")
        print("\nReview the hints in problem.md and try again.")
        return False

def compare_with_starter():
    """Compare solution with starter code"""
    print_header("Comparison with Starter Code")

    if not os.path.exists("starter.cu"):
        print_warning("starter.cu not found, skipping comparison")
        return

    if not build_program("starter.cu", "transpose_starter"):
        return

    # Run starter
    returncode, starter_output, _ = run_command("./transpose_starter")
    if returncode != 0:
        print_warning("Could not run starter code")
        return

    # Run solution
    returncode, solution_output, _ = run_command("./transpose_solution")
    if returncode != 0:
        return

    # Extract metrics
    starter_bw = extract_bandwidth(starter_output)
    solution_bw = extract_bandwidth(solution_output)

    if starter_bw and solution_bw:
        speedup = solution_bw / starter_bw
        print(f"Starter bandwidth: {starter_bw:.2f} GB/s")
        print(f"Solution bandwidth: {solution_bw:.2f} GB/s")
        print(f"Speedup: {speedup:.2f}x")

        if speedup >= 3.0:
            print_success(f"Excellent speedup! ({speedup:.2f}x faster)")
        elif speedup >= 2.0:
            print_success(f"Good speedup ({speedup:.2f}x faster)")
        else:
            print_warning(f"Modest speedup ({speedup:.2f}x). Expected >3x improvement.")

def main():
    """Main test function"""
    success = test_solution()

    # Optional: compare with starter
    compare_with_starter()

    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
