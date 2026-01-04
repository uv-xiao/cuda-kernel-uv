#!/usr/bin/env python3

import subprocess
import sys
import os
import re

def run_command(cmd):
    """Run a shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"

def compile_cuda(source_file, output_file):
    """Compile CUDA source file"""
    print(f"Compiling {source_file}...")
    cmd = f"nvcc -O3 -arch=sm_70 {source_file} -o {output_file}"
    returncode, stdout, stderr = run_command(cmd)

    if returncode != 0:
        print(f"Compilation failed:")
        print(stderr)
        return False

    print("Compilation successful!\n")
    return True

def run_test(executable):
    """Run the executable and parse output"""
    print(f"Running {executable}...\n")
    returncode, stdout, stderr = run_command(f"./{executable}")

    if returncode != 0:
        print(f"Execution failed:")
        print(stderr)
        return None

    print(stdout)
    return stdout

def parse_results(output):
    """Parse test output to extract results"""
    results = {
        'naive': {'passed': False, 'bandwidth': 0.0},
        'shared': {'passed': False, 'bandwidth': 0.0},
        'optimized': {'passed': False, 'bandwidth': 0.0}
    }

    if not output:
        return results

    # Parse naive results
    naive_match = re.search(r'Test 1:.*?Result: (\w+).*?Bandwidth: ([\d.]+)', output, re.DOTALL)
    if naive_match:
        results['naive']['passed'] = naive_match.group(1) == 'PASSED'
        results['naive']['bandwidth'] = float(naive_match.group(2))

    # Parse shared results
    shared_match = re.search(r'Test 2:.*?Result: (\w+).*?Bandwidth: ([\d.]+)', output, re.DOTALL)
    if shared_match:
        results['shared']['passed'] = shared_match.group(1) == 'PASSED'
        results['shared']['bandwidth'] = float(shared_match.group(2))

    # Parse optimized results
    opt_match = re.search(r'Test 3:.*?Result: (\w+).*?Bandwidth: ([\d.]+)', output, re.DOTALL)
    if opt_match:
        results['optimized']['passed'] = opt_match.group(1) == 'PASSED'
        results['optimized']['bandwidth'] = float(opt_match.group(2))

    return results

def check_performance(results):
    """Check if performance meets targets"""
    targets = {
        'naive': 100.0,      # GB/s
        'shared': 300.0,     # GB/s
        'optimized': 400.0   # GB/s
    }

    print("\n=== Performance Evaluation ===\n")
    print(f"{'Kernel':<20} {'Bandwidth':<15} {'Target':<15} {'Status'}")
    print("-" * 70)

    all_passed = True
    for kernel in ['naive', 'shared', 'optimized']:
        bw = results[kernel]['bandwidth']
        target = targets[kernel]
        status = "✓ PASS" if bw >= target else "✗ FAIL"

        if bw < target:
            all_passed = False

        print(f"{kernel.capitalize():<20} {bw:>8.2f} GB/s   {target:>8.2f} GB/s   {status}")

    return all_passed

def main():
    print("=" * 70)
    print("Matrix Transpose Exercise - Test Suite")
    print("=" * 70)
    print()

    # Check if starter.cu exists
    if not os.path.exists("starter.cu"):
        print("Error: starter.cu not found!")
        print("Please ensure you're running this script from the exercise directory.")
        return 1

    # Compile starter code
    if not compile_cuda("starter.cu", "transpose_starter"):
        return 1

    # Run starter tests
    print("=" * 70)
    print("Testing Your Implementation")
    print("=" * 70)
    print()

    output = run_test("transpose_starter")
    if output is None:
        return 1

    results = parse_results(output)

    # Check correctness
    print("\n=== Correctness Evaluation ===\n")
    all_correct = True
    for kernel in ['naive', 'shared', 'optimized']:
        status = "✓ PASS" if results[kernel]['passed'] else "✗ FAIL"
        print(f"{kernel.capitalize():<20} {status}")
        if not results[kernel]['passed']:
            all_correct = False

    if not all_correct:
        print("\n⚠ Some tests failed correctness checks!")
        print("Please review your implementation.")
        return 1

    # Check performance
    perf_passed = check_performance(results)

    # Final verdict
    print("\n" + "=" * 70)
    if all_correct and perf_passed:
        print("🎉 Congratulations! All tests passed!")
        print("=" * 70)
        return 0
    elif all_correct:
        print("⚠ Correctness: PASSED, but performance targets not met.")
        print("Consider reviewing the optimization techniques.")
        print("=" * 70)
        return 0
    else:
        print("❌ Tests failed. Please review your implementation.")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(main())
