#!/usr/bin/env python3
"""
Test script for SAXPY exercise

This script compiles and tests the SAXPY implementation with various
alpha values and input sizes to ensure correctness across different cases.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print(f"  ERROR: Command timed out")
        return False, "", "Timeout"
    except Exception as e:
        print(f"  ERROR: {e}")
        return False, "", str(e)

def check_compilation(source_file):
    """Compile the CUDA source file"""
    output_file = source_file.replace('.cu', '')
    compile_cmd = f"nvcc -o {output_file} {source_file}"

    success, stdout, stderr = run_command(compile_cmd, f"Compiling {source_file}")

    if success:
        print(f"  ✓ Compilation successful")
        return True, output_file
    else:
        print(f"  ✗ Compilation failed")
        print(f"  Error: {stderr}")
        return False, None

def run_test(executable, size, alpha, test_name):
    """Run the executable with given vector size and alpha"""
    cmd = f"./{executable} {size} {alpha}"
    success, stdout, stderr = run_command(cmd, f"Running {test_name}")

    if not success:
        print(f"  ✗ {test_name} FAILED (execution error)")
        print(f"  Error: {stderr}")
        return False

    # Check for SUCCESS in output
    if "SUCCESS" in stdout:
        print(f"  ✓ {test_name} PASSED")
        return True
    else:
        print(f"  ✗ {test_name} FAILED (incorrect results)")
        if "FAILURE" in stdout or "Mismatch" in stdout:
            # Print relevant error lines
            for line in stdout.split('\n'):
                if "Mismatch" in line or "FAILURE" in line:
                    print(f"    {line}")
        return False

def main():
    print("=" * 70)
    print("CUDA SAXPY (Y = α·X + Y) - Test Suite")
    print("=" * 70)

    # Determine which file to test
    if len(sys.argv) > 1:
        source_file = sys.argv[1]
    else:
        # Try solution.cu by default, fall back to starter.cu
        if os.path.exists("solution.cu"):
            source_file = "solution.cu"
        elif os.path.exists("starter.cu"):
            source_file = "starter.cu"
        else:
            print("Error: No source file found (solution.cu or starter.cu)")
            sys.exit(1)

    print(f"\nTesting: {source_file}")

    # Step 1: Compilation
    success, executable = check_compilation(source_file)
    if not success:
        print("\n" + "=" * 70)
        print("RESULT: COMPILATION FAILED")
        print("=" * 70)
        sys.exit(1)

    # Step 2: Run tests with different sizes and alpha values
    test_cases = [
        (1000, 2.0, "Small vector, alpha=2.0"),
        (100000, 1.0, "Medium vector, alpha=1.0 (simple add)"),
        (1000000, -1.0, "Large vector, alpha=-1.0 (subtract)"),
        (10000, 0.0, "Edge case: alpha=0.0 (Y unchanged)"),
        (50000, 0.5, "Fractional alpha=0.5"),
        (5000, -2.5, "Negative fractional alpha=-2.5"),
    ]

    all_passed = True
    results = []

    for size, alpha, name in test_cases:
        passed = run_test(executable, size, alpha, name)
        results.append((name, passed))
        all_passed = all_passed and passed

    # Step 3: Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {name}")

    print("\n" + "=" * 70)
    if all_passed:
        print("RESULT: ALL TESTS PASSED ✓")
        print("=" * 70)
        print("\nExcellent! Your SAXPY implementation is correct.")
        print("\nKey achievements:")
        print("  ✓ Scalar parameter handling")
        print("  ✓ In-place operation (Y = α·X + Y)")
        print("  ✓ Edge case handling (α=0)")
        print("  ✓ Negative and fractional alpha values")
        print("\nYou've completed Chapter 01 exercises!")
        print("Ready to move on to Chapter 02: Memory Optimization")
        sys.exit(0)
    else:
        print("RESULT: SOME TESTS FAILED ✗")
        print("=" * 70)
        print("\nPlease review your implementation and try again.")
        print("\nCommon issues to check:")
        print("  1. Did you implement the kernel correctly?")
        print("  2. Is the operation y[i] = alpha * x[i] + y[i]?")
        print("  3. Are you reading the ORIGINAL y[i] value?")
        print("  4. Did you include bounds checking (if i < n)?")
        print("  5. Did you save a copy of Y before GPU modifies it?")
        print("  6. Are you using the original Y for CPU verification?")
        print("\nFor alpha=0 test:")
        print("  - Y should remain unchanged")
        print("  - Result should equal original Y values")
        sys.exit(1)

if __name__ == "__main__":
    main()
