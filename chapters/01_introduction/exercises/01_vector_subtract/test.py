#!/usr/bin/env python3
"""
Test script for Vector Subtraction exercise

This script compiles and tests the vector subtraction implementation
with various input sizes to ensure correctness.
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

def run_test(executable, size, test_name):
    """Run the executable with a given vector size"""
    cmd = f"./{executable} {size}"
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
    print("CUDA Vector Subtraction - Test Suite")
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

    # Step 2: Run tests with different sizes
    test_cases = [
        (1000, "Small vector (1K elements)"),
        (100000, "Medium vector (100K elements)"),
        (1000000, "Large vector (1M elements)"),
    ]

    all_passed = True
    results = []

    for size, name in test_cases:
        passed = run_test(executable, size, name)
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
        print("\nCongratulations! Your implementation is correct.")
        print("You can now move on to the next exercise.")
        sys.exit(0)
    else:
        print("RESULT: SOME TESTS FAILED ✗")
        print("=" * 70)
        print("\nPlease review your implementation and try again.")
        print("\nCommon issues to check:")
        print("  1. Did you implement the kernel correctly?")
        print("  2. Is the subtraction operation correct (a[i] - b[i])?")
        print("  3. Did you include bounds checking (if i < n)?")
        print("  4. Are memory transfers in the correct direction?")
        print("  5. Did you copy the right variables?")
        sys.exit(1)

if __name__ == "__main__":
    main()
