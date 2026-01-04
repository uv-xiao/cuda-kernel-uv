#!/usr/bin/env python3
"""Test script for batched GEMM exercise"""

import subprocess
import sys

def run_test(executable):
    """Run the executable and check output"""
    try:
        result = subprocess.run(
            [executable],
            capture_output=True,
            text=True,
            timeout=60
        )

        print(result.stdout)

        if result.returncode != 0:
            print(f"Error running {executable}:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return False

        # Check for verification success
        if "✓" in result.stdout and "✗" not in result.stdout:
            print(f"\n✓ {executable} passed all tests!")
            return True
        else:
            print(f"\n✗ {executable} failed verification!")
            return False

    except subprocess.TimeoutExpired:
        print(f"Timeout running {executable}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"Executable not found: {executable}", file=sys.stderr)
        print("Make sure to build first: cmake .. && make", file=sys.stderr)
        return False

def main():
    print("=" * 70)
    print("Batched GEMM Test Suite")
    print("=" * 70)

    # Test starter (may fail)
    print("\nTesting starter code...")
    starter_passed = run_test("./batched_gemm_starter")

    # Test solution (should pass)
    print("\nTesting solution...")
    solution_passed = run_test("./batched_gemm_solution")

    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"Starter: {'PASS' if starter_passed else 'FAIL (expected if incomplete)'}")
    print(f"Solution: {'PASS' if solution_passed else 'FAIL'}")

    if solution_passed:
        print("\n✓ Solution is correct!")
        print("Compare your implementation with solution.cu")
        return 0
    else:
        print("\n✗ Solution failed - check implementation")
        return 1

if __name__ == "__main__":
    sys.exit(main())
