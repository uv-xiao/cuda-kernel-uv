#!/usr/bin/env python3
"""Test script for FP16 GEMM exercise"""

import subprocess
import sys

def run_test(executable):
    try:
        result = subprocess.run([executable], capture_output=True, text=True, timeout=120)
        print(result.stdout)

        if result.returncode != 0:
            print(f"Error: {result.stderr}", file=sys.stderr)
            return False

        return "✓" in result.stdout and "✗" not in result.stdout

    except subprocess.TimeoutExpired:
        print("Timeout", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"Not found: {executable}", file=sys.stderr)
        return False

def main():
    print("=" * 70)
    print("FP16 GEMM Test Suite")
    print("=" * 70)

    solution_passed = run_test("./fp16_gemm_solution")

    print("\n" + "=" * 70)
    if solution_passed:
        print("✓ Solution passed!")
        return 0
    else:
        print("✗ Failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
