#!/usr/bin/env python3
"""Test script for block-level scan exercise."""

import subprocess
import sys
import os

def compile_and_run(source_file):
    """Compile and run CUDA program."""
    executable = source_file.replace('.cu', '')

    print(f"Compiling {source_file}...")
    compile_cmd = ['nvcc', source_file, '-o', executable, '-arch=sm_60']
    result = subprocess.run(compile_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Compilation failed:")
        print(result.stderr)
        return False, None

    print(f"Compilation successful.\n")

    print(f"Running {executable}...")
    result = subprocess.run([f'./{executable}'], capture_output=True, text=True)

    if os.path.exists(executable):
        os.remove(executable)

    return True, result.stdout

def check_output(output):
    """Check if all tests passed."""
    if output is None:
        return False

    lines = output.strip().split('\n')
    pass_count = sum(1 for line in lines if 'PASS' in line)
    fail_count = sum(1 for line in lines if 'FAIL' in line)

    return pass_count == 3 and fail_count == 0

def main():
    """Main test function."""
    print("=" * 60)
    print("Testing Block-Level Scan Exercise")
    print("=" * 60)
    print()

    print("Testing starter.cu:")
    print("-" * 60)
    success, output = compile_and_run('starter.cu')
    if success:
        print(output)

    print("\nTesting solution.cu:")
    print("-" * 60)
    success, output = compile_and_run('solution.cu')
    if not success:
        print("Failed to compile or run solution.cu")
        sys.exit(1)

    print(output)

    if check_output(output):
        print("\n" + "=" * 60)
        print("SUCCESS! All tests passed.")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("FAILURE! Some tests failed.")
        print("=" * 60)
        sys.exit(1)

if __name__ == '__main__':
    main()
