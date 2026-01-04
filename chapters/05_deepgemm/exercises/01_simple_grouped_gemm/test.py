#!/usr/bin/env python3
"""Test harness for grouped GEMM exercise"""

import subprocess
import sys
import os

def compile_and_run(source_file):
    """Compile and run a CUDA source file"""
    print(f"\n{'='*80}")
    print(f"Testing: {source_file}")
    print(f"{'='*80}\n")

    # Compile
    executable = source_file.replace('.cu', '')
    compile_cmd = [
        'nvcc',
        '-O3',
        '--use_fast_math',
        '-arch=sm_80',  # Adjust for your GPU
        source_file,
        '-o', executable
    ]

    print("Compiling...")
    print(' '.join(compile_cmd))

    try:
        result = subprocess.run(compile_cmd, capture_output=True, text=True, check=True)
        print("Compilation successful!")
    except subprocess.CalledProcessError as e:
        print("Compilation failed!")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

    # Run
    print("\nRunning tests...")
    try:
        result = subprocess.run([f'./{executable}'], capture_output=True, text=True,
                               timeout=60, check=True)
        print(result.stdout)

        # Check for PASSED in output
        if 'PASSED' in result.stdout and 'FAILED' not in result.stdout:
            print(f"\n{source_file}: ALL TESTS PASSED")
            return True
        else:
            print(f"\n{source_file}: SOME TESTS FAILED")
            return False

    except subprocess.TimeoutExpired:
        print("Test timed out!")
        return False
    except subprocess.CalledProcessError as e:
        print("Test execution failed!")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    finally:
        # Cleanup
        if os.path.exists(executable):
            os.remove(executable)


def main():
    print("Grouped GEMM Exercise Test Suite")

    # Test solution first
    print("\n" + "="*80)
    print("Testing Reference Solution")
    print("="*80)

    if not compile_and_run('solution.cu'):
        print("\nWarning: Reference solution failed! Check your setup.")
        return 1

    # Test student solution if it exists
    if os.path.exists('student_solution.cu'):
        print("\n" + "="*80)
        print("Testing Student Solution")
        print("="*80)

        if compile_and_run('student_solution.cu'):
            print("\n" + "="*80)
            print("CONGRATULATIONS! All tests passed!")
            print("="*80)
            return 0
        else:
            print("\n" + "="*80)
            print("Some tests failed. Keep working on your solution!")
            print("="*80)
            return 1
    else:
        print("\nNo student_solution.cu found. Implement your solution and run again.")
        print("\nYou can start from starter.cu:")
        print("  cp starter.cu student_solution.cu")
        print("  # Edit student_solution.cu")
        print("  python test.py")
        return 0


if __name__ == '__main__':
    sys.exit(main())
