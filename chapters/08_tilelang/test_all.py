"""
Test All TileLang Examples
===========================

This script runs all examples to verify they work correctly.
Run this after installing TileLang to ensure everything is set up properly.

Usage:
    python test_all.py
"""

import sys
import subprocess
import os


def run_test(name, script_path):
    """Run a test script and report results."""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"{'='*70}")

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            print(f"✓ {name} PASSED")
            return True
        else:
            print(f"✗ {name} FAILED")
            print("STDERR:", result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print(f"✗ {name} TIMEOUT")
        return False
    except Exception as e:
        print(f"✗ {name} ERROR: {e}")
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("TileLang Chapter 08 - Test Suite")
    print("="*70)

    # Get base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # List of tests to run
    tests = [
        # Basic examples
        ("01 - Hello TileLang",
         os.path.join(base_dir, "examples/01_tilelang_basics/hello_tilelang.py")),
        ("01 - Memory Hierarchy",
         os.path.join(base_dir, "examples/01_tilelang_basics/memory_hierarchy.py")),

        # GEMM examples
        ("02 - Simple GEMM",
         os.path.join(base_dir, "examples/02_gemm/gemm_simple.py")),
        ("02 - Pipelined GEMM",
         os.path.join(base_dir, "examples/02_gemm/gemm_pipelined.py")),

        # Attention examples
        ("03 - FlashAttention",
         os.path.join(base_dir, "examples/03_attention/flash_attention.py")),

        # MLA examples
        ("04 - MLA Decode",
         os.path.join(base_dir, "examples/04_mla_decoding/mla_decode.py")),

        # Exercise solutions
        ("Exercise 1 - Tiled Reduction",
         os.path.join(base_dir, "exercises/01_tiled_reduction/solution.py")),
        ("Exercise 2 - Sliding Window Attention",
         os.path.join(base_dir, "exercises/02_custom_attention/solution.py")),
    ]

    # Run tests
    results = []
    for name, path in tests:
        if not os.path.exists(path):
            print(f"\nSkipping {name} (file not found: {path})")
            results.append((name, False))
            continue

        passed = run_test(name, path)
        results.append((name, passed))

    # Print summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {name}")

    print("="*70)
    print(f"Results: {passed}/{total} tests passed ({100*passed//total}%)")
    print("="*70)

    if passed == total:
        print("\n🎉 All tests passed! TileLang is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
