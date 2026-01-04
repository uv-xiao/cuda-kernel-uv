"""
Test script for Exercise 1: Tiled Reduction
"""

import sys
import os

# Add parent directory to path to import solution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from solution import test_correctness, benchmark_performance

    print("Running Exercise 1 Tests...")
    test_correctness()
    benchmark_performance()
    print("\n✓ All Exercise 1 tests passed!")

except ImportError:
    print("✗ Could not import solution. Make sure solution.py is complete.")
    sys.exit(1)
except Exception as e:
    print(f"✗ Tests failed with error: {e}")
    sys.exit(1)
