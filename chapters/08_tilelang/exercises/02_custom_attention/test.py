"""
Test script for Exercise 2: Custom Attention Pattern
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from solution import test_correctness, benchmark_performance

    print("Running Exercise 2 Tests...")
    test_correctness()
    benchmark_performance()
    print("\n✓ All Exercise 2 tests passed!")

except ImportError:
    print("✗ Could not import solution. Make sure solution.py is complete.")
    sys.exit(1)
except Exception as e:
    print(f"✗ Tests failed with error: {e}")
    sys.exit(1)
