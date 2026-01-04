"""
Exercise 1: Tiled Reduction - Starter Code
===========================================

Implement a tiled reduction that sums all elements of an array.

TODO: Complete the implementation following the hints in problem.md
"""

import tilelang as T
import torch


@T.prim_func
def tiled_reduction(
    A: T.Buffer((1048576,), "float32"),
    partial_sums: T.Buffer((4096,), "float32")  # One sum per block
):
    """
    Compute partial sums for reduction.

    Each block computes one partial sum.
    Final reduction happens in a second pass (CPU or another kernel).
    """
    N = 1048576
    BLOCK_SIZE = 256
    ELEMENTS_PER_THREAD = 4

    with T.block("root"):
        # TODO: Get block and thread indices
        # bx = T.thread_binding(...)
        # tx = T.thread_binding(...)

        # TODO: Allocate shared memory for block reduction
        # shared = T.alloc_shared(...)

        # TODO: Allocate register for thread-local accumulation
        # thread_sum = T.alloc_fragment(...)

        # TODO: Step 1 - Each thread accumulates multiple elements
        # for i in T.serial(ELEMENTS_PER_THREAD):
        #     idx = ...
        #     if idx < N:
        #         thread_sum[0] += A[idx]

        # TODO: Step 2 - Store thread sum in shared memory
        # shared[tx] = thread_sum[0]
        # T.sync_threads()

        # TODO: Step 3 - Tree reduction in shared memory
        # stride = BLOCK_SIZE // 2
        # while stride > 0:
        #     if tx < stride:
        #         shared[tx] = shared[tx] + shared[tx + stride]
        #     T.sync_threads()
        #     stride = stride // 2

        # TODO: Step 4 - Thread 0 writes block result
        # if tx == 0:
        #     partial_sums[bx] = shared[0]

        pass  # Remove this when you implement


def test_reduction():
    """Test the reduction implementation."""
    print("Testing tiled_reduction...")

    # Test case 1: All ones
    print("\nTest 1: Sum of ones")
    N = 1048576
    num_blocks = N // (256 * 4)

    A = torch.ones(N, device="cuda", dtype=torch.float32)
    partial_sums = torch.zeros(num_blocks, device="cuda", dtype=torch.float32)

    # Compile and run
    mod = T.compile(tiled_reduction, target="cuda")
    mod(A, partial_sums)

    # Final reduction on CPU
    result = partial_sums.sum().item()
    expected = float(N)

    print(f"  Expected: {expected}")
    print(f"  Got:      {result}")
    print(f"  Error:    {abs(result - expected)}")

    # Test case 2: Sequential values
    print("\nTest 2: Sum of 0..N-1")
    A = torch.arange(N, device="cuda", dtype=torch.float32)
    partial_sums = torch.zeros(num_blocks, device="cuda", dtype=torch.float32)

    mod(A, partial_sums)
    result = partial_sums.sum().item()
    expected = float(N * (N - 1) // 2)

    print(f"  Expected: {expected}")
    print(f"  Got:      {result}")
    print(f"  Relative error: {abs(result - expected) / expected}")


def main():
    """Main function."""
    print("="*60)
    print("Exercise 1: Tiled Reduction")
    print("="*60)

    test_reduction()

    print("\nTODO: Complete the implementation!")
    print("See problem.md for detailed requirements.")


if __name__ == "__main__":
    main()
