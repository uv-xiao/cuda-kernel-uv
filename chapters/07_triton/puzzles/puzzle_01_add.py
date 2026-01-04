"""
Puzzle 01: Vector Addition

Fill in the blanks to implement vector addition in Triton.

Learning: Basic Triton operations (program_id, arange, load, store)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    TODO: Fill in the blanks to compute output = x + y
    """
    # 1. Get the program ID
    pid = ___  # BLANK 1: Get program ID for axis 0

    # 2. Calculate starting offset for this block
    block_start = ___  # BLANK 2: pid * BLOCK_SIZE

    # 3. Create offsets for elements in this block
    offsets = ___  # BLANK 3: block_start + tl.arange(0, BLOCK_SIZE)

    # 4. Create mask for boundary checking
    mask = ___  # BLANK 4: offsets < n_elements

    # 5. Load data
    x = ___  # BLANK 5: tl.load(x_ptr + offsets, mask=mask)
    y = ___  # BLANK 6: tl.load(y_ptr + offsets, mask=mask)

    # 6. Compute result
    output = ___  # BLANK 7: x + y

    # 7. Store result
    ___  # BLANK 8: tl.store(output_ptr + offsets, output, mask=mask)


def test_add():
    """Test your implementation."""
    n = 10000
    x = torch.randn(n, device='cuda')
    y = torch.randn(n, device='cuda')
    output = torch.empty_like(x)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    add_kernel[grid](x, y, output, n, BLOCK_SIZE)

    expected = x + y
    assert torch.allclose(output, expected), "Test failed!"
    print("✓ Puzzle 01 solved!")


if __name__ == "__main__":
    # Uncomment when ready to test
    # test_add()
    print("Fill in the blanks and uncomment test_add() to check your solution.")
