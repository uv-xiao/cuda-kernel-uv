"""
Exercise 02: GELU Activation - Starter Code

TODO: Implement GELU activation function.
Formula: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
"""

import torch
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    TODO: Implement GELU kernel.

    Hints:
    1. Load input values
    2. Compute x^3
    3. Compute inner term with constants
    4. Apply tanh using tl.libdevice.tanh()
    5. Compute final GELU value
    6. Store output
    """
    # TODO: Get program ID and calculate offsets
    pid = # YOUR CODE HERE
    offsets = # YOUR CODE HERE
    mask = # YOUR CODE HERE

    # TODO: Load input
    x = # YOUR CODE HERE

    # Constants
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    coeff = 0.044715

    # TODO: Compute GELU
    # Step 1: Compute x^3
    x_cubed = # YOUR CODE HERE

    # Step 2: Compute inner term
    inner = # YOUR CODE HERE

    # Step 3: Apply tanh (use tl.libdevice.tanh)
    tanh_inner = # YOUR CODE HERE

    # Step 4: Compute final GELU
    gelu = # YOUR CODE HERE

    # TODO: Store output
    # YOUR CODE HERE


def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    """Wrapper function."""
    output = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    gelu_kernel[grid](x, output, n_elements, BLOCK_SIZE)

    return output


if __name__ == "__main__":
    print("Exercise 02: GELU Activation")
    print("=" * 70)
    print("TODO: Implement the gelu_kernel function")
    print()
    print("Formula: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))")
    print()
    print("When ready, run: python test.py")
