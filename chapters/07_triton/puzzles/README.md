# Triton Puzzles

Interactive coding challenges to learn Triton kernel programming.

Inspired by [Triton-Puzzles](https://github.com/srush/Triton-Puzzles) by Sasha Rush.

## Overview

These puzzles are fill-in-the-blank exercises that teach Triton concepts progressively. Each puzzle focuses on specific Triton operations and patterns.

## Puzzle List

1. **puzzle_01_add.py** - Vector addition (basics: program_id, load, store)
2. **puzzle_02_broadcast.py** - Broadcasting operations
3. **puzzle_03_matmul.py** - Matrix multiplication with tiling
4. **puzzle_04_softmax.py** - Reductions and numerical stability

## How to Use

1. **Open a puzzle file** (e.g., `puzzle_01_add.py`)
2. **Read the instructions** at the top
3. **Fill in the blanks** marked with `___` or `# YOUR CODE HERE`
4. **Run the test function** to check your solution
5. **Compare with solution** in `puzzle_solutions.py` if stuck

## Learning Approach

### Step 1: Understand the Goal
Read the docstring to understand what the kernel should compute.

### Step 2: Identify Missing Pieces
Look for blanks (`___`) or `# YOUR CODE HERE` comments.

### Step 3: Fill in Code
Use your knowledge of Triton operations to complete the kernel.

### Step 4: Test
Uncomment the test function and run the file.

### Step 5: Debug
If tests fail, use print statements or compare with solution.

## Puzzle Difficulty

- **Puzzle 01**: Beginner - Basic operations
- **Puzzle 02**: Beginner - Broadcasting
- **Puzzle 03**: Intermediate - 2D tiling
- **Puzzle 04**: Intermediate - Reductions

## Example: Puzzle 01

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    # 1. Get the program ID
    pid = ___  # BLANK 1

    # 2. Calculate starting offset
    block_start = ___  # BLANK 2

    # 3. Create offsets
    offsets = ___  # BLANK 3

    # 4. Create mask
    mask = ___  # BLANK 4

    # 5. Load data
    x = ___  # BLANK 5
    y = ___  # BLANK 6

    # 6. Compute
    output = ___  # BLANK 7

    # 7. Store
    ___  # BLANK 8
```

## Solutions

Solutions are provided in `puzzle_solutions.py`. Try to solve puzzles on your own first!

To check your solution:
```bash
python puzzle_01_add.py  # Run your attempt
python puzzle_solutions.py  # Run all solutions
```

## Tips

1. **Start Simple**: Begin with Puzzle 01, don't skip ahead
2. **Review Examples**: Check `examples/` directory for similar patterns
3. **Use Documentation**: Refer to Triton language reference
4. **Experiment**: Try different approaches, see what works
5. **Ask for Help**: Compare with solutions if stuck >15 minutes

## Common Triton Operations

Quick reference for puzzles:

```python
# Program ID
pid = tl.program_id(axis)  # 0, 1, or 2

# Create ranges
offsets = tl.arange(start, end)

# Memory operations
data = tl.load(ptr + offsets, mask=mask, other=0.0)
tl.store(ptr + offsets, data, mask=mask)

# Arithmetic
result = x + y
result = x * y
result = tl.maximum(x, y)
result = tl.exp(x)
result = tl.sqrt(x)

# Reductions
max_val = tl.max(x, axis=0)
sum_val = tl.sum(x, axis=0)

# Matrix operations
c = tl.dot(a, b)  # Matrix multiply

# Broadcasting
# Shapes automatically broadcast like NumPy
```

## Extension Challenges

After completing all puzzles, try:

1. **Optimize**: Make puzzles faster with better block sizes
2. **Extend**: Add features (e.g., fused operations)
3. **Create**: Design your own puzzles
4. **Benchmark**: Compare performance with PyTorch

## Next Steps

After puzzles:
- Work on **Exercises** for more complex problems
- Implement custom kernels for your own use cases
- Contribute puzzles to the community!

## Resources

- [Triton-Puzzles Original](https://github.com/srush/Triton-Puzzles)
- [Triton Language Reference](https://triton-lang.org/main/python-api/triton.language.html)
- [Examples Directory](../examples/)
