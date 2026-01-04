# Contributing to CUDA Kernel Tutorial

Thank you for your interest in contributing! This document provides guidelines for contributing to this educational repository.

## Ways to Contribute

### 1. Reporting Issues
- Use GitHub Issues to report bugs or suggest improvements
- Include clear descriptions and steps to reproduce
- Mention your CUDA version, GPU model, and OS

### 2. Improving Documentation
- Fix typos or clarify explanations
- Add more examples or exercises
- Improve README files

### 3. Adding New Content
- New examples demonstrating CUDA concepts
- Additional exercises with solutions
- Performance optimization techniques

### 4. Code Improvements
- Bug fixes in existing examples
- Performance optimizations
- Better error handling

## Development Guidelines

### Code Style

**CUDA/C++**:
```cpp
// Use descriptive variable names
int num_elements = 1024;

// Include comprehensive error checking
CUDA_CHECK(cudaMalloc(&d_data, size));

// Add comments explaining non-obvious code
// Warp-level reduction using butterfly pattern
for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_xor_sync(0xffffffff, sum, offset);
}
```

**Python**:
```python
# Use type hints
def benchmark_kernel(kernel_fn: Callable, warmup: int = 5) -> float:
    """Benchmark a kernel function and return average time in ms."""
    pass
```

### Testing

- All code must compile without warnings
- Include test scripts for exercises
- Verify correctness against reference implementations
- Document expected performance metrics

### Documentation

- Each example needs a README.md explaining:
  - What the example demonstrates
  - How to build and run
  - Expected output or metrics
  - Key learning points

- Use proper Markdown formatting
- Include code snippets with syntax highlighting
- Reference official NVIDIA documentation where appropriate

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-example`)
3. Make your changes
4. Run tests to verify
5. Submit a pull request with:
   - Clear description of changes
   - Why the change is needed
   - Any new dependencies

## Content Standards

### Examples Should:
- Be self-contained and runnable
- Include proper error handling
- Have comprehensive comments
- Build with standard CMake

### Exercises Should:
- Have clear problem statements
- Include starter code with TODOs
- Provide complete solutions
- Include automated tests

### Performance Claims:
- Specify GPU model and CUDA version
- Include reproducible benchmarks
- Compare against baseline (cuBLAS, PyTorch, etc.)
- Document measurement methodology

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
