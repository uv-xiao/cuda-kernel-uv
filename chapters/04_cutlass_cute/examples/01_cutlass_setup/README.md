# Example 01: CUTLASS Setup and Verification

## Overview

This example verifies that CUTLASS is properly installed and can be compiled with your CUDA environment. It demonstrates basic CUTLASS usage and CuTe tensor creation.

## Learning Objectives

- Install and configure CUTLASS library
- Verify compilation with CUTLASS headers
- Create basic CuTe tensors
- Understand CUTLASS directory structure
- Set up build environment for future examples

## Prerequisites

- CUDA Toolkit 11.0 or later
- CMake 3.18 or later
- C++17 compatible compiler
- NVIDIA GPU with compute capability 7.0+ (Volta or newer)

## Installation Instructions

### Step 1: Clone CUTLASS

```bash
# Navigate to a suitable location (e.g., your home directory or a libraries folder)
cd ~/libraries  # or wherever you keep third-party libraries

# Clone CUTLASS from GitHub
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass

# Checkout a stable version (recommended)
git checkout v3.4.0  # or latest stable release

# Note the installation path
export CUTLASS_DIR=$(pwd)
echo "export CUTLASS_DIR=$CUTLASS_DIR" >> ~/.bashrc
```

### Step 2: Verify CUTLASS Structure

CUTLASS has the following key directories:

```
cutlass/
├── include/              # Header files (what you'll use)
│   ├── cute/            # CuTe library headers
│   └── cutlass/         # CUTLASS library headers
├── examples/            # Official examples
│   └── cute/           # CuTe examples
├── test/               # Unit tests
├── tools/              # Utilities
└── media/docs/         # Documentation
```

### Step 3: Build This Example

```bash
# From the examples/01_cutlass_setup directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCUTLASS_DIR=$CUTLASS_DIR

# Build
make -j$(nproc)

# Run the test
./test_cutlass
```

## What This Example Does

The `test_cutlass.cu` program:

1. **Includes CUTLASS headers**: Verifies header files are accessible
2. **Creates CuTe tensors**: Demonstrates basic tensor creation
3. **Performs simple operations**: Shows layout and coordinate manipulation
4. **Validates results**: Ensures correct computation
5. **Prints diagnostics**: Displays tensor layouts and values

## Expected Output

```
=== CUTLASS Setup Verification ===

1. Testing CuTe Layout creation...
   Layout shape: (_4, _8)
   Layout stride: (_8, _1)
   Layout size: 32
   ✓ Layout created successfully

2. Testing CuTe Tensor creation...
   Tensor on host: 4x8 matrix
   ✓ Tensor created successfully

3. Testing basic tensor operations...
   Tensor element (2, 3) = 19
   ✓ Indexing works correctly

4. Testing device tensor...
   ✓ Device tensor operations successful

=== All tests passed! ===
CUTLASS is properly installed and working.
```

## Understanding the Code

### CuTe Layouts

```cpp
using namespace cute;

// Create a row-major 4x8 layout
auto layout = make_layout(make_shape(Int<4>{}, Int<8>{}),
                          make_stride(Int<8>{}, Int<1>{}));
```

**Key concepts:**
- `Shape`: Logical dimensions (4 rows, 8 columns)
- `Stride`: Memory offset per dimension (stride 8 for row, 1 for column)
- `Int<N>`: Compile-time integer constant (enables compile-time optimizations)

### CuTe Tensors

```cpp
// Host tensor with automatic storage
Tensor<float, decltype(layout)> tensor_host(layout);

// Device tensor (pointer + layout)
Tensor tensor_device = make_tensor(device_ptr, layout);
```

**Key concepts:**
- Tensor = Pointer + Layout
- Type-safe coordinate access
- Compile-time shape checking

### Memory Operations

```cpp
// Copy host to device
cudaMemcpy(tensor_device.data(), tensor_host.data(),
           size(tensor_host) * sizeof(float),
           cudaMemcpyHostToDevice);
```

Standard CUDA memory operations work with CuTe tensors via `.data()`.

## Troubleshooting

### Problem: "cute/tensor.hpp: No such file or directory"

**Solution:** Set `CUTLASS_DIR` correctly:
```bash
cmake .. -DCUTLASS_DIR=/path/to/cutlass
```

### Problem: "error: namespace 'cute' has no member 'make_layout'"

**Solution:** Ensure you're using CUTLASS 3.x (not 2.x):
```bash
cd $CUTLASS_DIR
git checkout v3.4.0
```

### Problem: Compilation errors about C++17

**Solution:** Update CMakeLists.txt or compiler:
```cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
```

### Problem: Runtime errors on GPU

**Solution:** Check compute capability:
```bash
# Your GPU must be SM70 or newer
nvidia-smi --query-gpu=compute_cap --format=csv
```

## Code Walkthrough

### 1. Header Inclusion

```cpp
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
```

These are the core CuTe headers. Most CuTe code only needs these two.

### 2. Layout Creation

```cpp
auto shape = make_shape(Int<4>{}, Int<8>{});
auto stride = make_stride(Int<8>{}, Int<1>{});
auto layout = make_layout(shape, stride);
```

Creates a row-major 4x8 layout at compile time.

### 3. Tensor Manipulation

```cpp
for (int i = 0; i < size<0>(layout); ++i) {
    for (int j = 0; j < size<1>(layout); ++j) {
        tensor_host(i, j) = i * size<1>(layout) + j;
    }
}
```

Uses compile-time dimensions with `size<N>()`.

### 4. Device Kernel

```cpp
__global__ void test_kernel(float* ptr, int M, int N) {
    auto tensor = make_tensor(ptr, make_layout(make_shape(M, N),
                                                make_stride(N, 1)));
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size(tensor)) {
        tensor(idx) *= 2.0f;  // Flat indexing
    }
}
```

Demonstrates creating tensors in device code.

## Next Steps

After successfully running this example:

1. **Explore the code**: Modify shapes and strides to see different layouts
2. **Try different layouts**: Column-major, blocked, etc.
3. **Add error checking**: Wrap CUDA calls with error macros
4. **Move to example 02**: Learn about layout algebra and transformations

## Additional Notes

### Compile-time vs Runtime

CuTe supports both:

```cpp
// Compile-time (preferred when known)
auto layout1 = make_layout(make_shape(Int<4>{}, Int<8>{}),
                           make_stride(Int<8>{}, Int<1>{}));

// Runtime (when sizes are dynamic)
auto layout2 = make_layout(make_shape(m, n),
                           make_stride(n, 1));
```

**Compile-time benefits:**
- Better optimization
- Smaller code size
- Compile-time error checking

### CUTLASS Versioning

This tutorial targets CUTLASS 3.x. Major differences from 2.x:

| Feature | CUTLASS 2.x | CUTLASS 3.x |
|---------|-------------|-------------|
| Core abstraction | GEMM hierarchy | CuTe |
| Layout handling | Hardcoded | Flexible algebra |
| Code size | Larger | More compact |
| Ease of use | Complex | Simplified |

### Performance Note

This example focuses on correctness, not performance. Performance-critical code will be introduced in later examples (03_cute_gemm onwards).

## References

- [CuTe Quickstart](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md)
- [CuTe Layout Tutorial](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/01_layout.md)
- [CUTLASS Examples](https://github.com/NVIDIA/cutlass/tree/main/examples/cute)
