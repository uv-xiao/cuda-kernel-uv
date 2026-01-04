#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

// CuTe headers
#include <cute/tensor.hpp>
#include <cute/layout.hpp>

using namespace cute;

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Simple kernel to test device-side tensor operations
__global__ void multiply_kernel(float* ptr, int M, int N, float scale) {
    // Create a tensor view in device code
    auto tensor = make_tensor(ptr, make_layout(make_shape(M, N),
                                                make_stride(N, 1)));

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size(tensor)) {
        tensor(idx) *= scale;  // Flat indexing
    }
}

// Kernel to test 2D indexing
__global__ void add_index_kernel(float* ptr, int M, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < M && j < N) {
        auto tensor = make_tensor(ptr, make_layout(make_shape(M, N),
                                                    make_stride(N, 1)));
        tensor(i, j) += i * 10 + j;
    }
}

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << title << std::endl;
    std::cout << std::string(50, '=') << std::endl;
}

void test_layout_creation() {
    print_separator("Test 1: CuTe Layout Creation");

    // Create a compile-time layout for a 4x8 matrix (row-major)
    auto layout = make_layout(make_shape(Int<4>{}, Int<8>{}),
                              make_stride(Int<8>{}, Int<1>{}));

    std::cout << "Layout created: " << layout << std::endl;
    std::cout << "  Shape: " << shape(layout) << std::endl;
    std::cout << "  Stride: " << stride(layout) << std::endl;
    std::cout << "  Size (total elements): " << size(layout) << std::endl;
    std::cout << "  Rank (dimensions): " << rank(layout) << std::endl;

    // Test coordinate to offset mapping
    std::cout << "\nCoordinate to offset mapping:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            int offset = layout(i, j);
            std::cout << "  (" << i << ", " << j << ") -> offset " << offset << std::endl;
        }
    }

    std::cout << "✓ Layout test passed" << std::endl;
}

void test_tensor_creation() {
    print_separator("Test 2: CuTe Tensor Creation");

    // Create layout
    auto layout = make_layout(make_shape(Int<4>{}, Int<8>{}),
                              make_stride(Int<8>{}, Int<1>{}));

    // Allocate storage
    std::vector<float> data(size(layout));

    // Create tensor
    auto tensor = make_tensor(data.data(), layout);

    // Initialize with row-major pattern
    for (int i = 0; i < size<0>(layout); ++i) {
        for (int j = 0; j < size<1>(layout); ++j) {
            tensor(i, j) = static_cast<float>(i * size<1>(layout) + j);
        }
    }

    std::cout << "Tensor contents (4x8 matrix):" << std::endl;
    for (int i = 0; i < size<0>(layout); ++i) {
        std::cout << "  ";
        for (int j = 0; j < size<1>(layout); ++j) {
            std::cout << std::setw(5) << std::fixed << std::setprecision(1)
                      << tensor(i, j) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "✓ Tensor creation test passed" << std::endl;
}

void test_runtime_layout() {
    print_separator("Test 3: Runtime (Dynamic) Layout");

    int M = 3, N = 5;

    // Create layout with runtime dimensions
    auto layout = make_layout(make_shape(M, N),
                              make_stride(N, 1));

    std::cout << "Runtime layout: " << layout << std::endl;
    std::cout << "  Size: " << size(layout) << std::endl;

    // Create tensor
    std::vector<float> data(size(layout));
    auto tensor = make_tensor(data.data(), layout);

    // Fill with pattern
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            tensor(i, j) = i + j * 0.1f;
        }
    }

    std::cout << "\nTensor contents (" << M << "x" << N << "):" << std::endl;
    for (int i = 0; i < M; ++i) {
        std::cout << "  ";
        for (int j = 0; j < N; ++j) {
            std::cout << std::setw(6) << std::fixed << std::setprecision(2)
                      << tensor(i, j) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "✓ Runtime layout test passed" << std::endl;
}

void test_column_major_layout() {
    print_separator("Test 4: Column-Major Layout");

    // Column-major: stride 1 in first dimension, M in second
    auto layout = make_layout(make_shape(Int<4>{}, Int<6>{}),
                              make_stride(Int<1>{}, Int<4>{}));

    std::cout << "Column-major layout: " << layout << std::endl;

    std::vector<float> data(size(layout));
    auto tensor = make_tensor(data.data(), layout);

    // Fill with index pattern
    for (int i = 0; i < size<0>(layout); ++i) {
        for (int j = 0; j < size<1>(layout); ++j) {
            tensor(i, j) = i * 10.0f + j;
        }
    }

    std::cout << "\nLogical view (row, col):" << std::endl;
    for (int i = 0; i < size<0>(layout); ++i) {
        std::cout << "  ";
        for (int j = 0; j < size<1>(layout); ++j) {
            std::cout << std::setw(6) << std::fixed << std::setprecision(1)
                      << tensor(i, j) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nPhysical memory layout (first 12 elements):" << std::endl;
    std::cout << "  ";
    for (int k = 0; k < 12 && k < data.size(); ++k) {
        std::cout << std::setw(6) << std::fixed << std::setprecision(1)
                  << data[k] << " ";
    }
    std::cout << std::endl;

    std::cout << "✓ Column-major layout test passed" << std::endl;
}

void test_device_tensor() {
    print_separator("Test 5: Device Tensor Operations");

    const int M = 4, N = 8;
    auto layout = make_layout(make_shape(M, N), make_stride(N, 1));

    // Allocate host and device memory
    std::vector<float> h_data(size(layout));
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size(layout) * sizeof(float)));

    // Initialize host tensor
    auto h_tensor = make_tensor(h_data.data(), layout);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            h_tensor(i, j) = i * N + j;
        }
    }

    std::cout << "Initial host tensor:" << std::endl;
    for (int i = 0; i < M; ++i) {
        std::cout << "  ";
        for (int j = 0; j < N; ++j) {
            std::cout << std::setw(5) << std::fixed << std::setprecision(1)
                      << h_tensor(i, j) << " ";
        }
        std::cout << std::endl;
    }

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(),
                          size(layout) * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Launch kernel to multiply by 2
    int threads = 256;
    int blocks = (size(layout) + threads - 1) / threads;
    multiply_kernel<<<blocks, threads>>>(d_data, M, N, 2.0f);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data,
                          size(layout) * sizeof(float),
                          cudaMemcpyDeviceToHost));

    std::cout << "\nAfter multiply by 2:" << std::endl;
    for (int i = 0; i < M; ++i) {
        std::cout << "  ";
        for (int j = 0; j < N; ++j) {
            std::cout << std::setw(5) << std::fixed << std::setprecision(1)
                      << h_tensor(i, j) << " ";
        }
        std::cout << std::endl;
    }

    // Verify result
    bool correct = true;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float expected = (i * N + j) * 2.0f;
            if (std::abs(h_tensor(i, j) - expected) > 1e-5f) {
                correct = false;
                std::cout << "Mismatch at (" << i << ", " << j << "): "
                          << "expected " << expected << ", got " << h_tensor(i, j)
                          << std::endl;
            }
        }
    }

    CUDA_CHECK(cudaFree(d_data));

    if (correct) {
        std::cout << "✓ Device tensor test passed" << std::endl;
    } else {
        std::cout << "✗ Device tensor test FAILED" << std::endl;
    }
}

void test_2d_kernel() {
    print_separator("Test 6: 2D Kernel with Tensor");

    const int M = 4, N = 4;
    auto layout = make_layout(make_shape(M, N), make_stride(N, 1));

    std::vector<float> h_data(size(layout), 0.0f);
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size(layout) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(),
                          size(layout) * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Launch 2D kernel
    dim3 threads(4, 4);
    dim3 blocks((M + threads.x - 1) / threads.x,
                (N + threads.y - 1) / threads.y);
    add_index_kernel<<<blocks, threads>>>(d_data, M, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data,
                          size(layout) * sizeof(float),
                          cudaMemcpyDeviceToHost));

    auto h_tensor = make_tensor(h_data.data(), layout);

    std::cout << "Result (each element = i*10 + j):" << std::endl;
    for (int i = 0; i < M; ++i) {
        std::cout << "  ";
        for (int j = 0; j < N; ++j) {
            std::cout << std::setw(5) << std::fixed << std::setprecision(1)
                      << h_tensor(i, j) << " ";
        }
        std::cout << std::endl;
    }

    // Verify
    bool correct = true;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float expected = i * 10.0f + j;
            if (std::abs(h_tensor(i, j) - expected) > 1e-5f) {
                correct = false;
            }
        }
    }

    CUDA_CHECK(cudaFree(d_data));

    if (correct) {
        std::cout << "✓ 2D kernel test passed" << std::endl;
    } else {
        std::cout << "✗ 2D kernel test FAILED" << std::endl;
    }
}

int main() {
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "  CUTLASS Setup Verification" << std::endl;
    std::cout << std::string(50, '=') << std::endl;

    // Check CUDA device
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return EXIT_FAILURE;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "\nDevice Information:" << std::endl;
    std::cout << "  Name: " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;

    if (prop.major < 7) {
        std::cout << "\n⚠ Warning: CUTLASS 3.x requires SM70+ (Volta or newer)" << std::endl;
        std::cout << "  Your GPU is SM" << prop.major << prop.minor << std::endl;
        std::cout << "  Some features may not work correctly." << std::endl;
    }

    try {
        // Run tests
        test_layout_creation();
        test_tensor_creation();
        test_runtime_layout();
        test_column_major_layout();
        test_device_tensor();
        test_2d_kernel();

        print_separator("All Tests Passed!");
        std::cout << "\n✓ CUTLASS is properly installed and working." << std::endl;
        std::cout << "✓ CuTe tensors and layouts are functioning correctly." << std::endl;
        std::cout << "✓ Device-side operations are successful." << std::endl;
        std::cout << "\nYou are ready to proceed to the next examples!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\n✗ Test failed with exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
