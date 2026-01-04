#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>

#include <cute/tensor.hpp>
#include <cute/layout.hpp>

using namespace cute;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << title << std::endl;
    std::cout << std::string(70, '=') << std::endl;
}

// Demonstrate row-major layout
void test_row_major() {
    print_separator("Test 1: Row-Major Layout");

    // Create a 4x8 row-major layout
    // Stride: (8, 1) means advancing row increases offset by 8, column by 1
    auto layout = make_layout(make_shape(Int<4>{}, Int<8>{}),
                              make_stride(Int<8>{}, Int<1>{}));

    std::cout << "Layout: " << layout << std::endl;
    std::cout << "Shape: " << shape(layout) << std::endl;
    std::cout << "Stride: " << stride(layout) << std::endl;
    std::cout << "Total size: " << size(layout) << std::endl;

    // Create tensor and fill with index pattern
    std::vector<float> data(size(layout));
    auto tensor = make_tensor(data.data(), layout);

    for (int i = 0; i < size<0>(layout); ++i) {
        for (int j = 0; j < size<1>(layout); ++j) {
            tensor(i, j) = i * size<1>(layout) + j;
        }
    }

    // Show logical view
    std::cout << "\nLogical view (tensor(i, j)):" << std::endl;
    for (int i = 0; i < size<0>(layout); ++i) {
        std::cout << "  ";
        for (int j = 0; j < size<1>(layout); ++j) {
            std::cout << std::setw(5) << std::fixed << std::setprecision(0)
                      << tensor(i, j) << " ";
        }
        std::cout << std::endl;
    }

    // Show physical memory layout
    std::cout << "\nPhysical memory layout (data[k]):" << std::endl;
    std::cout << "  ";
    for (size_t k = 0; k < data.size(); ++k) {
        std::cout << std::setw(5) << std::fixed << std::setprecision(0)
                  << data[k] << " ";
        if ((k + 1) % 8 == 0 && k + 1 < data.size()) {
            std::cout << "\n  ";
        }
    }
    std::cout << std::endl;

    // Show coordinate to offset mapping
    std::cout << "\nCoordinate to offset mapping:" << std::endl;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << "  (" << i << ", " << j << ") -> offset "
                      << layout(i, j) << std::endl;
        }
    }
}

// Demonstrate column-major layout
void test_column_major() {
    print_separator("Test 2: Column-Major Layout");

    // Create a 4x8 column-major layout
    // Stride: (1, 4) means advancing row increases offset by 1, column by 4
    auto layout = make_layout(make_shape(Int<4>{}, Int<8>{}),
                              make_stride(Int<1>{}, Int<4>{}));

    std::cout << "Layout: " << layout << std::endl;
    std::cout << "Shape: " << shape(layout) << std::endl;
    std::cout << "Stride: " << stride(layout) << std::endl;

    std::vector<float> data(size(layout));
    auto tensor = make_tensor(data.data(), layout);

    // Fill with same pattern
    for (int i = 0; i < size<0>(layout); ++i) {
        for (int j = 0; j < size<1>(layout); ++j) {
            tensor(i, j) = i * size<1>(layout) + j;
        }
    }

    // Show logical view (same as row-major)
    std::cout << "\nLogical view (tensor(i, j)):" << std::endl;
    for (int i = 0; i < size<0>(layout); ++i) {
        std::cout << "  ";
        for (int j = 0; j < size<1>(layout); ++j) {
            std::cout << std::setw(5) << std::fixed << std::setprecision(0)
                      << tensor(i, j) << " ";
        }
        std::cout << std::endl;
    }

    // Show physical memory (different from row-major!)
    std::cout << "\nPhysical memory layout (data[k]):" << std::endl;
    std::cout << "  ";
    for (size_t k = 0; k < data.size(); ++k) {
        std::cout << std::setw(5) << std::fixed << std::setprecision(0)
                  << data[k] << " ";
        if ((k + 1) % 8 == 0 && k + 1 < data.size()) {
            std::cout << "\n  ";
        }
    }
    std::cout << std::endl;

    std::cout << "\nNote: Same logical view, different physical layout!" << std::endl;
}

// Demonstrate compile-time vs runtime layouts
void test_compile_vs_runtime() {
    print_separator("Test 3: Compile-Time vs Runtime Layouts");

    // Compile-time layout: dimensions known at compile time
    auto static_layout = make_layout(make_shape(Int<4>{}, Int<8>{}),
                                     make_stride(Int<8>{}, Int<1>{}));

    // Runtime layout: dimensions determined at runtime
    int M = 4, N = 8;
    auto dynamic_layout = make_layout(make_shape(M, N),
                                      make_stride(N, 1));

    std::cout << "Static (compile-time) layout: " << static_layout << std::endl;
    std::cout << "Dynamic (runtime) layout: " << dynamic_layout << std::endl;

    std::cout << "\nBoth produce the same mapping, but compile-time has benefits:" << std::endl;
    std::cout << "  - Better optimization (loop unrolling, constant propagation)" << std::endl;
    std::cout << "  - Smaller code size (no runtime overhead)" << std::endl;
    std::cout << "  - Compile-time error checking" << std::endl;

    // Verify they produce same offsets
    std::cout << "\nVerifying equivalent mappings:" << std::endl;
    bool identical = true;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int offset_static = static_layout(i, j);
            int offset_dynamic = dynamic_layout(i, j);
            if (offset_static != offset_dynamic) {
                identical = false;
                std::cout << "  Mismatch at (" << i << ", " << j << "): "
                          << offset_static << " vs " << offset_dynamic << std::endl;
            }
        }
    }

    if (identical) {
        std::cout << "  ✓ All mappings identical" << std::endl;
    }
}

// Demonstrate 1D layouts
void test_1d_layout() {
    print_separator("Test 4: 1D Layouts");

    // Simple 1D layout
    auto layout = make_layout(Int<16>{});

    std::cout << "1D Layout: " << layout << std::endl;
    std::cout << "Shape: " << shape(layout) << std::endl;
    std::cout << "Stride: " << stride(layout) << std::endl;

    std::vector<float> data(size(layout));
    auto tensor = make_tensor(data.data(), layout);

    for (int i = 0; i < size(layout); ++i) {
        tensor(i) = i * 1.5f;
    }

    std::cout << "\nTensor contents:" << std::endl;
    std::cout << "  ";
    for (int i = 0; i < size(layout); ++i) {
        std::cout << std::setw(6) << std::fixed << std::setprecision(1)
                  << tensor(i) << " ";
    }
    std::cout << std::endl;
}

// Demonstrate hierarchical (nested) layouts
void test_hierarchical_layout() {
    print_separator("Test 5: Hierarchical Layouts");

    // Create a 2-level hierarchy: outer (2x2) and inner (4x4)
    // Total shape: 2*4 x 2*4 = 8x8
    auto layout = make_layout(
        make_shape(make_shape(Int<2>{}, Int<2>{}),  // Outer: 2x2 tiles
                   make_shape(Int<4>{}, Int<4>{})), // Inner: 4x4 elements per tile
        make_stride(make_stride(Int<32>{}, Int<4>{}),  // Outer strides
                    make_stride(Int<8>{}, Int<1>{}))   // Inner strides
    );

    std::cout << "Hierarchical layout: " << layout << std::endl;
    std::cout << "Total size: " << size(layout) << std::endl;

    // Create tensor
    std::vector<float> data(size(layout));
    auto tensor = make_tensor(data.data(), layout);

    // Fill with hierarchical pattern
    // First index: outer tile coordinates (tile_i, tile_j)
    // Second index: inner element coordinates (elem_i, elem_j)
    for (int tile_i = 0; tile_i < 2; ++tile_i) {
        for (int tile_j = 0; tile_j < 2; ++tile_j) {
            for (int elem_i = 0; elem_i < 4; ++elem_i) {
                for (int elem_j = 0; elem_j < 4; ++elem_j) {
                    tensor(make_coord(tile_i, tile_j), make_coord(elem_i, elem_j)) =
                        tile_i * 100 + tile_j * 10 + elem_i * 2 + elem_j * 0.1f;
                }
            }
        }
    }

    std::cout << "\nTensor contents (tile_ij.elem_ij):" << std::endl;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            std::cout << "\nTile (" << i << ", " << j << "):" << std::endl;
            for (int ei = 0; ei < 4; ++ei) {
                std::cout << "  ";
                for (int ej = 0; ej < 4; ++ej) {
                    std::cout << std::setw(8) << std::fixed << std::setprecision(1)
                              << tensor(make_coord(i, j), make_coord(ei, ej)) << " ";
                }
                std::cout << std::endl;
            }
        }
    }
}

// Demonstrate strided layouts for coalescing analysis
void test_strided_layout() {
    print_separator("Test 6: Strided Layouts and Coalescing");

    std::cout << "Understanding coalescing with different strides:\n" << std::endl;

    // Layout 1: Good coalescing (stride-1 in last dimension)
    auto coalesced = make_layout(make_shape(Int<4>{}, Int<32>{}),
                                  make_stride(Int<32>{}, Int<1>{}));

    // Layout 2: Poor coalescing (stride-1 in first dimension)
    auto strided = make_layout(make_shape(Int<4>{}, Int<32>{}),
                               make_stride(Int<1>{}, Int<4>{}));

    std::cout << "Coalesced layout (row-major): " << coalesced << std::endl;
    std::cout << "Strided layout (column-major): " << strided << std::endl;

    std::cout << "\nFor warp of 32 threads (threadIdx.x = 0..31):" << std::endl;

    std::cout << "\nCoalesced access pattern (threads access column 0):" << std::endl;
    std::cout << "  Threads 0-7 access offsets: ";
    for (int t = 0; t < 8; ++t) {
        std::cout << coalesced(0, t) << " ";
    }
    std::cout << "... (consecutive!)" << std::endl;

    std::cout << "\nStrided access pattern (threads access column 0):" << std::endl;
    std::cout << "  Threads 0-7 access offsets: ";
    for (int t = 0; t < 8; ++t) {
        std::cout << strided(0, t) << " ";
    }
    std::cout << "... (stride-4!)" << std::endl;

    std::cout << "\n✓ Coalesced: 32 threads in warp access bytes [0-127] consecutively" << std::endl;
    std::cout << "✗ Strided: 32 threads in warp access with stride 4 (poor coalescing)" << std::endl;
}

int main() {
    print_separator("CuTe Layout Basics");

    try {
        test_row_major();
        test_column_major();
        test_compile_vs_runtime();
        test_1d_layout();
        test_hierarchical_layout();
        test_strided_layout();

        print_separator("All Tests Completed Successfully!");

        std::cout << "\nKey Takeaways:" << std::endl;
        std::cout << "  1. Layout = Shape + Stride" << std::endl;
        std::cout << "  2. Same logical view can have different physical layouts" << std::endl;
        std::cout << "  3. Compile-time layouts enable better optimization" << std::endl;
        std::cout << "  4. Hierarchical layouts support multi-level tiling" << std::endl;
        std::cout << "  5. Stride-1 in innermost dimension enables coalescing" << std::endl;

        std::cout << "\nNext: Run layout_operations to learn about layout transformations!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
