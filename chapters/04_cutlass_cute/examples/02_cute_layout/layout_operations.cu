#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>

#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/swizzle.hpp>

using namespace cute;

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << title << std::endl;
    std::cout << std::string(70, '=') << std::endl;
}

// Demonstrate layout composition
void test_composition() {
    print_separator("Test 1: Layout Composition");

    std::cout << "Composition combines two layouts: compose(L1, L2) = L1(L2(coord))\n" << std::endl;

    // Layout 1: Maps 4 elements in a row
    auto layout1 = make_layout(Int<4>{}, Int<2>{});  // 4 elements, stride 2
    std::cout << "Layout1: " << layout1 << std::endl;
    std::cout << "  Maps: ";
    for (int i = 0; i < 4; ++i) {
        std::cout << i << "->" << layout1(i) << " ";
    }
    std::cout << std::endl;

    // Layout 2: Maps 3 elements
    auto layout2 = make_layout(Int<3>{}, Int<1>{});  // 3 elements, stride 1
    std::cout << "\nLayout2: " << layout2 << std::endl;
    std::cout << "  Maps: ";
    for (int i = 0; i < 3; ++i) {
        std::cout << i << "->" << layout2(i) << " ";
    }
    std::cout << std::endl;

    // Composition: first apply layout2, then layout1
    auto composed = composition(layout1, layout2);
    std::cout << "\nComposed (layout1 ∘ layout2): " << composed << std::endl;
    std::cout << "  Maps: ";
    for (int i = 0; i < 3; ++i) {
        std::cout << i << "->" << composed(i) << " ";
    }
    std::cout << std::endl;

    std::cout << "\nInterpretation: coord -> layout2(coord) -> layout1(layout2(coord))" << std::endl;
    std::cout << "  0 -> layout2(0)=0 -> layout1(0)=0" << std::endl;
    std::cout << "  1 -> layout2(1)=1 -> layout1(1)=2" << std::endl;
    std::cout << "  2 -> layout2(2)=2 -> layout1(2)=4" << std::endl;
}

// Demonstrate 2D composition
void test_2d_composition() {
    print_separator("Test 2: 2D Layout Composition");

    // Outer layout: 2x2 tiles
    auto outer = make_layout(make_shape(Int<2>{}, Int<2>{}),
                             make_stride(Int<8>{}, Int<4>{}));

    // Inner layout: 4x4 elements per tile
    auto inner = make_layout(make_shape(Int<4>{}, Int<4>{}),
                             make_stride(Int<4>{}, Int<1>{}));

    std::cout << "Outer layout (2x2 tiles): " << outer << std::endl;
    std::cout << "Inner layout (4x4 per tile): " << inner << std::endl;

    auto composed = composition(outer, inner);
    std::cout << "\nComposed layout: " << composed << std::endl;

    std::cout << "\nSample mappings:" << std::endl;
    for (int tile_i = 0; tile_i < 2; ++tile_i) {
        for (int tile_j = 0; tile_j < 2; ++tile_j) {
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    auto coord = make_coord(make_coord(tile_i, tile_j), make_coord(i, j));
                    int offset = composed(coord);
                    std::cout << "  Tile(" << tile_i << "," << tile_j << ") "
                              << "Elem(" << i << "," << j << ") -> " << offset << std::endl;
                }
            }
        }
    }
}

// Demonstrate logical_divide (tiling)
void test_logical_divide() {
    print_separator("Test 3: Logical Divide (Tiling)");

    std::cout << "logical_divide splits a layout into tiles\n" << std::endl;

    // Create a 16x16 matrix layout
    auto layout = make_layout(make_shape(Int<16>{}, Int<16>{}),
                              make_stride(Int<16>{}, Int<1>{}));

    std::cout << "Original layout (16x16): " << layout << std::endl;
    std::cout << "  Total size: " << size(layout) << std::endl;

    // Divide into 4x4 tiles
    auto tiled = logical_divide(layout, make_shape(Int<4>{}, Int<4>{}));

    std::cout << "\nTiled layout (4x4 tiles): " << tiled << std::endl;
    std::cout << "  Outer shape (tiles): " << shape<0>(tiled) << std::endl;
    std::cout << "  Inner shape (elements): " << shape<1>(tiled) << std::endl;

    // Access pattern: tiled_layout(tile_coord, elem_coord)
    std::cout << "\nSample tile accesses:" << std::endl;
    for (int tile_i = 0; tile_i < 2; ++tile_i) {
        for (int tile_j = 0; tile_j < 2; ++tile_j) {
            std::cout << "\n  Tile (" << tile_i << ", " << tile_j << ") first element:" << std::endl;
            int offset = tiled(make_coord(tile_i, tile_j), make_coord(0, 0));
            std::cout << "    Offset: " << offset << std::endl;

            std::cout << "    First few elements: ";
            for (int i = 0; i < 4; ++i) {
                int off = tiled(make_coord(tile_i, tile_j), make_coord(0, i));
                std::cout << off << " ";
            }
            std::cout << std::endl;
        }
    }
}

// Demonstrate block/thread decomposition
void test_block_thread_decomposition() {
    print_separator("Test 4: Block-Thread Decomposition");

    std::cout << "Common pattern: Decompose matrix into block tiles and thread tiles\n" << std::endl;

    // Global matrix: 128x128
    const int M = 128, N = 128;
    auto gmem_layout = make_layout(make_shape(M, N), make_stride(N, 1));

    std::cout << "Global memory layout (128x128): " << gmem_layout << std::endl;

    // Block tile: 32x32
    const int BM = 32, BN = 32;
    auto block_tiled = logical_divide(gmem_layout, make_shape(BM, BN));

    std::cout << "\nAfter block tiling (32x32 tiles):" << std::endl;
    std::cout << "  Number of blocks: " << shape<0>(block_tiled) << std::endl;
    std::cout << "  Elements per block: " << shape<1>(block_tiled) << std::endl;

    // Further divide each block among threads
    const int TM = 8, TN = 8;
    auto thread_tiled = logical_divide(block_tiled, make_shape(TM, TN));

    std::cout << "\nAfter thread tiling (8x8 per thread):" << std::endl;
    std::cout << "  Layout rank: " << rank(thread_tiled) << std::endl;
    std::cout << "  Hierarchy: block_idx -> thread_idx -> element_idx" << std::endl;

    // Example: Block (1, 1), Thread (2, 2), Element (3, 3)
    int block_i = 1, block_j = 1;
    int thread_i = 2, thread_j = 2;
    int elem_i = 3, elem_j = 3;

    auto block_coord = make_coord(block_i, block_j);
    auto thread_coord = make_coord(thread_i, thread_j);
    auto elem_coord = make_coord(elem_i, elem_j);

    int offset = thread_tiled(block_coord, thread_coord, elem_coord);

    std::cout << "\nExample access:" << std::endl;
    std::cout << "  Block (" << block_i << ", " << block_j << ")" << std::endl;
    std::cout << "  Thread (" << thread_i << ", " << thread_j << ")" << std::endl;
    std::cout << "  Element (" << elem_i << ", " << elem_j << ")" << std::endl;
    std::cout << "  Global offset: " << offset << std::endl;

    // Verify manually
    int expected = (block_i * BM + thread_i * TM + elem_i) * N +
                   (block_j * BN + thread_j * TN + elem_j);
    std::cout << "  Expected offset: " << expected << std::endl;
    std::cout << "  " << (offset == expected ? "✓ Correct!" : "✗ Mismatch!") << std::endl;
}

// Demonstrate complement
void test_complement() {
    print_separator("Test 5: Layout Complement");

    std::cout << "complement finds orthogonal coordinates that fill a shape\n" << std::endl;

    // Layout covering some coordinates
    auto layout = make_layout(make_shape(Int<4>{}, Int<2>{}),
                              make_stride(Int<2>{}, Int<1>{}));

    std::cout << "Original layout: " << layout << std::endl;
    std::cout << "  Covers " << size(layout) << " coordinates" << std::endl;
    std::cout << "  Coordinate range: " << cosize(layout) << std::endl;

    // Find complement within shape of 16
    auto comp = complement(layout, Int<16>{});

    std::cout << "\nComplement (within size 16): " << comp << std::endl;
    std::cout << "  Shape: " << shape(comp) << std::endl;

    std::cout << "\nOriginal layout maps to offsets: ";
    for (int i = 0; i < size<0>(layout); ++i) {
        for (int j = 0; j < size<1>(layout); ++j) {
            std::cout << layout(i, j) << " ";
        }
    }
    std::cout << std::endl;

    std::cout << "Complement maps to offsets: ";
    for (int i = 0; i < size(comp); ++i) {
        std::cout << comp(i) << " ";
    }
    std::cout << std::endl;

    std::cout << "\nTogether they cover all offsets [0, 16)" << std::endl;
}

// Demonstrate swizzle for bank conflict avoidance
void test_swizzle() {
    print_separator("Test 6: Swizzle (Bank Conflict Avoidance)");

    std::cout << "Swizzle applies XOR pattern to avoid shared memory bank conflicts\n" << std::endl;

    // Base layout: 32 rows x 32 columns
    auto base = make_layout(make_shape(Int<32>{}, Int<32>{}),
                            make_stride(Int<32>{}, Int<1>{}));

    std::cout << "Base layout (32x32): " << base << std::endl;

    // Apply swizzle
    // Swizzle<B, M, S>: XOR bits [M+S-1:M] with bits [M-1:M-B]
    // Common: Swizzle<3, 0, 3> for 8-way swizzle
    using SwizzleOp = Swizzle<3, 0, 3>;
    auto swizzled = composition(base, SwizzleOp{});

    std::cout << "\nSwizzled layout: " << swizzled << std::endl;

    std::cout << "\nBank conflict example (column access):" << std::endl;
    std::cout << "  Without swizzle - all threads access column 0:" << std::endl;
    std::cout << "    Threads 0-7 access banks: ";
    for (int t = 0; t < 8; ++t) {
        int offset = base(t, 0);
        int bank = (offset * sizeof(float)) % 32;  // 32 banks, 4-byte words
        std::cout << bank / 4 << " ";
    }
    std::cout << "(all same bank!)" << std::endl;

    std::cout << "\n  With swizzle - XOR spreads accesses:" << std::endl;
    std::cout << "    Threads 0-7 access offsets: ";
    for (int t = 0; t < 8; ++t) {
        int offset = swizzled(t, 0);
        std::cout << offset << " ";
    }
    std::cout << std::endl;

    std::cout << "\n✓ Swizzling reduces bank conflicts in shared memory" << std::endl;
    std::cout << "  Use this pattern when multiple threads access same column" << std::endl;
}

// Demonstrate flatten
void test_flatten() {
    print_separator("Test 7: Flatten Multi-Dimensional Layout");

    // Multi-dimensional layout
    auto layout_2d = make_layout(make_shape(Int<4>{}, Int<8>{}),
                                 make_stride(Int<8>{}, Int<1>{}));

    std::cout << "2D layout: " << layout_2d << std::endl;
    std::cout << "  Rank: " << rank(layout_2d) << std::endl;
    std::cout << "  Size: " << size(layout_2d) << std::endl;

    // Flatten to 1D
    auto flattened = make_layout(size(layout_2d));

    std::cout << "\nFlattened to 1D: " << flattened << std::endl;
    std::cout << "  Rank: " << rank(flattened) << std::endl;
    std::cout << "  Size: " << size(flattened) << std::endl;

    std::cout << "\nUse case: Convert multi-dimensional indexing to linear indexing" << std::endl;
}

int main() {
    print_separator("CuTe Layout Operations and Transformations");

    try {
        test_composition();
        test_2d_composition();
        test_logical_divide();
        test_block_thread_decomposition();
        test_complement();
        test_swizzle();
        test_flatten();

        print_separator("All Tests Completed Successfully!");

        std::cout << "\nKey Takeaways:" << std::endl;
        std::cout << "  1. composition: Combine layouts (function composition)" << std::endl;
        std::cout << "  2. logical_divide: Split layout into tiles" << std::endl;
        std::cout << "  3. complement: Find orthogonal coordinates" << std::endl;
        std::cout << "  4. Swizzle: Avoid bank conflicts with XOR pattern" << std::endl;
        std::cout << "  5. Block-thread decomposition: Common pattern for GEMM" << std::endl;

        std::cout << "\nNext: Proceed to example 03_cute_gemm to apply these concepts!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
