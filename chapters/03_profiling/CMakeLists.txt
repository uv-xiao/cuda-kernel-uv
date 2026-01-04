cmake_minimum_required(VERSION 3.18)
project(Chapter03_Profiling CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures for different GPU generations
# Adjust based on your GPU:
#   60, 61: Pascal (GTX 10xx)
#   70: Volta (V100)
#   75: Turing (RTX 20xx, T4)
#   80, 86: Ampere (A100, RTX 30xx)
#   89: Ada Lovelace (RTX 40xx)
#   90: Hopper (H100)
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# Optional: Print register usage during compilation
# Uncomment to see resource usage information
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-v")

# Add example subdirectories
add_subdirectory(examples/01_matmul_naive)
add_subdirectory(examples/02_matmul_coalesced)
add_subdirectory(examples/03_matmul_tiled)
add_subdirectory(examples/04_matmul_optimized)

# Print configuration information
message(STATUS "Chapter 03: Profiling & Optimization")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")

# Installation (optional)
# Uncomment if you want to install binaries
# install(TARGETS matmul_naive matmul_coalesced matmul_tiled matmul_optimized
#         RUNTIME DESTINATION bin/chapter03)
