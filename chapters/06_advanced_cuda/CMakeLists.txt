cmake_minimum_required(VERSION 3.18)
project(Chapter06_AdvancedCUDA CUDA CXX)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Add subdirectories for examples
add_subdirectory(examples/01_warp_primitives)
add_subdirectory(examples/02_warp_reduction)
add_subdirectory(examples/03_cooperative_groups)
add_subdirectory(examples/04_cuda_graphs)
add_subdirectory(examples/05_scan)

# Note: Exercises are built separately as they are learning materials
# To build exercises, navigate to each exercise directory and compile manually:
#   cd exercises/01_warp_reduce && nvcc solution.cu -o warp_reduce_max
#   cd exercises/02_block_scan && nvcc solution.cu -o block_scan
