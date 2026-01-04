cmake_minimum_required(VERSION 3.18)
project(GroupedGEMM CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)  # A100, RTX4090, H100, H200

# Naive implementation
add_executable(grouped_gemm_naive grouped_gemm_naive.cu)
target_link_libraries(grouped_gemm_naive PRIVATE CUDA::cublas CUDA::cudart)
set_target_properties(grouped_gemm_naive PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Tiled implementation
add_executable(grouped_gemm_tiled grouped_gemm_tiled.cu)
target_link_libraries(grouped_gemm_tiled PRIVATE CUDA::cudart)
set_target_properties(grouped_gemm_tiled PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Print build info
message(STATUS "Building Grouped GEMM examples")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
