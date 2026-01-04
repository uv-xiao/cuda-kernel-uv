cmake_minimum_required(VERSION 3.18)
project(deepgemm_chapter CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Detect GPU architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Default to common architectures
    # 80: A100, 86: RTX 30-series, 89: RTX 40-series, 90: H100
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
endif()

# Include subdirectories with examples
add_subdirectory(examples/01_fp8_basics)
add_subdirectory(examples/02_quantization)
add_subdirectory(examples/03_grouped_gemm)

# Exercise 1: Simple Grouped GEMM
add_executable(grouped_gemm_starter exercises/01_simple_grouped_gemm/starter.cu)
target_link_libraries(grouped_gemm_starter CUDA::cudart)
set_target_properties(grouped_gemm_starter PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/exercises"
)

add_executable(grouped_gemm_solution exercises/01_simple_grouped_gemm/solution.cu)
target_link_libraries(grouped_gemm_solution CUDA::cudart)
set_target_properties(grouped_gemm_solution PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/exercises"
)

# Compiler flags
target_compile_options(grouped_gemm_starter PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>
)

target_compile_options(grouped_gemm_solution PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>
)

# Print configuration
message(STATUS "DeepGEMM Chapter Configuration:")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
