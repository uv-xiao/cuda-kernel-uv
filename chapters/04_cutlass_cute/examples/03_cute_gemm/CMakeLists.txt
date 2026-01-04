cmake_minimum_required(VERSION 3.18)
project(cute_gemm CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# CUTLASS directory
if(NOT DEFINED CUTLASS_DIR)
    if(DEFINED ENV{CUTLASS_DIR})
        set(CUTLASS_DIR $ENV{CUTLASS_DIR})
    else()
        message(FATAL_ERROR "CUTLASS_DIR not set")
    endif()
endif()

include_directories(${CUTLASS_DIR}/include)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# Build executables
add_executable(gemm_simple gemm_simple.cu)
target_link_libraries(gemm_simple CUDA::cudart CUDA::cublas)

add_executable(gemm_tiled gemm_tiled.cu)
target_link_libraries(gemm_tiled CUDA::cudart CUDA::cublas)

message(STATUS "Building CuTe GEMM examples for SM${CMAKE_CUDA_ARCHITECTURES}")
