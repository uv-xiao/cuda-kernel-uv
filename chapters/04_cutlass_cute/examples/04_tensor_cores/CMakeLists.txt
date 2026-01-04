cmake_minimum_required(VERSION 3.18)
project(tensor_cores CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

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

add_executable(wmma_gemm wmma_gemm.cu)
target_link_libraries(wmma_gemm CUDA::cudart CUDA::cublas)

add_executable(mma_gemm mma_gemm.cu)
target_link_libraries(mma_gemm CUDA::cudart)

message(STATUS "Building Tensor Core examples for SM${CMAKE_CUDA_ARCHITECTURES}")
