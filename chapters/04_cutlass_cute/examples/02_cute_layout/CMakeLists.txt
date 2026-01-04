cmake_minimum_required(VERSION 3.18)
project(cute_layout CUDA CXX)

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

# Build executables
add_executable(layout_basics layout_basics.cu)
target_link_libraries(layout_basics CUDA::cudart)

add_executable(layout_operations layout_operations.cu)
target_link_libraries(layout_operations CUDA::cudart)

message(STATUS "Building CuTe Layout examples for SM${CMAKE_CUDA_ARCHITECTURES}")
