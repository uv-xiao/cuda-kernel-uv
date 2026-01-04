cmake_minimum_required(VERSION 3.18)

project(Chapter02_CUDA_Basics LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable verbose output
set(CMAKE_VERBOSE_MAKEFILE ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures (support common GPU architectures)
# Volta (70), Turing (75), Ampere (80, 86), Ada (89), Hopper (90)
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")

# Add compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "===========================================")
message(STATUS "Chapter 02: CUDA Basics & Memory")
message(STATUS "===========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "")

# Add example subdirectories
add_subdirectory(examples/01_2d_indexing)
add_subdirectory(examples/02_shared_memory)
add_subdirectory(examples/03_reduction)
add_subdirectory(examples/04_histogram)

message(STATUS "Examples configured:")
message(STATUS "  - 01_2d_indexing")
message(STATUS "  - 02_shared_memory")
message(STATUS "  - 03_reduction")
message(STATUS "  - 04_histogram")
message(STATUS "")

# Exercises are built separately (not part of main build)
# Students should build them individually

message(STATUS "Exercises available (build separately):")
message(STATUS "  - exercises/01_matrix_transpose")
message(STATUS "  - exercises/02_dot_product")
message(STATUS "")
message(STATUS "To build exercises:")
message(STATUS "  cd exercises/01_matrix_transpose")
message(STATUS "  nvcc -O3 starter.cu -o transpose")
message(STATUS "")
message(STATUS "===========================================")
