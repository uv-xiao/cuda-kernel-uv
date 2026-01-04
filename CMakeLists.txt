cmake_minimum_required(VERSION 3.18)
project(cuda-kernel-tutorial LANGUAGES CXX CUDA)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architectures (can be overridden via -DCMAKE_CUDA_ARCHITECTURES)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
endif()

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Optional: Find CUTLASS
option(USE_CUTLASS "Enable CUTLASS support" OFF)
if(USE_CUTLASS)
    if(DEFINED CUTLASS_DIR)
        include_directories(${CUTLASS_DIR}/include)
        include_directories(${CUTLASS_DIR}/tools/util/include)
    else()
        message(STATUS "CUTLASS_DIR not set, looking for system installation")
        find_path(CUTLASS_INCLUDE_DIR cutlass/cutlass.h
            HINTS /usr/local/include $ENV{HOME}/cutlass/include)
        if(CUTLASS_INCLUDE_DIR)
            include_directories(${CUTLASS_INCLUDE_DIR})
        else()
            message(WARNING "CUTLASS not found. Chapter 04+ examples may not build.")
        endif()
    endif()
endif()

# Include common utilities
include_directories(${CMAKE_SOURCE_DIR}/common/include)

# Find packages
find_package(CUDAToolkit REQUIRED)

# Add subdirectories for each chapter
add_subdirectory(common)
add_subdirectory(chapters/01_introduction)
add_subdirectory(chapters/02_cuda_basics)
add_subdirectory(chapters/03_profiling)

# Advanced chapters (optional dependencies)
if(USE_CUTLASS)
    add_subdirectory(chapters/04_cutlass_cute)
    add_subdirectory(chapters/05_deepgemm)
endif()

add_subdirectory(chapters/06_advanced_cuda)

# Test targets
enable_testing()
add_subdirectory(tests)

# Custom targets
add_custom_target(format
    COMMAND find ${CMAKE_SOURCE_DIR}/chapters ${CMAKE_SOURCE_DIR}/common
            -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.hpp" |
            xargs clang-format -i
    COMMENT "Formatting source files"
)

add_custom_target(lint
    COMMAND find ${CMAKE_SOURCE_DIR}/chapters ${CMAKE_SOURCE_DIR}/common
            -name "*.cu" -o -name "*.cuh" |
            xargs -I {} nvcc --compiler-options -Wall {} -c -o /dev/null
    COMMENT "Linting CUDA files"
)

# Print configuration
message(STATUS "")
message(STATUS "CUDA Kernel Tutorial Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUTLASS support: ${USE_CUTLASS}")
message(STATUS "")
