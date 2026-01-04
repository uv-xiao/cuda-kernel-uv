cmake_minimum_required(VERSION 3.18)
project(cutlass_cute_chapter CUDA CXX)

# CUDA and C++ standards
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# CUTLASS directory
if(NOT DEFINED CUTLASS_DIR)
    if(DEFINED ENV{CUTLASS_DIR})
        set(CUTLASS_DIR $ENV{CUTLASS_DIR})
    else()
        message(WARNING
            "CUTLASS_DIR is not set. Please set it via:\n"
            "  cmake -DCUTLASS_DIR=/path/to/cutlass ..\n"
            "or set the environment variable:\n"
            "  export CUTLASS_DIR=/path/to/cutlass\n"
            "Trying to find CUTLASS in common locations...")

        # Try common locations
        set(CUTLASS_SEARCH_PATHS
            "$ENV{HOME}/cutlass"
            "$ENV{HOME}/libraries/cutlass"
            "/opt/cutlass"
            "/usr/local/cutlass"
        )

        foreach(search_path ${CUTLASS_SEARCH_PATHS})
            if(EXISTS "${search_path}/include/cute/tensor.hpp")
                set(CUTLASS_DIR ${search_path})
                message(STATUS "Found CUTLASS at: ${CUTLASS_DIR}")
                break()
            endif()
        endforeach()

        if(NOT DEFINED CUTLASS_DIR)
            message(FATAL_ERROR
                "CUTLASS not found. Please install:\n"
                "  git clone https://github.com/NVIDIA/cutlass.git ~/cutlass\n"
                "Then run:\n"
                "  cmake -DCUTLASS_DIR=~/cutlass ..")
        endif()
    endif()
endif()

message(STATUS "Using CUTLASS from: ${CUTLASS_DIR}")

# Verify CUTLASS installation
if(NOT EXISTS "${CUTLASS_DIR}/include/cute/tensor.hpp")
    message(FATAL_ERROR
        "CUTLASS installation incomplete at ${CUTLASS_DIR}\n"
        "Missing: ${CUTLASS_DIR}/include/cute/tensor.hpp\n"
        "Please ensure CUTLASS is properly cloned:\n"
        "  git clone https://github.com/NVIDIA/cutlass.git\n"
        "  cd cutlass && git checkout v3.4.0")
endif()

# Add CUTLASS include directory
include_directories(${CUTLASS_DIR}/include)

# CUDA architecture (adjust based on your GPU)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Auto-detect GPU architecture
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE GPU_COMPUTE_CAP
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    if(GPU_COMPUTE_CAP)
        string(REPLACE "." "" GPU_ARCH ${GPU_COMPUTE_CAP})
        set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCH})
        message(STATUS "Auto-detected GPU architecture: SM${GPU_ARCH}")
    else()
        set(CMAKE_CUDA_ARCHITECTURES 80)
        message(STATUS "Using default architecture: SM80 (Ampere)")
    endif()
endif()

message(STATUS "Building for CUDA architecture: SM${CMAKE_CUDA_ARCHITECTURES}")

# CUDA compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# Optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
else()
    set(CMAKE_BUILD_TYPE "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

# Add subdirectories for examples
add_subdirectory(examples/01_cutlass_setup)
add_subdirectory(examples/02_cute_layout)
add_subdirectory(examples/03_cute_gemm)
add_subdirectory(examples/04_tensor_cores)

# Add subdirectories for exercises
# Exercise CMakeLists
add_subdirectory(exercises/01_batched_gemm)
add_subdirectory(exercises/02_fp16_gemm)

# Print build summary
message(STATUS "")
message(STATUS "==================== Build Configuration ====================")
message(STATUS "CUTLASS Directory: ${CUTLASS_DIR}")
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architecture: SM${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS}")
message(STATUS "=============================================================")
message(STATUS "")
