cmake_minimum_required(VERSION 3.18)
project(cutlass_setup CUDA CXX)

# CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# CUTLASS directory (can be set via command line: -DCUTLASS_DIR=/path/to/cutlass)
if(NOT DEFINED CUTLASS_DIR)
    if(DEFINED ENV{CUTLASS_DIR})
        set(CUTLASS_DIR $ENV{CUTLASS_DIR})
    else()
        message(FATAL_ERROR
            "CUTLASS_DIR is not set. Please set it via:\n"
            "  cmake -DCUTLASS_DIR=/path/to/cutlass ..\n"
            "or set the environment variable:\n"
            "  export CUTLASS_DIR=/path/to/cutlass")
    endif()
endif()

message(STATUS "Using CUTLASS from: ${CUTLASS_DIR}")

# Verify CUTLASS directory exists
if(NOT EXISTS "${CUTLASS_DIR}/include/cute/tensor.hpp")
    message(FATAL_ERROR
        "CUTLASS not found at ${CUTLASS_DIR}\n"
        "Please ensure CUTLASS is cloned and the path is correct:\n"
        "  git clone https://github.com/NVIDIA/cutlass.git\n"
        "  cmake -DCUTLASS_DIR=/path/to/cutlass ..")
endif()

# Add CUTLASS include directories
include_directories(${CUTLASS_DIR}/include)

# CUDA architecture (adjust based on your GPU)
# SM70: Volta (V100)
# SM75: Turing (RTX 20xx, T4)
# SM80: Ampere (A100, RTX 30xx)
# SM86: Ampere (RTX 30xx mobile)
# SM89: Ada Lovelace (RTX 40xx)
# SM90: Hopper (H100)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)  # Default to Ampere
endif()

message(STATUS "Building for CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

# Build executable
add_executable(test_cutlass test_cutlass.cu)

# Link CUDA runtime
target_link_libraries(test_cutlass CUDA::cudart)

# Set output directory
set_target_properties(test_cutlass PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Print build information
message(STATUS "")
message(STATUS "Build Configuration:")
message(STATUS "  CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "  CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUTLASS Directory: ${CUTLASS_DIR}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
