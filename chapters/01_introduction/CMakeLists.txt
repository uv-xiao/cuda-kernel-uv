cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(chapter01_introduction LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Print CUDA information
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Toolkit Root: ${CUDAToolkit_TARGET_DIR}")

# Set CUDA architectures
# Adjust these based on your GPU
# Common values:
#   - 60,61: Pascal (GTX 10xx, Tesla P100)
#   - 70,75: Volta/Turing (V100, RTX 20xx)
#   - 80,86: Ampere (A100, RTX 30xx)
#   - 89,90: Ada/Hopper (RTX 40xx, H100)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86")
endif()
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Enable verbose output for debugging
option(CMAKE_VERBOSE_MAKEFILE "Verbose build output" OFF)

# ============================================================================
# Examples
# ============================================================================

# Example 01: Hello CUDA
add_subdirectory(examples/01_hello_cuda)

# Example 02: Vector Addition
add_subdirectory(examples/02_vector_add)

# Example 03: Device Query
add_subdirectory(examples/03_device_query)

# ============================================================================
# Exercises
# ============================================================================

# Exercise 01: Vector Subtraction - Starter
add_executable(vector_subtract_starter
    exercises/01_vector_subtract/starter.cu
)
set_target_properties(vector_subtract_starter PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)
target_link_libraries(vector_subtract_starter CUDA::cudart)

# Exercise 01: Vector Subtraction - Solution
add_executable(vector_subtract_solution
    exercises/01_vector_subtract/solution.cu
)
set_target_properties(vector_subtract_solution PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)
target_link_libraries(vector_subtract_solution CUDA::cudart)

# Exercise 02: SAXPY - Starter
add_executable(saxpy_starter
    exercises/02_saxpy/starter.cu
)
set_target_properties(saxpy_starter PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)
target_link_libraries(saxpy_starter CUDA::cudart)

# Exercise 02: SAXPY - Solution
add_executable(saxpy_solution
    exercises/02_saxpy/solution.cu
)
set_target_properties(saxpy_solution PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)
target_link_libraries(saxpy_solution CUDA::cudart)

# ============================================================================
# Installation (optional)
# ============================================================================

# Install executables
install(TARGETS
    hello_cuda
    vector_add
    device_query
    vector_subtract_solution
    saxpy_solution
    RUNTIME DESTINATION bin
)

# ============================================================================
# Custom targets for convenience
# ============================================================================

# Build all examples
add_custom_target(examples
    DEPENDS hello_cuda vector_add device_query
)

# Build all exercise solutions
add_custom_target(solutions
    DEPENDS vector_subtract_solution saxpy_solution
)

# Build everything
add_custom_target(all_chapter01
    DEPENDS examples solutions
)

# Run all examples (for testing)
add_custom_target(run_examples
    COMMAND echo "Running Hello CUDA..."
    COMMAND $<TARGET_FILE:hello_cuda>
    COMMAND echo ""
    COMMAND echo "Running Vector Addition..."
    COMMAND $<TARGET_FILE:vector_add> 100000
    COMMAND echo ""
    COMMAND echo "Running Device Query..."
    COMMAND $<TARGET_FILE:device_query>
    DEPENDS examples
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# ============================================================================
# Print build information
# ============================================================================

message(STATUS "")
message(STATUS "Chapter 01: GPU & CUDA Introduction")
message(STATUS "------------------------------------")
message(STATUS "Examples:")
message(STATUS "  - hello_cuda: Basic kernel and thread hierarchy")
message(STATUS "  - vector_add: Complete CUDA workflow")
message(STATUS "  - device_query: GPU properties and capabilities")
message(STATUS "")
message(STATUS "Exercises:")
message(STATUS "  - vector_subtract: Element-wise subtraction (starter + solution)")
message(STATUS "  - saxpy: Y = α·X + Y operation (starter + solution)")
message(STATUS "")
message(STATUS "Build targets:")
message(STATUS "  make                    - Build all")
message(STATUS "  make examples           - Build examples only")
message(STATUS "  make solutions          - Build exercise solutions only")
message(STATUS "  make run_examples       - Build and run all examples")
message(STATUS "")
message(STATUS "After building:")
message(STATUS "  ./examples/01_hello_cuda/hello_cuda")
message(STATUS "  ./examples/02_vector_add/vector_add")
message(STATUS "  ./examples/03_device_query/device_query")
message(STATUS "  ./exercises/01_vector_subtract/vector_subtract_solution")
message(STATUS "  ./exercises/02_saxpy/saxpy_solution")
message(STATUS "")
