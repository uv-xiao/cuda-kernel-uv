cmake_minimum_required(VERSION 3.18)
project(flash_attention_minimal CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA architectures (adjust for your GPU)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall")

# Add executable
add_executable(flash_fwd flash_fwd.cu)

# Set include directories
target_include_directories(flash_fwd PRIVATE ${CUDA_INCLUDE_DIRS})

# Link libraries
target_link_libraries(flash_fwd ${CUDA_LIBRARIES})

# Print configuration info
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
