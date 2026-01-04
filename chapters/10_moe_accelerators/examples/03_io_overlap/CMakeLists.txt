cmake_minimum_required(VERSION 3.18)
project(IOOverlap CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")

# Target SM90 for TMA support (H100/H200)
# Fall back to SM80 (A100) for basic async memory
set(CMAKE_CUDA_ARCHITECTURES 80 90)

# IO overlap example
add_executable(io_overlap io_overlap.cu)
target_link_libraries(io_overlap PRIVATE CUDA::cudart)
set_target_properties(io_overlap PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Pipelined MoE example
add_executable(pipelined_moe pipelined_moe.cu)
target_link_libraries(pipelined_moe PRIVATE CUDA::cudart)
set_target_properties(pipelined_moe PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

message(STATUS "Building IO Overlap examples")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Note: TMA features require SM90+ (H100/H200)")
