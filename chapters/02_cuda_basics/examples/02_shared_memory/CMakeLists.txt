cmake_minimum_required(VERSION 3.18)

project(shared_memory LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# Add executable
add_executable(shared_mem shared_mem.cu)

# Set CUDA architectures
set_target_properties(shared_mem PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86;89;90"
)

# Add compiler flags
target_compile_options(shared_mem PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)

# Print build information
message(STATUS "Building Shared Memory Example")
message(STATUS "CUDA Architectures: 70;75;80;86;89;90")
