cmake_minimum_required(VERSION 3.18)

project(2d_indexing LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# Add executable
add_executable(indexing_2d indexing_2d.cu)

# Set CUDA architectures
set_target_properties(indexing_2d PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86;89;90"
)

# Add compiler flags for better error reporting
target_compile_options(indexing_2d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# Print build information
message(STATUS "Building 2D Indexing Example")
message(STATUS "CUDA Architectures: 70;75;80;86;89;90")
