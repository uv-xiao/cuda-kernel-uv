cmake_minimum_required(VERSION 3.18)

project(histogram LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# Add executable
add_executable(histogram histogram.cu)

# Set CUDA architectures
set_target_properties(histogram PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86;89;90"
)

# Add compiler flags
target_compile_options(histogram PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# Print build information
message(STATUS "Building Histogram Example")
message(STATUS "CUDA Architectures: 70;75;80;86;89;90")
