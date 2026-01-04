cmake_minimum_required(VERSION 3.18)

project(reduction LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# Add executable
add_executable(reduction reduction.cu)

# Set CUDA architectures
set_target_properties(reduction PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86;89;90"
)

# Add compiler flags
target_compile_options(reduction PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

# Print build information
message(STATUS "Building Reduction Example")
message(STATUS "CUDA Architectures: 70;75;80;86;89;90")
