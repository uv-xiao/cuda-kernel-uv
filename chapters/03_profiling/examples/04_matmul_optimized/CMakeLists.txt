cmake_minimum_required(VERSION 3.18)
project(matmul_optimized CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(matmul_optimized matmul_optimized.cu)

set_target_properties(matmul_optimized PROPERTIES CUDA_ARCHITECTURES "70;75;80;86")

target_compile_options(matmul_optimized PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>  # Print register usage
)
