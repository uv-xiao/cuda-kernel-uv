cmake_minimum_required(VERSION 3.18)
project(matmul_tiled CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(matmul_tiled matmul_tiled.cu)

set_target_properties(matmul_tiled PROPERTIES CUDA_ARCHITECTURES "70;75;80;86")

target_compile_options(matmul_tiled PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
