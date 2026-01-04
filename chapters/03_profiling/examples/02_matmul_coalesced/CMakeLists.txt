cmake_minimum_required(VERSION 3.18)
project(matmul_coalesced CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(matmul_coalesced matmul_coalesced.cu)

set_target_properties(matmul_coalesced PROPERTIES CUDA_ARCHITECTURES "70;75;80;86")

target_compile_options(matmul_coalesced PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
