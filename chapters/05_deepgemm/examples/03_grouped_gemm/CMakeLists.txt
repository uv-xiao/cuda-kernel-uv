cmake_minimum_required(VERSION 3.18)
project(grouped_gemm CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
endif()

add_executable(grouped_gemm grouped_gemm.cu)
target_link_libraries(grouped_gemm CUDA::cudart CUDA::cublas)
set_target_properties(grouped_gemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(variable_sizes variable_sizes.cu)
target_link_libraries(variable_sizes CUDA::cudart)
set_target_properties(variable_sizes PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(grouped_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>
)

target_compile_options(variable_sizes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>
)
