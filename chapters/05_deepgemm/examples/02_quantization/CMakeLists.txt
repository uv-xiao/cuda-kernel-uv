cmake_minimum_required(VERSION 3.18)
project(quantization CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
endif()

# Quantization examples
add_executable(quantize quantize.cu)
target_link_libraries(quantize CUDA::cudart)
set_target_properties(quantize PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(fine_grained_scaling fine_grained_scaling.cu)
target_link_libraries(fine_grained_scaling CUDA::cudart)
set_target_properties(fine_grained_scaling PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Compiler optimizations
target_compile_options(quantize PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>
)

target_compile_options(fine_grained_scaling PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>
)
