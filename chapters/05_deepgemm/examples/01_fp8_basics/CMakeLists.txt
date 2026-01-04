cmake_minimum_required(VERSION 3.18)
project(fp8_basics CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Detect GPU architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
endif()

# FP8 types example
add_executable(fp8_types fp8_types.cu)
target_link_libraries(fp8_types CUDA::cudart)
set_target_properties(fp8_types PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# FP8 conversion example
add_executable(fp8_conversion fp8_conversion.cu)
target_link_libraries(fp8_conversion CUDA::cudart)
set_target_properties(fp8_conversion PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Compiler flags for optimal performance
target_compile_options(fp8_types PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -lineinfo
    >
)

target_compile_options(fp8_conversion PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -lineinfo
    >
)

# Enable warnings
target_compile_options(fp8_types PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
)

target_compile_options(fp8_conversion PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
)
