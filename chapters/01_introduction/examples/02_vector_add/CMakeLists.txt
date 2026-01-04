cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(vector_add LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(vector_add vector_add.cu)

set_target_properties(vector_add PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;80;86"
)

target_link_libraries(vector_add CUDA::cudart)
