cmake_minimum_required(VERSION 3.18)
project(sparse_patterns CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

# Local attention
add_executable(local_attention local_attention.cu)
target_include_directories(local_attention PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(local_attention ${CUDA_LIBRARIES})

# Strided attention
add_executable(strided_attention strided_attention.cu)
target_include_directories(strided_attention PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(strided_attention ${CUDA_LIBRARIES})
