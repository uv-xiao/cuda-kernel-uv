cmake_minimum_required(VERSION 3.18)
project(flash_attention_v2 CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)

# Aggressive optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-O3,-march=native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --maxrregcount=128")

add_executable(flash_v2 flash_v2.cu)
target_include_directories(flash_v2 PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(flash_v2 ${CUDA_LIBRARIES})
