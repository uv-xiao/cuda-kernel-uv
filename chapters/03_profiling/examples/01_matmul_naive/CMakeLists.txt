cmake_minimum_required(VERSION 3.18)
project(matmul_naive CUDA)

# Set C++ standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add executable
add_executable(matmul_naive matmul_naive.cu)

# Set CUDA architectures (compute capability)
# Adjust based on your GPU: 70 (V100), 75 (T4), 80 (A100), 86 (RTX 30xx)
set_target_properties(matmul_naive PROPERTIES CUDA_ARCHITECTURES "70;75;80;86")

# Optimization flags
target_compile_options(matmul_naive PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
