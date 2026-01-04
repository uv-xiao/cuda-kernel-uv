cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(hello_cuda LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Add executable
add_executable(hello_cuda hello.cu)

# Set CUDA architectures (adjust based on your GPU)
# Common values:
#   - 60,61: Pascal (GTX 10xx, Tesla P100)
#   - 70,75: Volta/Turing (V100, RTX 20xx)
#   - 80,86: Ampere (A100, RTX 30xx)
#   - 89,90: Ada/Hopper (RTX 40xx, H100)
set_target_properties(hello_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;80;86"
)

# Link CUDA runtime
target_link_libraries(hello_cuda CUDA::cudart)

# Enable verbose output for debugging
set(CMAKE_VERBOSE_MAKEFILE ON)
