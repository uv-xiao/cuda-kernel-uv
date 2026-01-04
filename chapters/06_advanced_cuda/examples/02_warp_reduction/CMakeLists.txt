cmake_minimum_required(VERSION 3.18)

# Warp reduction examples
add_executable(warp_reduce warp_reduce.cu)

# Set CUDA architectures
set_target_properties(warp_reduce PROPERTIES
    CUDA_ARCHITECTURES "60;70;75;80;86;89;90"
    CUDA_SEPARABLE_COMPILATION ON
)

# Set C++ standard
target_compile_features(warp_reduce PUBLIC cxx_std_11)
