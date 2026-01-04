cmake_minimum_required(VERSION 3.18)

# Warp primitives examples
add_executable(warp_shuffle warp_shuffle.cu)
add_executable(warp_vote warp_vote.cu)

# Set CUDA architectures
set_target_properties(warp_shuffle warp_vote PROPERTIES
    CUDA_ARCHITECTURES "60;70;75;80;86;89;90"
    CUDA_SEPARABLE_COMPILATION ON
)

# Set C++ standard
target_compile_features(warp_shuffle PUBLIC cxx_std_11)
target_compile_features(warp_vote PUBLIC cxx_std_11)
