cmake_minimum_required(VERSION 3.18)

add_executable(prefix_scan prefix_scan.cu)

set_target_properties(prefix_scan PROPERTIES
    CUDA_ARCHITECTURES "60;70;75;80;86;89;90"
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_features(prefix_scan PUBLIC cxx_std_11)
