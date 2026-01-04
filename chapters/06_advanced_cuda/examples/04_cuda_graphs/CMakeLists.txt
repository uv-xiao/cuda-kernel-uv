cmake_minimum_required(VERSION 3.18)

# CUDA Graphs examples
add_executable(graph_basics graph_basics.cu)
add_executable(graph_capture graph_capture.cu)

# Set CUDA architectures
set_target_properties(graph_basics graph_capture PROPERTIES
    CUDA_ARCHITECTURES "60;70;75;80;86;89;90"
    CUDA_SEPARABLE_COMPILATION ON
)

# Set C++ standard
target_compile_features(graph_basics PUBLIC cxx_std_11)
target_compile_features(graph_capture PUBLIC cxx_std_11)
