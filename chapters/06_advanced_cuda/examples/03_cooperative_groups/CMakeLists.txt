cmake_minimum_required(VERSION 3.18)

# Cooperative groups examples
add_executable(cg_basics cg_basics.cu)
add_executable(grid_sync grid_sync.cu)

# Set CUDA architectures (grid sync requires CC 6.0+)
set_target_properties(cg_basics grid_sync PROPERTIES
    CUDA_ARCHITECTURES "60;70;75;80;86;89;90"
    CUDA_SEPARABLE_COMPILATION ON
)

# Set C++ standard
target_compile_features(cg_basics PUBLIC cxx_std_11)
target_compile_features(grid_sync PUBLIC cxx_std_11)

# Enable relocatable device code for cooperative groups
target_compile_options(cg_basics PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--relocatable-device-code=true>)
target_compile_options(grid_sync PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--relocatable-device-code=true>)
