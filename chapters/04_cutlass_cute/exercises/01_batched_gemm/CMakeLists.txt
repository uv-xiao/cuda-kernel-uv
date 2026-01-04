cmake_minimum_required(VERSION 3.18)

# Batched GEMM starter
add_executable(batched_gemm_starter starter.cu)
target_link_libraries(batched_gemm_starter CUDA::cudart CUDA::cublas)
set_target_properties(batched_gemm_starter PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/exercises
)

# Batched GEMM solution
add_executable(batched_gemm_solution solution.cu)
target_link_libraries(batched_gemm_solution CUDA::cudart CUDA::cublas)
set_target_properties(batched_gemm_solution PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/exercises
)

message(STATUS "Building Exercise 01: Batched GEMM")
