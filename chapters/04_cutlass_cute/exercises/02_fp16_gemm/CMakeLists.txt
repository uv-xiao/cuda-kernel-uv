cmake_minimum_required(VERSION 3.18)

# Check compute capability for Tensor Cores
if(CMAKE_CUDA_ARCHITECTURES LESS 70)
    message(WARNING
        "FP16 GEMM requires SM70+ (Volta or newer) for Tensor Cores.\n"
        "Current architecture: SM${CMAKE_CUDA_ARCHITECTURES}\n"
        "Skipping FP16 GEMM exercise build.")
    return()
endif()

# FP16 GEMM starter
add_executable(fp16_gemm_starter starter.cu)
target_link_libraries(fp16_gemm_starter CUDA::cudart CUDA::cublas)
set_target_properties(fp16_gemm_starter PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/exercises
)

# FP16 GEMM solution
add_executable(fp16_gemm_solution solution.cu)
target_link_libraries(fp16_gemm_solution CUDA::cudart CUDA::cublas)
set_target_properties(fp16_gemm_solution PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/exercises
)

message(STATUS "Building Exercise 02: FP16 GEMM with Tensor Cores")
