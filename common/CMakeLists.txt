# Common utilities library

# Header-only library for common utilities
add_library(cuda_tutorial_common INTERFACE)
target_include_directories(cuda_tutorial_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Optional: Build utilities as a static library if needed
# add_library(cuda_utils STATIC src/utils.cu)
# target_include_directories(cuda_utils PUBLIC include)
