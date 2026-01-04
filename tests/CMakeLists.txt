# Test CMakeLists.txt

# Note: Most tests are in individual chapter directories
# This file provides a unified test target

# Add a simple compilation test
add_executable(test_common test_common.cu)
target_link_libraries(test_common cuda_tutorial_common)

# Register tests with CTest
add_test(NAME test_common COMMAND test_common)

# Custom target to run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_common
    COMMENT "Running all tests"
)
