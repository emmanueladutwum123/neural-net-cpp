cmake_minimum_required(VERSION 3.16)
project(NeuralNetCPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

# -O3 without -march=native (which ties the binary to the exact CPU that built it —
# a portability problem for anyone else building this from source) or -ffast-math
# (relaxes IEEE 754 compliance in ways that can silently interact badly with the
# epsilon-based numerical stability tricks used in Adam/BatchNorm/softmax).
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -Wall -Wextra)
endif()

include_directories(include)

# mnist_loader shared object
add_library(mnist_loader STATIC src/mnist_loader.cpp)
target_include_directories(mnist_loader PUBLIC include)

# Training executable
add_executable(train src/train.cpp)
target_link_libraries(train mnist_loader)

# Benchmark executable
add_executable(benchmark src/benchmark.cpp)

# Tests using Google Test (if available)
find_package(GTest QUIET)
if(GTest_FOUND)
    enable_testing()
    add_executable(test_nn tests/test_nn.cpp)
    target_link_libraries(test_nn GTest::GTest GTest::Main mnist_loader)
    gtest_discover_tests(test_nn)
else()
    message(STATUS "GTest not found — skipping unit tests")
endif()
