cmake_minimum_required(VERSION 3.22)
project(pheonix)

add_subdirectory(./third_party/pybind11)
# find_package(./third_party/eventuals/)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a -fcoroutines")
# file(GLOB_RECURSE SOURCES "*.cpp")

pybind11_add_module(pheonix libpheonix.cpp)

target_link_libraries(pheonix PRIVATE curl aws-cpp-sdk-core aws-cpp-sdk-s3 stdc++)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(pheonix
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})