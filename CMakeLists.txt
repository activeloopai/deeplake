cmake_minimum_required(VERSION 3.4...3.18)
project(pheonix)

include_directories(./pheonix/third_party/pybind11/include/)
include_directories(/home/davit/miniconda3/include/python3.9)
include_directories(/usr/include/eigen3/)
add_subdirectory(./pheonix/third_party/pybind11)


# find_package(pybind11 REQUIRED)  # or add_subdirectory(pybind11)



set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a -fcoroutines")
# set(BUILD_TESTING "OFF")
file(GLOB_RECURSE SOURCES "./pheonix/*.cpp" "./pheonix/*.c")


ADD_LIBRARY(pheonix SHARED ${SOURCES})
# pybind11_add_module(pheonix ${SOURCES})

pybind11_extension(pheonix)
pybind11_strip(pheonix)

#set_target_properties(pheonix PROPERTIES CXX_VISIBILITY_PRESET "hidden"
#                                         CUDA_VISIBILITY_PRESET "hidden")
target_link_libraries(pheonix PRIVATE pybind11::module pybind11::lto pybind11::windows_extras)
target_link_libraries(pheonix PRIVATE curl aws-cpp-sdk-core aws-cpp-sdk-s3 stdc++)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(pheonix
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})