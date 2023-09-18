cmake_minimum_required(VERSION 3.16)
project(deeplake)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
	cmake_policy(SET CMP0135 NEW)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(PYTHON_EXECUTABLE /opt/homebrew/opt/python@3.10/bin/python3.10)

if(APPLE)
    set (CMAKE_OSX_DEPLOYMENT_TARGET 10.15)
else()
    #skip multi architecture build for linux
    set (CMAKE_OSX_ARCHITECTURES)
endif()

set(DEFAULT_PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindStduuid.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindJson.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindPybind11.cmake)
include(${DEFAULT_PARENT_DIR}/cmake/FindGoogleTest.cmake)

add_subdirectory(deeplog)
add_subdirectory(tests)
add_subdirectory(py_api)

