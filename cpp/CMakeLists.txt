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
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindJson.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindPybind11.cmake)

# file(GLOB_RECURSE SOURCES "cpp/deeplog/*.cpp")
# pybind11_add_module(deeplog ${SOURCES})


#TODO revisit the external dependencies
include(FetchContent)

# FetchContent_Declare(stduuid URL https://github.com/mariusbancila/stduuid/archive/refs/tags/v1.2.3.zip)
# FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
# FetchContent_Declare(
#         googletest
#         URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
# )
# FetchContent_Declare(pybind11 URL https://github.com/pybind/pybind11/archive/refs/tags/v2.11.1.tar.gz)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# FetchContent_MakeAvailable(googletest stduuid)

enable_testing()

# add_executable(
#         runTests
#         tests/deeplog/actions/protocol_action_test.cpp
#         tests/deeplog/actions/metadata_action_test.cpp
#         tests/deeplog/actions/add_file_action_test.cpp
#         tests/deeplog/actions/create_branch_action_test.cpp
#         tests/deeplog/actions/remove_file_action_test.cpp
#         tests/deeplog/deeplog_test.cpp
#         tests/deeplog/last_checkpoint_test.cpp
# )

add_subdirectory(deeplog)

pybind11_add_module(_deeplake
    py_api/deeplog.cpp
)

# target_link_libraries(deeplog PUBLIC nlohmann_json::nlohmann_json stduuid Arrow::arrow_static Parquet::parquet_static ArrowDataset::arrow_dataset_static)
# target_link_libraries(runTests
#         deeplog
#         GTest::gtest_main
# )

target_link_libraries(_deeplake PUBLIC deeplog)

# include(GoogleTest)
# gtest_discover_tests(runTests)
