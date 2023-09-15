cmake_minimum_required(VERSION 3.26)
project(deeplake)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(FetchContent)

FetchContent_Declare(stduuid URL https://github.com/mariusbancila/stduuid/archive/refs/tags/v1.2.3.zip)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_Declare(pybind11 URL https://github.com/pybind/pybind11/archive/refs/tags/v2.11.1.tar.gz)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest json stduuid pybind11)

find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
find_package(ArrowDataset REQUIRED)

enable_testing()

add_library(deeplog
        deeplog/actions/action.cpp
        deeplog/actions/protocol_action.cpp
        deeplog/actions/metadata_action.cpp
        deeplog/actions/add_file_action.cpp
        deeplog/actions/add_file_action.hpp
        deeplog/actions/create_branch_action.cpp
        deeplog/actions/create_branch_action.hpp
        deeplog/actions/remove_file_action.cpp
        deeplog/actions/remove_file_action.hpp
        deeplog/deeplog.cpp
        deeplog/deeplog.hpp
        deeplog/last_checkpoint.cpp
        deeplog/last_checkpoint.hpp
)

add_executable(
        runTests
        deeplog/actions/protocol_action.test.cpp
        deeplog/actions/metadata_action.test.cpp
        deeplog/actions/add_file_action.test.cpp
        deeplog/actions/create_branch_action.test.cpp
        deeplog/actions/remove_file_action.test.cpp
        deeplog/deeplog.test.cpp
        deeplog/last_checkpoint.test.cpp
)

target_link_libraries(deeplog nlohmann_json::nlohmann_json stduuid Arrow::arrow_static Parquet::parquet_static ArrowDataset::arrow_dataset_static pybind11::module pybind11::lto pybind11::windows_extras)
target_link_libraries(runTests
        deeplog
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(runTests)
