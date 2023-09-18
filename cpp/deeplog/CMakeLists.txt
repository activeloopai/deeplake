project(deeplog)

include(FetchContent)

FetchContent_Declare(stduuid URL https://github.com/mariusbancila/stduuid/archive/refs/tags/v1.2.3.zip)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest stduuid)

find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
find_package(ArrowDataset REQUIRED)

file(GLOB_RECURSE SOURCES "*.cpp")

add_library(deeplog ${SOURCES})

target_link_libraries(deeplog PUBLIC stduuid Arrow::arrow_static Parquet::parquet_static ArrowDataset::arrow_dataset_static)
