project(deeplog)

include(FetchContent)

find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
find_package(ArrowDataset REQUIRED)

file(GLOB_RECURSE SOURCES "*.cpp")

add_library(deeplog ${SOURCES})

target_link_libraries(deeplog PUBLIC stduuid nlohmann_json::nlohmann_json Arrow::arrow_static Parquet::parquet_static ArrowDataset::arrow_dataset_static)
