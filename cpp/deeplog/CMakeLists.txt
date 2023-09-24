project(deeplog)

include(FetchContent)

find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
find_package(ArrowDataset REQUIRED)

file(GLOB_RECURSE SOURCES "*.cpp")

add_library(deeplog ${SOURCES})

target_link_libraries(deeplog PUBLIC stduuid nlohmann_json::nlohmann_json)
target_link_libraries(deeplog PUBLIC "$<IF:$<BOOL:${ARROW_BUILD_STATIC}>,Arrow::arrow_static,Arrow::arrow_shared>")
target_link_libraries(deeplog PUBLIC "$<IF:$<BOOL:${ARROW_BUILD_STATIC}>,Parquet::parquet_static,Parquet::parquet_shared>")
target_link_libraries(deeplog PUBLIC "$<IF:$<BOOL:${ARROW_BUILD_STATIC}>,ArrowDataset::arrow_dataset_static,ArrowDataset::arrow_dataset_shared>")
