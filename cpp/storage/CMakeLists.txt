project(storage)

include(FetchContent)

file(GLOB_RECURSE SOURCES "*.cpp")

add_library(storage ${SOURCES})

target_link_libraries(storage PUBLIC spdlog::spdlog)
