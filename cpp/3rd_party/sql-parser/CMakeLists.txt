project(hsql)

file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.c")

ADD_LIBRARY(hsql ${SOURCES})

target_link_libraries(hsql PRIVATE fmt::fmt-header-only)