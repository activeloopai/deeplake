project(json)

file(GLOB_RECURSE SOURCES "*.cpp" "*.c" "*.hpp")

add_library(json ${SOURCES})