project(py_api)

include(FetchContent)

file(GLOB_RECURSE SOURCES "*.cpp")

pybind11_add_module(py_api ${SOURCES})

target_link_libraries(py_api PUBLIC deeplog)
set_target_properties(py_api PROPERTIES OUTPUT_NAME "_deeplake")

