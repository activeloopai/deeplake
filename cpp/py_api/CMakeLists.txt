project(py_api)

include(FetchContent)

file(GLOB_RECURSE SOURCES "*.cpp")

pybind11_add_module(py_api ${SOURCES})

target_link_libraries(py_api PUBLIC deeplog)
set_target_properties(py_api PROPERTIES OUTPUT_NAME "_deeplake")

add_custom_command(TARGET py_api POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy
        "$<TARGET_FILE:py_api>"
        "${PYTHON_SOURCE}/$<TARGET_FILE_NAME:py_api>"
        COMMENT "Copying module to python source directory")

add_custom_command(TARGET py_api POST_BUILD
        WORKING_DIRECTORY "${PYTHON_SOURCE}"
        COMMAND "${PYTHON_EXECUTABLE}" -m pybind11_stubgen _deeplake -o .
        COMMENT "Generating stubs")