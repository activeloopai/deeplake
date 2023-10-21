include(FetchContent)

set(pybind11_URL https://github.com/pybind/pybind11.git)
set(pybind11_TAG v2.11.1)
set(pybind11_SOURCE_DIR ${DEFAULT_PARENT_DIR}/external/pybind11)

FetchContent_Declare(pybind11
    GIT_REPOSITORY ${pybind11_URL}
    GIT_TAG ${pybind11_TAG}
    SOURCE_DIR ${pybind11_SOURCE_DIR}
)

FetchContent_MakeAvailable(pybind11)
