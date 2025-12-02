include(FetchContent)

set(pybind11_URL "https://github.com/pybind/pybind11/archive/refs/tags/v2.13.4.zip")

set(pybind11_SOURCE_DIR "${DEFAULT_PARENT_DIR}/.ext/pybind11")

FetchContent_Declare(
    pybind11
    URL "${pybind11_URL}"
    URL_HASH SHA256=1ad07926d387986c84ba06d66ecabd54d5598b7925eab028c278ecd8d97c3385
    SOURCE_DIR "${pybind11_SOURCE_DIR}")

FetchContent_MakeAvailable(pybind11)
