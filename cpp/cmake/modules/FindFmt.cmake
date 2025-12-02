include(FetchContent)

set(fmt_URL https://github.com/fmtlib/fmt/releases/download/10.2.1/fmt-10.2.1.zip)
set(fmt_URL_HASH 312151a2d13c8327f5c9c586ac6cf7cddc1658e8f53edae0ec56509c8fa516c9)
set(fmt_SOURCE_DIR ${DEFAULT_PARENT_DIR}/.ext/fmt)
set(fmt_INCLUDE_DIR ${fmt_SOURCE_DIR}/include)


message(STATUS "Adding fmtlib::fmt as an external project")

FetchContent_Declare(
        fmt
        URL ${fmt_URL}
        URL_HASH SHA256=${fmt_URL_HASH}
        SOURCE_DIR ${fmt_SOURCE_DIR})
FetchContent_MakeAvailable(fmt)
include_directories(${fmt_INCLUDE_DIR})
