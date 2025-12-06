include(FetchContent)

set(lz4_TAG v1.9.4)
set(lz4_URL https://github.com/lz4/lz4/archive/refs/tags/${lz4_TAG}.zip)
set(lz4_SOURCE_DIR ${DEFAULT_PARENT_DIR}/.ext/lz4)
set(lz4_SHA256_CHECKSUM 37e63d56fb9cbe2e430c7f737a404cd4b98637b05e1467459d5c8fe1a4364cc3)
set(lz4_INCLUDE_DIR ${lz4_SOURCE_DIR})
set(lz4_BUILD_INSTALL_PREFIX ${DEFAULT_PARENT_DIR}/.ext)
set(lz4_INSTALL_DIR ${lz4_BUILD_INSTALL_PREFIX}/lib)

find_path(
    lz4_LOCAL
    NAMES lib/lz4.h
    PATHS ${lz4_SOURCE_DIR})

if(${lz4_LOCAL} STREQUAL lz4_LOCAL-NOTFOUND)
    message(STATUS "Adding LZ4 to project")
    FetchContent_Declare(
        lz4
        URL ${lz4_URL}
        URL_HASH SHA256=${lz4_SHA256_CHECKSUM}
        SOURCE_DIR ${lz4_SOURCE_DIR})

    FetchContent_MakeAvailable(lz4)
endif()

file(GLOB_RECURSE SOURCES "${lz4_SOURCE_DIR}/lib/*.cpp" "${lz4_SOURCE_DIR}/lib/*.c")

include_directories(${lz4_SOURCE_DIR}/lib/)
add_library(lz4_static STATIC ${SOURCES})
set_property(TARGET lz4_static PROPERTY POSITION_INDEPENDENT_CODE ON)
