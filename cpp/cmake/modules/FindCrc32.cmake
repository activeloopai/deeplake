set(crc32_BUILD_INSTALL_PREFIX ${DEFAULT_PARENT_DIR}/.ext)
set(crc32_BUILD_VERSION 1.1.2)
set(crc32_URL "https://github.com/google/crc32c/archive/refs/tags/${crc32_BUILD_VERSION}.tar.gz")
set(crc32_SHA256 "ac07840513072b7fcebda6e821068aa04889018f24e10e46181068fb214d7e56")

macro(build_crc32c_once)
    if(NOT TARGET crc32c_ep)
        message(STATUS "Building crc32c from source")
        # Build crc32c
        set(CRC32C_INCLUDE_DIR "${crc32_BUILD_INSTALL_PREFIX}/include")
        set(CRC32C_CMAKE_ARGS
            -DCMAKE_BUILD_TYPE:STRING=Release
            -DCRC32C_BUILD_TESTS=OFF
            -DCRC32C_BUILD_BENCHMARKS=OFF
            -DCRC32C_USE_GLOG=OFF
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_INSTALL_LIBDIR=lib
            -DCMAKE_PREFIX_PATH=${crc32_BUILD_INSTALL_PREFIX}
            -DCMAKE_INSTALL_PREFIX=${crc32_BUILD_INSTALL_PREFIX}
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
            -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET})

        set(_CRC32C_STATIC_LIBRARY
            "${crc32_BUILD_INSTALL_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}crc32c${CMAKE_STATIC_LIBRARY_SUFFIX}")
        set(CRC32C_BUILD_BYPRODUCTS ${_CRC32C_STATIC_LIBRARY})
        set(CRC32C_LIBRARIES crc32c)

        ExternalProject_Add(
            crc32c_ep
            INSTALL_DIR ${crc32_BUILD_INSTALL_PREFIX}
            URL ${crc32_URL}
            URL_HASH SHA256=${crc32_SHA256}
            CMAKE_ARGS ${CRC32C_CMAKE_ARGS}
            BUILD_BYPRODUCTS ${CRC32C_BUILD_BYPRODUCTS}
            LOG_DOWNLOAD ON
            LOG_CONFIGURE ON
            LOG_BUILD ON
            LOG_INSTALL ON
            LOG_OUTPUT_ON_FAILURE ON)

        # Work around https://gitlab.kitware.com/cmake/cmake/issues/15052
        file(MAKE_DIRECTORY "${CRC32C_INCLUDE_DIR}")
        add_library(Crc32c::crc32c STATIC IMPORTED)
        set_target_properties(Crc32c::crc32c PROPERTIES IMPORTED_LOCATION ${_CRC32C_STATIC_LIBRARY}
                                                        INTERFACE_INCLUDE_DIRECTORIES "${CRC32C_INCLUDE_DIR}")
        add_dependencies(Crc32c::crc32c crc32c_ep)
    endif()
endmacro()
