include(ExternalProject)

set(duckdb_SOURCE_DIR ${DEFAULT_PARENT_DIR}/.ext/duckdb)

set(duckdb_URL https://github.com/duckdb/duckdb.git)
set(duckdb_TAG v1.4.1)
set(duckdb_INSTALL ${DEFAULT_PARENT_DIR}/.ext/duckdb-build/install)

set(DUCKDB_LIB_PATH ${duckdb_INSTALL}/lib/libduckdb_static.a)
if(WIN32)
    set(DUCKDB_LIB_PATH ${duckdb_INSTALL}/lib/duckdb_static.lib)
endif()


if(EXISTS ${DUCKDB_LIB_PATH})
    message(STATUS "DuckDB already built, skipping rebuild")
    add_library(duckdb_static STATIC IMPORTED)
    set_target_properties(duckdb_static PROPERTIES
        IMPORTED_LOCATION ${DUCKDB_LIB_PATH}
        INTERFACE_INCLUDE_DIRECTORIES ${duckdb_INSTALL}/include
    )

else()
    if (NOT TARGET duckdb_ep)
        message(STATUS "Adding duckdb external project")

        ExternalProject_Add(
            duckdb_ep
            GIT_REPOSITORY ${duckdb_URL}
            GIT_TAG ${duckdb_TAG}
            INSTALL_DIR ${duckdb_INSTALL}
            SOURCE_DIR ${duckdb_SOURCE_DIR}
            CMAKE_ARGS
                -DCMAKE_BUILD_TYPE:STRING=Release
                -DBUILD_UNITTESTS:BOOL=FALSE
                -DENABLE_UNITTEST_CPP_TESTS:BOOL=FALSE
                -DBUILD_SHELL:BOOL=FALSE
                -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                -DBUILD_SHARED_LIBS=OFF
                -DCMAKE_INSTALL_LIBDIR=lib
                -DCMAKE_INSTALL_PREFIX:STRING=${duckdb_INSTALL}
            LOG_CONFIGURE ON
            LOG_BUILD ON
            LOG_INSTALL ON
            LOG_DOWNLOAD ON
            LOG_OUTPUT_ON_FAILURE ON
        )

    endif()
endif()

link_directories(${duckdb_INSTALL}/lib)
