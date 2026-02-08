option(BUILD_PG_16 "Build PostgreSQL 16 extension" OFF)
option(BUILD_PG_17 "Build PostgreSQL 17 extension" OFF)
option(BUILD_PG_18 "Build PostgreSQL 18 extension" ON)
option(USE_DEEPLAKE_SHARED "Use shared library for deeplake_api (default: auto-detect)" OFF)

set(PG_MODULE deeplake_pg)
set(PG_VERSIONS)

if(BUILD_PG_16)
    list(APPEND PG_VERSIONS 16)
endif()

if(BUILD_PG_17)
    list(APPEND PG_VERSIONS 17)
endif()

if(BUILD_PG_18)
    list(APPEND PG_VERSIONS 18)
endif()

project(${PG_MODULE})
file(GLOB_RECURSE PG_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/${PG_MODULE}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/${PG_MODULE}/*.c")
message(STATUS "PG_MODULE sources found: ${PG_SOURCES}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-register -Wno-deprecated-declarations -Wno-unused-parameter -Wno-unused-variable")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/FindPostgres.cmake)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT WIN32)
    add_compile_options(-fno-omit-frame-pointer)
endif()

if(NOT WIN32)
    # OpenSSL
    find_package(OpenSSL REQUIRED)
    list(APPEND DEEPLAKE_STATIC_LINK_LIBS OpenSSL::SSL OpenSSL::Crypto)
endif()

# Z
find_package(ZLIB REQUIRED)
list(APPEND DEEPLAKE_STATIC_LINK_LIBS ZLIB::ZLIB)

find_package(CURL CONFIG REQUIRED)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/FindDuckDB.cmake)

# Find DeepLake API library
set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/.ext/deeplake_api ${CMAKE_PREFIX_PATH})
find_package(DeepLakeAPI REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/.ext/deeplake_api/lib/cmake/deeplake_api NO_DEFAULT_PATH)

# AWS SDK - required by deeplake_api (symbols not bundled in the prebuilt library)
find_package(AWSSDK COMPONENTS core s3 identity-management)
if(AWSSDK_FOUND)
    message(STATUS "Found AWS SDK: ${AWSSDK_LIBRARIES}")
    list(APPEND DEEPLAKE_STATIC_LINK_LIBS ${AWSSDK_LIBRARIES})
else()
    message(STATUS "AWS SDK not found via find_package, trying manual discovery...")
    # Try to find AWS SDK libraries in common vcpkg locations
    set(_AWS_SEARCH_PATHS
        "$ENV{VCPKG_ROOT}/installed/arm64-linux/lib"
        "$ENV{VCPKG_ROOT}/packages/aws-sdk-cpp_arm64-linux/lib"
    )
    # Also check for AWS libs in the build's vcpkg_installed
    file(GLOB _VCPKG_INSTALLED_DIRS "${CMAKE_BINARY_DIR}/vcpkg_installed/*/lib")
    list(APPEND _AWS_SEARCH_PATHS ${_VCPKG_INSTALLED_DIRS})

    set(_AWS_LIBS
        aws-cpp-sdk-s3
        aws-cpp-sdk-core
        aws-cpp-sdk-identity-management
        aws-cpp-sdk-cognito-identity
        aws-cpp-sdk-sts
        aws-crt-cpp
        aws-c-s3
        aws-c-auth
        aws-c-http
        aws-c-mqtt
        aws-c-event-stream
        aws-c-io
        aws-c-cal
        aws-c-compression
        aws-c-sdkutils
        aws-c-common
        aws-checksums
        s2n
    )
    set(_FOUND_AWS_LIBS)
    foreach(_lib ${_AWS_LIBS})
        find_library(_LIB_${_lib} NAMES ${_lib} PATHS ${_AWS_SEARCH_PATHS} NO_DEFAULT_PATH)
        if(_LIB_${_lib})
            list(APPEND _FOUND_AWS_LIBS ${_LIB_${_lib}})
            message(STATUS "  Found: ${_LIB_${_lib}}")
        endif()
    endforeach()
    if(_FOUND_AWS_LIBS)
        list(APPEND DEEPLAKE_STATIC_LINK_LIBS ${_FOUND_AWS_LIBS})
        message(STATUS "Linked ${CMAKE_LIST_LENGTH(_FOUND_AWS_LIBS)} AWS SDK libraries manually")
    else()
        message(WARNING "AWS SDK libraries not found. pg_deeplake may fail to load at runtime.")
    endif()
endif()

# }

include_directories(${DEFAULT_PARENT_DIR}/.ext/duckdb/src/include)

set(POSTGRES_DIR "${DEFAULT_PARENT_DIR}/../postgres")

foreach(PG_VERSION ${PG_VERSIONS})
    set(PG_LIB "pg_deeplake_${PG_VERSION}")
    message(STATUS "Creating library ${PG_LIB} with sources: ${PG_SOURCES}")
    ADD_LIBRARY(${PG_LIB} SHARED ${PG_SOURCES})

    set(PG_TARGET_NAME "configure_postgres_REL_${PG_VERSION}_0")

    if(TARGET ${PG_TARGET_NAME})
        add_dependencies(${PG_LIB} ${PG_TARGET_NAME})
    endif()

    if (TARGET duckdb_ep)
        add_dependencies(${PG_LIB} duckdb_ep)
    endif()

    set(PG_SERVER_INCLUDE_DIR "${postgres_INSTALL_DIR_REL_${PG_VERSION}_0}/include/server")
    set(PG_PKGLIBDIR "${postgres_INSTALL_DIR_REL_${PG_VERSION}_0}/lib")
    set(PG_SHAREDIR "${postgres_INSTALL_DIR_REL_${PG_VERSION}_0}/share")

    set_target_properties(${PG_LIB} PROPERTIES
        PREFIX ""
        POSITION_INDEPENDENT_CODE ON
    )

    target_include_directories(${PG_LIB}
        SYSTEM PRIVATE ${PG_SERVER_INCLUDE_DIR}
        PRIVATE
        ${indicators_INCLUDE_DIRS}
    )

    get_filename_component(DUCKDB_LIB_DIR "${DUCKDB_LIB_PATH}" DIRECTORY)
    target_link_directories(${PG_LIB} PUBLIC ${postgres_INSTALL_DIR_REL_${PG_VERSION}_0}/lib/ ${CMAKE_CURRENT_SOURCE_DIR}/../lib ${DUCKDB_LIB_DIR})
    target_link_directories(${PG_LIB} PUBLIC ${PNG_LIBRARIES_DIR})

    # Link DeepLake API using the appropriate target (static or shared)
    if(USE_DEEPLAKE_SHARED)
        message(STATUS "Linking ${PG_LIB} with DeepLake API shared library")
        target_link_libraries(${PG_LIB} PUBLIC
            pq ecpg ecpg_compat pgcommon_shlib pgfeutils pgport_shlib pgtypes pgcommon pgport
            DeepLakeAPI::deeplake_api_shared
            ${3RD_PARTY_LIBS}
            PRIVATE ${DEEPLAKE_STATIC_LINK_LIBS}
        )
    else()
        message(STATUS "Linking ${PG_LIB} with DeepLake API static library")
        target_link_libraries(${PG_LIB} PUBLIC
            pq ecpg ecpg_compat pgcommon_shlib pgfeutils pgport_shlib pgtypes pgcommon pgport
            -Wl,--whole-archive
            DeepLakeAPI::deeplake_api_static
            -Wl,--no-whole-archive
            ${3RD_PARTY_LIBS}
            PRIVATE ${DEEPLAKE_STATIC_LINK_LIBS}
            # Allow multiple definitions due to duplicate object files in unified library
            -Wl,--allow-multiple-definition
        )
    endif()

    target_link_libraries(${PG_LIB} PRIVATE
        fmt::fmt-header-only OpenSSL::SSL
     )

    # Link DuckDB libraries by name (not path) so they're found via link_directories
    # This works whether DuckDB is pre-built or being built by external project
    target_link_libraries(${PG_LIB} PUBLIC
        duckdb_static
        core_functions_extension
        duckdb_fastpforlib
        duckdb_fmt
        duckdb_fsst
        duckdb_hyperloglog
        duckdb_mbedtls
        duckdb_miniz
        duckdb_pg_query
        duckdb_re2
        duckdb_skiplistlib
        duckdb_utf8proc
        duckdb_yyjson
        duckdb_zstd
        jemalloc_extension
        parquet_extension
    )

    set(PG_DEFAULT_LIB_NAME "pg_deeplake${CMAKE_SHARED_LIBRARY_SUFFIX}")
    install(TARGETS ${PG_LIB}
        LIBRARY DESTINATION ${PG_PKGLIBDIR}
    )
    install(CODE "
        execute_process(COMMAND \${CMAKE_COMMAND} -E rename 
            ${PG_PKGLIBDIR}/${PG_LIB}${CMAKE_SHARED_LIBRARY_SUFFIX} 
            ${PG_PKGLIBDIR}/${PG_DEFAULT_LIB_NAME})
    ")
    install(FILES
        ${POSTGRES_DIR}/pg_deeplake.control
        ${POSTGRES_DIR}/pg_deeplake--1.0.sql
        DESTINATION ${PG_SHAREDIR}/extension
    )

    add_custom_command(TARGET ${PG_LIB} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${PG_LIB}${CMAKE_SHARED_LIBRARY_SUFFIX} "${POSTGRES_DIR}/"
        COMMENT "Copied ${CMAKE_BINARY_DIR}/${PG_LIB}${CMAKE_SHARED_LIBRARY_SUFFIX} to ${POSTGRES_DIR}/"
    )
endforeach()
