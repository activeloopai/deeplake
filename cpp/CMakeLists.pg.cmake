option(BUILD_PG_16 "Build PostgreSQL 16 extension" OFF)
option(BUILD_PG_17 "Build PostgreSQL 17 extension" OFF)
option(BUILD_PG_18 "Build PostgreSQL 18 extension" ON)

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


# TODO try to remove {
set(INDRA_CLOUD_DEPENDENCIES)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/FindLZ4.cmake)
find_package(ICU REQUIRED COMPONENTS uc i18n data)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Cloud.cmake)
find_package(libjpeg-turbo CONFIG REQUIRED)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/FindLibPNG.cmake)
find_package(opentelemetry-cpp CONFIG COMPONENTS opentelemetry_trace REQUIRED)
find_package(Boost REQUIRED COMPONENTS
    algorithm system container iterator mpl histogram
)

find_package(boost_json CONFIG REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package(DCMTK CONFIG REQUIRED)
find_package(roaring CONFIG REQUIRED)

# OpenGL and GLEW dependencies for engine module
find_package(OpenGL REQUIRED)
find_package(GLEW 2.0 REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})

# FreeType dependency
find_package(Freetype REQUIRED)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/FindDuckDB.cmake)

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
        ${CMAKE_SOURCE_DIR}
        ${indicators_INCLUDE_DIRS}
    )

    get_filename_component(DUCKDB_LIB_DIR "${DUCKDB_LIB_PATH}" DIRECTORY)
    target_link_directories(${PG_LIB} PUBLIC ${postgres_INSTALL_DIR_REL_${PG_VERSION}_0}/lib/ ${CMAKE_CURRENT_SOURCE_DIR}/../lib ${DUCKDB_LIB_DIR})
    target_link_directories(${PG_LIB} PUBLIC ${PNG_LIBRARIES_DIR})

    # Use the unified static library instead of separate libraries
    set(DEEPLAKE_STATIC_LIB "${CMAKE_CURRENT_SOURCE_DIR}/.ext/deeplake_static/lib/libdeeplake_static.a")

    message(STATUS "Using unified Deeplake library: ${DEEPLAKE_STATIC_LIB}")

    target_link_libraries(${PG_LIB} PUBLIC
        pq ecpg ecpg_compat pgcommon_shlib pgfeutils pgport_shlib pgtypes pgcommon pgport
        -Wl,--whole-archive
        ${DEEPLAKE_STATIC_LIB}
        -Wl,--no-whole-archive
        ${3RD_PARTY_LIBS} ${FREETYPE_LIBRARIES}
        PRIVATE ${INDRA_STATIC_LINK_LIBS}
        # Allow multiple definitions due to duplicate object files in unified library
        -Wl,--allow-multiple-definition
    )

    target_link_libraries(${PG_LIB} PRIVATE
        ${INDRA_CLOUD_DEPENDENCIES}
        fmt::fmt-header-only ${OPENTELEMETRY_CPP_LIBRARIES}
        png lz4_static OpenSSL::SSL ICU::uc ICU::data ICU::i18n Boost::json
     $<IF:$<TARGET_EXISTS:libjpeg-turbo::turbojpeg>,libjpeg-turbo::turbojpeg,libjpeg-turbo::turbojpeg-static>)

    target_link_libraries(${PG_LIB} PRIVATE DCMTK::dcmdata DCMTK::oflog DCMTK::ofstd DCMTK::dcmimage DCMTK::dcmjpeg)
    target_link_libraries(${PG_LIB} PRIVATE roaring::roaring)

    # Link OpenGL, GLEW, and FreeType
    target_link_libraries(${PG_LIB} PRIVATE OpenGL::GL OpenGL::GLU GLEW Freetype::Freetype)

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
