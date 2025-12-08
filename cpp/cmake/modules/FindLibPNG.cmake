include(ExternalProject)

set(png_SOURCE_DIR ${DEFAULT_PARENT_DIR}/.ext/png)

set(png_URL https://git.code.sf.net/p/libpng/code)
set(png_TAG v1.6.45)
set(png_INSTALL ${DEFAULT_PARENT_DIR}/.ext/png-build/install)

find_path(
    png_INCLUDE_CHECK
    NAMES png.h
    PATHS ${png_INSTALL}/include)

find_path(
    png_LIB_CHECK
    NAMES libpng${CMAKE_STATIC_LIBRARY_SUFFIX}
    PATHS ${png_INSTALL}/lib)

if(${png_INCLUDE_CHECK} STREQUAL png_INCLUDE_CHECK-NOTFOUND OR ${png_LIB_CHECK} STREQUAL png_LIB_CHECK-NOTFOUND)
    message(STATUS "Could NOT find png (missing: png_DIR)")
    set(png_FOUND 0)
else()
    set(png_FOUND 1)
endif()

if(NOT ${png_FOUND})

    find_package(ZLIB REQUIRED)

    message(STATUS "Adding png external project")
    ExternalProject_Add(
        png_ep
        GIT_REPOSITORY ${png_URL}
        GIT_TAG ${png_TAG}
        INSTALL_DIR ${png_INSTALL}
        SOURCE_DIR ${png_SOURCE_DIR}
        CMAKE_ARGS -DPNG_SHARED:BOOL=OFF
        -DPNG_TESTS:BOOL=OFF
        -DCMAKE_INSTALL_LIBDIR=lib
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DCMAKE_INSTALL_PREFIX:STRING=${png_INSTALL}
        -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
        -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
        -DZLIB_ROOT=${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}
        -DZLIB_LIBRARY=${ZLIB_LIBRARIES}
        -DZLIB_INCLUDE_DIR=${ZLIB_INCLUDE_DIRS}
        DEPENDS ${DEPENDS}
        LOG_CONFIGURE ON
        LOG_BUILD ON
        LOG_INSTALL ON
        LOG_DOWNLOAD ON
        LOG_OUTPUT_ON_FAILURE ON)
endif()

set(PNG_INCLUDE_DIRS ${png_INSTALL}/include)
set(PNG_LIBRARIES_DIR ${png_INSTALL}/lib)

if(WIN32)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(PNG_LIB_NAME "libpng16_staticd")
    else()
        set(PNG_LIB_NAME "libpng16_static")
    endif()
else()
    set(PNG_LIB_NAME "png16")
endif()

find_library(PNG_LIBRARIES
    NAMES ${PNG_LIB_NAME}
    PATHS ${PNG_LIBRARIES_DIR}
    NO_DEFAULT_PATH)

if(NOT PNG_LIBRARIES AND WIN32)
    find_library(PNG_LIBRARIES
        NAMES libpng16_staticd libpng16_static
        PATHS ${PNG_LIBRARIES_DIR}
        NO_DEFAULT_PATH)
endif()

if(NOT PNG_LIBRARIES)
    find_library(PNG_LIBRARIES
        NAMES png libpng
        PATHS ${PNG_LIBRARIES_DIR}
        NO_DEFAULT_PATH)
endif()

if(PNG_LIBRARIES AND NOT TARGET libpng16_static)
    add_library(libpng16_static STATIC IMPORTED GLOBAL)

    set(ZLIB_LIB_DEBUG "")
    set(ZLIB_LIB_RELEASE "")
    list(LENGTH ZLIB_LIBRARIES zlib_count)
    if(zlib_count GREATER 1)
        list(GET ZLIB_LIBRARIES 1 ZLIB_LIB_RELEASE)
        if(zlib_count GREATER 3)
            list(GET ZLIB_LIBRARIES 3 ZLIB_LIB_DEBUG)
        else()
            set(ZLIB_LIB_DEBUG "${ZLIB_LIB_RELEASE}")
        endif()
    else()
        list(GET ZLIB_LIBRARIES 0 ZLIB_LIB_RELEASE)
        set(ZLIB_LIB_DEBUG "${ZLIB_LIB_RELEASE}")
    endif()

    set_target_properties(libpng16_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${PNG_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "$<$<CONFIG:Debug>:${ZLIB_LIB_DEBUG}>$<$<NOT:$<CONFIG:Debug>>:${ZLIB_LIB_RELEASE}>")

    if(WIN32 AND MSVC)
        find_library(PNG_LIBRARIES_DEBUG
            NAMES libpng16_staticd
            PATHS ${PNG_LIBRARIES_DIR}
            NO_DEFAULT_PATH)

        find_library(PNG_LIBRARIES_RELEASE
            NAMES libpng16_static
            PATHS ${PNG_LIBRARIES_DIR}
            NO_DEFAULT_PATH)

        if(PNG_LIBRARIES_DEBUG AND PNG_LIBRARIES_RELEASE)
            set_target_properties(libpng16_static PROPERTIES
                IMPORTED_LOCATION_DEBUG "${PNG_LIBRARIES_DEBUG}"
                IMPORTED_LOCATION_RELEASE "${PNG_LIBRARIES_RELEASE}"
                IMPORTED_LOCATION_RELWITHDEBINFO "${PNG_LIBRARIES_RELEASE}"
                IMPORTED_LOCATION_MINSIZEREL "${PNG_LIBRARIES_RELEASE}")
        elseif(PNG_LIBRARIES_DEBUG)
            set_target_properties(libpng16_static PROPERTIES
                IMPORTED_LOCATION "${PNG_LIBRARIES_DEBUG}")
        elseif(PNG_LIBRARIES_RELEASE)
            set_target_properties(libpng16_static PROPERTIES
                IMPORTED_LOCATION "${PNG_LIBRARIES_RELEASE}")
        else()
            set_target_properties(libpng16_static PROPERTIES
                IMPORTED_LOCATION "${PNG_LIBRARIES}")
        endif()
    endif()

    if(WIN32)
        add_library(libpng ALIAS libpng16_static)
    endif()
endif()
