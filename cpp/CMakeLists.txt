cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED ENV{VCPKG_ROOT})
    message(FATAL_ERROR "You must set VCPKG_ROOT environment variable to point to your vcpkg installation")
endif()


set(VCPKG_ROOT_PATH "$ENV{VCPKG_ROOT}")
file(TO_CMAKE_PATH "${VCPKG_ROOT_PATH}" VCPKG_ROOT_PATH)
set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT_PATH}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file" FORCE)
message(STATUS "Using toolchain file: ${CMAKE_TOOLCHAIN_FILE}")

project(indra)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard" FORCE)
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "Require C++ standard" FORCE)
set(CMAKE_CXX_EXTENSIONS OFF CACHE BOOL "Disable compiler extensions" FORCE)

if(MSVC)
    add_compile_options("/permissive-")
    add_compile_options(/std:c++20)
    add_compile_options(/Zc:__cplusplus)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
    add_compile_options(/W4 /wd4101 /wd4146 /wd4244 /wd4251 /wd4456 /wd4457 /wd4702 /wd4800 /wd4018 /wd4100 /wd4389 /wd4099 /wd4505 /wd4005)

    add_compile_definitions("_CRT_SECURE_NO_WARNINGS")
    add_compile_options(/wd4996)
    add_compile_definitions("NOGDI")

    add_compile_options("/MP")
    add_compile_definitions("$<IF:$<CONFIG:Debug>,DEBUG,NDEBUG>")
    add_compile_options(
        "$<$<CONFIG:Debug>:/Od>"
        "$<$<CONFIG:Release,RelWithDebInfo>:/Ox>"
        "$<$<CONFIG:Debug,RelWithDebInfo>:/Zi>")
    add_compile_options("/bigobj")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /bigobj")


    add_compile_options("$<$<CONFIG:Debug>:/DDEBUG;/Od;/Z7;/FS>")
    add_compile_options("$<$<CONFIG:Release>:/DNDEBUG;/Ox;/FS>")
    set(CMAKE_GENERATOR "Visual Studio 17 2022")
    add_compile_definitions(NOMINMAX)
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
    add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

include(Options)

string(TOLOWER ${CMAKE_SYSTEM_NAME} AL_SYSTEM)

set(RELEASE_DEBUG_INFO)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    set(RELEASE_DEBUG_INFO " ")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(RELEASE_DEBUG_INFO "-g1")
endif()

if(${AL_DEBUG})
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -DAL_DEBUG")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g")
    endif()

    set(CMAKE_BUILD_TYPE Debug)

else()
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
    string(REPLACE "-g" "${RELEASE_DEBUG_INFO}" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    string(REPLACE "-g" "${RELEASE_DEBUG_INFO}" CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO}")
endif(${AL_DEBUG})

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_CXX_VISIBILITY_PRESET hidden)
endif()

include(CheckIncludeFileCXX)
CHECK_INCLUDE_FILE_CXX("span" HAVE_STANDARD_SPAN)

if(NOT HAVE_STANDARD_SPAN)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/span/)
endif()

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

set(DEFAULT_PARENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE STRING "Project parent directory")
set(ENV{DEFAULT_PARENT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set(old_log_level ${CMAKE_MESSAGE_LOG_LEVEL})
set(CMAKE_MESSAGE_LOG_LEVEL ERROR)

find_package(fmt CONFIG REQUIRED)

set(CMAKE_MESSAGE_LOG_LEVEL ${old_log_level})

if (${AL_PG})
    find_package(PkgConfig)

    pkg_check_modules(liburing IMPORTED_TARGET GLOBAL liburing>=2.0)
    
    include(CheckCSourceCompiles)
    set(CMAKE_REQUIRED_DEFINITIONS -D_GNU_SOURCE)
    check_c_source_compiles("
        #define _GNU_SOURCE
        #include <sys/stat.h>
        #include <fcntl.h>
        int main() {
            struct statx stx;
            statx(AT_FDCWD, \"/\", 0, STATX_ALL, &stx);
            return 0;
        }
    " HAVE_STATX)
    
    if(liburing_FOUND AND HAVE_STATX)
        set(HAVE_LIBURING TRUE)
        add_compile_definitions(HAVE_LIBURING=1)
        message(STATUS "liburing and statx found - io_uring support enabled")
    else()
        set(HAVE_LIBURING FALSE)
        if(NOT liburing_FOUND)
            message(STATUS "liburing not found - io_uring support disabled")
        endif()
        if(NOT HAVE_STATX)
            message(STATUS "statx not available - io_uring support disabled")
        endif()
    endif()
endif()

include_directories(BEFORE "${PACKAGE_PREFIX_DIR}/include")

set(CMAKE_FIND_FRAMEWORK LAST)

if(${AL_PG})
    include(CMakeLists.pg.cmake)
endif(${AL_PG})
