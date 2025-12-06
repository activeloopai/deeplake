# ----------------------------------------------------------------------------
# CMake file for libtiff. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------
project(${TIFF_LIBRARY})

include(CheckCSourceCompiles)
include(CheckFunctionExists)
include(CheckIncludeFile)
include(CheckTypeSize)

# Find libm, if available
find_library(M_LIBRARY m)

set(CMAKE_EXTRA_INCLUDE_FILES_SAVE ${CMAKE_EXTRA_INCLUDE_FILES})
set(CMAKE_EXTRA_INCLUDE_FILES ${CMAKE_EXTRA_INCLUDE_FILES} "stddef.h")
set(CMAKE_EXTRA_INCLUDE_FILES ${CMAKE_EXTRA_INCLUDE_FILES_SAVE})

check_type_size("size_t" SIZEOF_SIZE_T)
check_type_size("ptrdiff_t" SIZEOF_PTRDIFF_T)
check_type_size("signed short" SIZEOF_SIGNED_SHORT)
check_type_size("unsigned short" SIZEOF_UNSIGNED_SHORT)
check_type_size("signed int" SIZEOF_SIGNED_INT)
check_type_size("unsigned int" SIZEOF_UNSIGNED_INT)
check_type_size("signed long" SIZEOF_SIGNED_LONG)
check_type_size("unsigned long" SIZEOF_UNSIGNED_LONG)
check_type_size("signed long long" SIZEOF_SIGNED_LONG_LONG)
check_type_size("unsigned long long" SIZEOF_UNSIGNED_LONG_LONG)
check_type_size("unsigned char *" SIZEOF_UNSIGNED_CHAR_P)

set(TIFF_INT8_T "signed char")
set(TIFF_UINT8_T "unsigned char")

set(TIFF_INT16_T "signed short")
set(TIFF_UINT16_T "unsigned short")

string(TOLOWER ${CMAKE_SYSTEM_NAME} AL_SYSTEM)

if("${AL_SYSTEM}" STREQUAL "emscripten")
    set(TIFF_INT32_T "signed int")
    set(TIFF_INT32_FORMAT "%d")

    set(TIFF_UINT32_T "unsigned int")
    set(TIFF_UINT32_FORMAT "%u")

    set(TIFF_INT64_T "signed long")
    set(TIFF_INT64_FORMAT "%ld")

    set(TIFF_UINT64_T "unsigned long")
    set(TIFF_UINT64_FORMAT "%lu")

    set(TIFF_SIZE_T "unsigned long")
    set(TIFF_SIZE_FORMAT "%lu")
    set(TIFF_SSIZE_T "signed long")

    set(TIFF_SSIZE_FORMAT "%ld")

    set(TIFF_PTRDIFF_T "ptrdiff_t")
    set(TIFF_PTRDIFF_FORMAT "%ld")

    set(SIZEOF_SIZE_T 4)
    set(SIZEOF_PTRDIFF_T 4)
    set(SIZEOF_SIGNED_SHORT 2)
    set(SIZEOF_UNSIGNED_SHORT 2)
    set(SIZEOF_SIGNED_INT 4)
    set(SIZEOF_UNSIGNED_INT 4)
    set(SIZEOF_SIGNED_LONG 4)
    set(SIZEOF_UNSIGNED_LONG 4)
    set(SIZEOF_SIGNED_LONG_LONG 8)
    set(SIZEOF_UNSIGNED_LONG_LONG 8)
    set(SIZEOF_UNSIGNED_CHAR_P 4)
    message(STATUS "LEVON Configured libtiff for Emscripten target " , ${SIZEOF_SIZE_T}, ${SIZEOF_PTRDIFF_T}, ${SIZEOF_UNSIGNED_LONG})

else()
    if(SIZEOF_SIGNED_INT EQUAL 4)
        set(TIFF_INT32_T "signed int")
        set(TIFF_INT32_FORMAT "%d")
    elseif(SIZEOF_SIGNED_LONG EQUAL 4)
        set(TIFF_INT32_T "signed long")
        set(TIFF_INT32_FORMAT "%ld")
    endif()

    if(SIZEOF_UNSIGNED_INT EQUAL 4)
        set(TIFF_UINT32_T "unsigned int")
        set(TIFF_UINT32_FORMAT "%u")
    elseif(SIZEOF_UNSIGNED_LONG EQUAL 4)
        set(TIFF_UINT32_T "unsigned long")
        set(TIFF_UINT32_FORMAT "%lu")
    endif()

    if(SIZEOF_SIGNED_LONG EQUAL 8)
        set(TIFF_INT64_T "signed long")
        set(TIFF_INT64_FORMAT "%ld")
    elseif(SIZEOF_SIGNED_LONG_LONG EQUAL 8)
        set(TIFF_INT64_T "signed long long")

        if(MINGW)
            set(TIFF_INT64_FORMAT "%I64d")
        else()
            set(TIFF_INT64_FORMAT "%lld")
        endif()
    endif()

    if(SIZEOF_UNSIGNED_LONG EQUAL 8)
        set(TIFF_UINT64_T "unsigned long")
        set(TIFF_UINT64_FORMAT "%lu")
    elseif(SIZEOF_UNSIGNED_LONG_LONG EQUAL 8)
        set(TIFF_UINT64_T "unsigned long long")

        if(MINGW)
            set(TIFF_UINT64_FORMAT "%I64u")
        else()
            set(TIFF_UINT64_FORMAT "%llu")
        endif()
    endif()

    if(SIZEOF_UNSIGNED_INT EQUAL SIZEOF_SIZE_T)
        set(TIFF_SIZE_T "uint32_t")
        set(TIFF_SIZE_FORMAT "%u")
        set(TIFF_SSIZE_T "int32_t")
        set(TIFF_SSIZE_FORMAT "%d")
    elseif(SIZEOF_UNSIGNED_LONG EQUAL SIZEOF_SIZE_T)
        set(TIFF_SIZE_T "uint64_t")
        set(TIFF_SIZE_FORMAT "%lu")
        set(TIFF_SSIZE_T "int64_t")
        set(TIFF_SSIZE_FORMAT "%ld")
    elseif(SIZEOF_UNSIGNED_LONG_LONG EQUAL SIZEOF_SIZE_T)
        set(TIFF_SIZE_T "uint64_t")

        if(MINGW)
            set(TIFF_SIZE_FORMAT "%I64u")
            set(TIFF_SSIZE_FORMAT "%I64d")
        else()
            set(TIFF_SIZE_FORMAT "%llu")
            set(TIFF_SSIZE_FORMAT "%lld")
        endif()
    endif()

    if(SIZEOF_SIGNED_INT EQUAL SIZEOF_UNSIGNED_CHAR_P)
    elseif(SIZEOF_SIGNED_LONG EQUAL SIZEOF_UNSIGNED_CHAR_P)
    elseif(SIZEOF_SIGNED_LONG_LONG EQUAL SIZEOF_UNSIGNED_CHAR_P)
        set(TIFF_SSIZE_T "signed long long")

        if(MINGW)
        else()
        endif()
    endif()

    if(NOT SIZEOF_PTRDIFF_T)
        set(TIFF_PTRDIFF_T "${TIFF_SSIZE_T}")
        set(TIFF_PTRDIFF_FORMAT "${SSIZE_FORMAT}")
    else()
        set(TIFF_PTRDIFF_T "ptrdiff_t")
        set(TIFF_PTRDIFF_FORMAT "%ld")
    endif()
endif()

# Check functions
if(NOT MSVC)
    set(CMAKE_REQUIRED_LIBRARIES_SAVE ${CMAKE_REQUIRED_LIBRARIES})
    set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} ${M_LIBRARY})
    set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES_SAVE})
endif()

# Check for headers
include(CheckIncludeFile)
check_include_file("fcntl.h" HAVE_FCNTL_H)
check_include_file("io.h" HAVE_IO_H)
check_include_file("unistd.h" HAVE_UNISTD_H)

# CPU bit order
set(fillorder FILLORDER_MSB2LSB)

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "i.*86.*" OR
    CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "amd64.*" OR

    # AMD64 on Windows
    CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "AMD64" OR
    CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64.*")
    set(fillorder FILLORDER_LSB2MSB)
endif()

set(HOST_FILLORDER ${fillorder} CACHE STRING "Native CPU bit order")
mark_as_advanced(HOST_FILLORDER)

if(bigendian)
    set(bigendian ON)
else()
    set(bigendian OFF)
endif()

set(HOST_BIG_ENDIAN ${bigendian} CACHE STRING "Native CPU bit order")
mark_as_advanced(HOST_BIG_ENDIAN)

if(HOST_BIG_ENDIAN)
    set(HOST_BIG_ENDIAN 1)
else()
    set(HOST_BIG_ENDIAN 0)
endif()

if(HOST_BIG_ENDIAN)
    add_definitions(-DWORDS_BIGENDIAN)
endif()

# IEEE floating point
set(HAVE_IEEEFP 1 CACHE STRING "IEEE floating point is available")
mark_as_advanced(HAVE_IEEEFP)

# Large file support
if(UNIX OR MINGW)
    if(ANDROID AND(ANDROID_NATIVE_API_LEVEL LESS 21) AND CV_GCC)
    # Android NDK build problem: 'mmap' issue with GCC and API<21
    else()
        # This might not catch every possibility catered for by
        # AC_SYS_LARGEFILE.
        add_definitions(-D_FILE_OFFSET_BITS=64)
        set(FILE_OFFSET_BITS 64)
    endif()
endif()

# Documentation install directory (default to cmake project docdir)
set(LIBTIFF_DOCDIR "${CMAKE_INSTALL_FULL_DOCDIR}")

# Options to enable and disable internal codecs
option(ccitt "support for CCITT Group 3 & 4 algorithms" ON)
set(CCITT_SUPPORT ${ccitt})

option(packbits "support for Macintosh PackBits algorithm" ON)
set(PACKBITS_SUPPORT ${packbits})

option(lzw "support for LZW algorithm" ON)
set(LZW_SUPPORT ${lzw})

option(thunder "support for ThunderScan 4-bit RLE algorithm" ON)
set(THUNDER_SUPPORT ${thunder})

option(next "support for NeXT 2-bit RLE algorithm" ON)
set(NEXT_SUPPORT ${next})

option(logluv "support for LogLuv high dynamic range algorithm" ON)
set(LOGLUV_SUPPORT ${logluv})

# Option for Microsoft Document Imaging
option(mdi "support for Microsoft Document Imaging" ON)
set(MDI_SUPPORT ${mdi})

# ZLIB
set(ZLIB_SUPPORT 0)

if(ZLIB_LIBRARY)
    set(ZLIB_SUPPORT 1)
endif()

set(ZIP_SUPPORT ${ZLIB_SUPPORT})

set(PIXARLOG_SUPPORT FALSE)

# JPEG
set(JPEG_SUPPORT FALSE)

if(HAVE_JPEG)
    set(JPEG_SUPPORT TRUE)

    if(TARGET ${JPEG_LIBRARY} AND DEFINED ${JPEG_LIBRARY}_BINARY_DIR)
        include_directories("${${JPEG_LIBRARY}_BINARY_DIR}")
    endif()

    include_directories(${JPEG_INCLUDE_DIR})
endif()

set(OJPEG_SUPPORT FALSE)

set(JBIG_SUPPORT 0)
set(LZMA_SUPPORT 0)
set(JPEG12_FOUND FALSE)
set(STRIPCHOP_DEFAULT)
set(STRIP_SIZE_DEFAULT 8192)

# Win32 IO
set(win32_io FALSE)

if(WIN32 AND NOT WINRT)
    set(win32_io TRUE)
endif()

set(USE_WIN32_FILEIO ${win32_io} CACHE BOOL "Use win32 IO system (Microsoft Windows only)")

if(USE_WIN32_FILEIO)
    set(USE_WIN32_FILEIO TRUE)
else()
    set(USE_WIN32_FILEIO FALSE)
endif()

# Orthogonal features
set(SUBIFD_SUPPORT 1)
set(DEFAULT_EXTRASAMPLE_AS_ALPHA 1)
set(CHECK_JPEG_YCBCR_SUBSAMPLING 1)

if(JPEG_INCLUDE_DIR)
    list(APPEND TIFF_INCLUDES ${JPEG_INCLUDE_DIR})
endif()

if(JPEG12_INCLUDE_DIR)
    list(APPEND TIFF_INCLUDES ${JPEG12_INCLUDE_DIR})
endif()

if(JBIG_INCLUDE_DIR)
    list(APPEND TIFF_INCLUDES ${JBIG_INCLUDE_DIR})
endif()

if(LIBLZMA_INCLUDE_DIRS)
    list(APPEND TIFF_INCLUDES ${LIBLZMA_INCLUDE_DIRS})
endif()

# Libraries required by libtiff
set(TIFF_LIBRARY_DEPS)

if(M_LIBRARY)
    list(APPEND TIFF_LIBRARY_DEPS ${M_LIBRARY})
endif()

if(ZLIB_LIBRARIES)
    list(APPEND TIFF_LIBRARY_DEPS ${ZLIB_LIBRARIES})
endif()

if(JPEG_LIBRARIES)
    list(APPEND TIFF_LIBRARY_DEPS ${JPEG_LIBRARIES})
endif()

if(JPEG12_LIBRARIES)
    list(APPEND TIFF_LIBRARY_DEPS ${JPEG12_LIBRARIES})
endif()

if(JBIG_LIBRARIES)
    list(APPEND TIFF_LIBRARY_DEPS ${JBIG_LIBRARIES})
endif()

if(LIBLZMA_LIBRARIES)
    list(APPEND TIFF_LIBRARY_DEPS ${LIBLZMA_LIBRARIES})
endif()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/tif_config.h.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/tif_config.h"
    @ONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/tiffconf.h.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/tiffconf.h"
    @ONLY)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}" ${ZLIB_INCLUDE_DIRS})

set(lib_srcs
    tif_aux.c
    tif_close.c
    tif_codec.c
    tif_color.c
    tif_compress.c
    tif_dir.c
    tif_dirinfo.c
    tif_dirread.c
    tif_dirwrite.c
    tif_dumpmode.c
    tif_error.c
    tif_extension.c
    tif_fax3.c
    tif_fax3sm.c
    tif_flush.c
    tif_getimage.c
    tif_hash_set.c
    tif_jbig.c
    tif_jpeg_12.c
    tif_jpeg.c
    tif_luv.c
    tif_lzma.c
    tif_lzw.c
    tif_next.c
    tif_ojpeg.c
    tif_open.c
    tif_packbits.c
    tif_pixarlog.c
    tif_predict.c
    tif_print.c
    tif_read.c
    tif_strip.c
    tif_swab.c
    tif_thunder.c
    tif_tile.c
    tif_version.c
    tif_warning.c
    tif_webp.c
    tif_write.c
    tif_zip.c
    tif_zstd.c

    # Skip C++ stream API (tif_stream.cxx) - not needed for our use case and has deprecated type issues
    t4.h
    tif_dir.h
    tif_fax3.h
    tif_hash_set.h
    tiff.h
    tiffio.h
    tiffiop.h
    tiffvers.h
    tif_predict.h
    uvcode.h
    tiffio.hxx
    "${CMAKE_CURRENT_BINARY_DIR}/tif_config.h"
    "${CMAKE_CURRENT_BINARY_DIR}/tiffconf.h"
)

if(WIN32 AND NOT WINRT)
    list(APPEND lib_srcs tif_win32.c)
else()
    list(APPEND lib_srcs tif_unix.c)
endif()

add_library(${TIFF_LIBRARY} STATIC ${lib_srcs})
target_compile_options(${TIFF_LIBRARY} PRIVATE "-w")
target_compile_definitions(${TIFF_LIBRARY} PRIVATE TIFF_DISABLE_DEPRECATED)
target_link_libraries(${TIFF_LIBRARY} PUBLIC ${ZLIB_LIBRARIES})