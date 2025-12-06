cmake_policy(SET CMP0003 NEW)
if(POLICY CMP0042)
  cmake_policy(SET CMP0042 NEW)
endif()

set(OPENJPEG_LIBRARY_NAME openjp2)

project(openjpeg C)

#-----------------------------------------------------------------------------
# OPENJPEG version number, useful for packaging and doxygen doc:
set(OPENJPEG_VERSION_MAJOR 2)
set(OPENJPEG_VERSION_MINOR 4)
set(OPENJPEG_VERSION_BUILD 0)
set(OPENJPEG_VERSION
  "${OPENJPEG_VERSION_MAJOR}.${OPENJPEG_VERSION_MINOR}.${OPENJPEG_VERSION_BUILD}")
set(PACKAGE_VERSION
  "${OPENJPEG_VERSION_MAJOR}.${OPENJPEG_VERSION_MINOR}.${OPENJPEG_VERSION_BUILD}")

# Because autotools does not support X.Y notation for SOVERSION, we have to use
# two numbering, one for the openjpeg version and one for openjpeg soversion
# version | soversion
#   1.0   |  0
#   1.1   |  1
#   1.2   |  2
#   1.3   |  3
#   1.4   |  4
#   1.5   |  5
#   1.5.1 |  5
#   2.0   |  6
#   2.0.1 |  6
#   2.1   |  7
#   2.1.1 |  7
#   2.1.2 |  7
#   2.2.0 |  7
#   2.3.0 |  7
#   2.3.1 |  7
#   2.4.0 |  7
# above is the recommendation by the OPJ team. If you really need to override this default,
# you can specify your own OPENJPEG_SOVERSION at cmake configuration time:
# cmake -DOPENJPEG_SOVERSION:STRING=42 /path/to/openjpeg
if(NOT OPENJPEG_SOVERSION)
  set(OPENJPEG_SOVERSION 7)
endif()

set(OPENJPEG_LIBRARY_PROPERTIES
  VERSION   "${OPENJPEG_VERSION_MAJOR}.${OPENJPEG_VERSION_MINOR}.${OPENJPEG_VERSION_BUILD}"
  SOVERSION "${OPENJPEG_SOVERSION}"
)

set(OPENJPEG_BUILD "openjp2-${OPENJPEG_VERSION}")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(OPENJPEG_BUILD "${OPENJPEG_BUILD}-debug")
endif()

message(STATUS "OpenJPEG: VERSION = ${OPENJPEG_VERSION}, BUILD = ${OPENJPEG_BUILD}")

#-----------------------------------------------------------------------------
# opj_config.h generation (1/2)

# Check if some include files are provided by the system
# These files are mandatory, so if they are not provided OpenJPEG library can't be built
include(CheckIncludeFile)
macro(ensure_file_include INCLUDE_FILENAME VARIABLE_NAME MANDATORY_STATUS)
  check_include_file(${INCLUDE_FILENAME} ${VARIABLE_NAME})
  if(NOT ${VARIABLE_NAME})
    if(${MANDATORY_STATUS})
      message(STATUS "The file '${INCLUDE_FILENAME}' is mandatory for OpenJPEG build, but not found on your system")
      return()
    else()
      message(STATUS "The file '${INCLUDE_FILENAME}' is optional for OpenJPEG build and not found on your system."
              " Internal implementation will be used.")
    endif()
  endif()
endmacro()

# Allocating Aligned Memory Blocks
include(CheckIncludeFiles)
include(CheckSymbolExists)
#-----------------------------------------------------------------------------
# opj_config.h generation (2/2)
configure_file(
  ${CMAKE_CURRENT_LIST_DIR}/openjp2/opj_config.h.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/openjp2/opj_config.h
  @ONLY
)

configure_file(
  ${CMAKE_CURRENT_LIST_DIR}/openjp2/opj_config_private.h.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/openjp2/opj_config_private.h
  @ONLY
)

add_subdirectory(openjp2)

# Setting all necessary variables
set(OPENJPEG_LIBRARIES     ${OPENJPEG_LIBRARY_NAME}  PARENT_SCOPE)
set(OPENJPEG_VERSION       ${OPENJPEG_VERSION}       PARENT_SCOPE)
set(OPENJPEG_MAJOR_VERSION ${OPENJPEG_VERSION_MAJOR} PARENT_SCOPE)
set(OPENJPEG_MINOR_VERSION ${OPENJPEG_VERSION_MINOR} PARENT_SCOPE)
set(OPENJPEG_BUILD_VERSION ${OPENJPEG_VERSION_BUILD} PARENT_SCOPE)
get_target_property(_openjpeg_include_dirs ${OPENJPEG_LIBRARY_NAME} INCLUDE_DIRECTORIES)
set(OPENJPEG_INCLUDE_DIRS  ${_openjpeg_include_dirs} PARENT_SCOPE)

# OpenJPEG can't be built only if configuration script doesn't encounter any problem
if(NOT DEFINED CAN_BUILD_OPENJPEG)
  # all prerequisites are fulfilled
  set(CAN_BUILD_OPENJPEG TRUE PARENT_SCOPE)
endif()
