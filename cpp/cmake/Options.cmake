# ###########################################################
# Indra Build options
# ###########################################################

option(AL_PG "Enable PostgreSQL" OFF)
option(AL_DEBUG "Debug build" OFF)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D HAVE_ZLIB")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D HAVE_ZLIB")
set(CMAKE_CXX_EXTENSIONS OFF)


if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # Disable MSVC warnings not present with gcc.
  #   C4204: non-constant aggregate initializer
  #   C4221: aggregate initializer with local variable
  #   C4305: implicit truncation of double to float
  add_compile_options(/wd4204 /wd4221 /wd4305)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wdeprecated-declarations")
endif()


if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-volatile -Wno-deprecated-builtins -Wno-deprecated-comma-subscript -Wno-#pragma-messages -Wno-unknown-warning-option")
endif()
