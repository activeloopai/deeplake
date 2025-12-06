# ###########################################################
# Indra Build options
# ###########################################################

option(AL_PYTHON "PYTHON build" OFF)
option(AL_ASSERTIONS "Build with logging enabled" OFF)
option(AL_DEBUG "Debug build" OFF)
option(AL_TESTS "Tests build" OFF)
option(AL_V4 "V4 build" OFF)
option(AL_COVERAGE "Enable code coverage" OFF)
option(AL_ENTERPRISE "Enterprise build" OFF)
option(AL_ASAN "run with address sanitizer" OFF)
option(AL_TSAN "run with thread sanitizer" OFF)
option(AL_UBSAN "run with undefined behavior sanitizer" OFF)
option(AL_ENABLE_SENTRY "Enable Sentry" OFF)
option(AL_PG "Enable PostgreSQL" OFF)
option(AL_GO "GO build" OFF)
option(AL_VIEWER_API "Build viewer_api WASM module" OFF)

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

if(${AL_TESTS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAL_TESTS")
endif(${AL_TESTS})

if(${AL_ASSERTIONS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAL_ASSERTIONS")
endif(${AL_ASSERTIONS})

if(${AL_ENTERPRISE})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAL_ENTERPRISE")
endif(${AL_ENTERPRISE})

if(${AL_ASAN})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAL_ASAN")
endif(${AL_ASAN})

if(${AL_TSAN})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAL_TSAN")
endif(${AL_TSAN})

if(${AL_UBSAN})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAL_UBSAN")
endif(${AL_UBSAN})

if(${AL_COVERAGE})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
endif(${AL_COVERAGE})

if(${AL_ENABLE_SENTRY})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAL_ENABLE_SENTRY")
endif(${AL_ENABLE_SENTRY})

if(${AL_PG})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAL_PG")
endif(${AL_PG})