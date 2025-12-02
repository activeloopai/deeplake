include(ExternalProject)

set(sentry_version "0.6.7")
set(sentr_url "https://github.com/getsentry/sentry-native/releases/download/${sentry_version}/sentry-native.zip")
set(sentry_INSTALL_DIR ${DEFAULT_PARENT_DIR}/.ext/sentry)
set(sentry_SOURCE_DIR ${DEFAULT_PARENT_DIR}/.ext/sentry)
set(sentry_INCLUDE_DIR ${sentry_SOURCE_DIR}/include)

find_package(CURL CONFIG REQUIRED)

ExternalProject_Add(
    sentry_ep
    URL ${sentr_url}
    CMAKE_ARGS -DSENTRY_BACKEND=inproc
               -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -DCMAKE_INSTALL_PREFIX=${sentry_SOURCE_DIR}
               -DSENTRY_BUILD_SHARED_LIBS=OFF
               -DSENTRY_BUILD_EXAMPLES=OFF
               -DSENTRY_BUILD_TESTS=OFF
               -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
               -DSENTRY_TRANSPORT_CURL=curl
               -DCMAKE_INSTALL_LIBDIR=lib
               -DCURL_FOUND=${CURL_FOUND}
               -DCURL_INCLUDE_DIR=${CURL_INCLUDE_DIRS}
            #    -DCURL_LIBRARIES=${CURL_LIBRARIES}
    SOURCE_DIR ${sentry_SOURCE_DIR}
)

set(SENTRY_INCLUDE_DIRS ${sentry_INSTALL_DIR}/include)
if(MSVC)
    set(SENTRY_LIBRARY ${sentry_INSTALL_DIR}/lib/sentry${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(SENTRY_LIBRARIES ${sentry_INSTALL_DIR}/lib/sentry${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
    set(SENTRY_LIBRARY ${sentry_INSTALL_DIR}/lib/libsentry${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(SENTRY_LIBRARIES ${sentry_INSTALL_DIR}/lib/libsentry${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()
include_directories(${SENTRY_INCLUDE_DIRS})

file(MAKE_DIRECTORY ${SENTRY_INCLUDE_DIRS})
add_library(sentry::sentry STATIC IMPORTED)
set_target_properties(sentry::sentry PROPERTIES IMPORTED_LOCATION ${SENTRY_LIBRARIES} INTERFACE_INCLUDE_DIRECTORIES
                                                                                      ${SENTRY_INCLUDE_DIRS})

add_dependencies(sentry::sentry sentry_ep)
