include(FetchContent)

set(googletest_URL "https://github.com/google/googletest/archive/refs/tags/release-1.12.0.zip")
set(googletest_SOURCE_DIR "${DEFAULT_PARENT_DIR}/.ext/googletest")
set(googletest_INCLUDE_DIRS "${googletest_SOURCE_DIR}/googletest/include" "${googletest_SOURCE_DIR}/googlemock/include")

FetchContent_Declare(
    indra_googletest
    URL "${googletest_URL}"
    URL_HASH SHA256=ce7366fe57eb49928311189cb0e40e0a8bf3d3682fca89af30d884c25e983786
    SOURCE_DIR "${googletest_SOURCE_DIR}")

FetchContent_MakeAvailable(indra_googletest)

include_directories(${googletest_INCLUDE_DIRS})
