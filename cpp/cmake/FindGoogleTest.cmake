include(FetchContent)

set(googletest_URL https://github.com/google/googletest.git)
set(googletest_TAG v1.12.0)
set(googletest_SOURCE_DIR ${DEFAULT_PARENT_DIR}/external/googletest)
set(googletest_INCLUDE_DIRS ${SOURCE_DIR}/googletest/include ${SOURCE_DIR}/googlemock/include)

FetchContent_Declare(googletest
    GIT_REPOSITORY ${googletest_URL}
    GIT_TAG ${googletest_TAG}
    SOURCE_DIR ${googletest_SOURCE_DIR}
)

FetchContent_MakeAvailable(googletest)
include_directories(${googletest_INCLUDE_DIR})