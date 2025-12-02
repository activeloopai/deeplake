include(FetchContent)

set(svgpp_URL https://github.com/svgpp/svgpp.git)
set(svgpp_TAG v1.3.0)
set(svgpp_SOURCE_DIR ${DEFAULT_PARENT_DIR}/.ext/svgpp)
set(svgpp_INCLUDE_DIR ${DEFAULT_PARENT_DIR}/.ext/svgpp/include)

find_path(
    svgpp_LOCAL
    NAMES include/svgpp
    PATHS ${svgpp_SOURCE_DIR})

if(${svgpp_LOCAL} STREQUAL svgpp_LOCAL-NOTFOUND)
    FetchContent_Declare(
        svgpp
        GIT_REPOSITORY ${svgpp_URL}
        GIT_TAG ${svgpp_TAG}
        SOURCE_DIR ${svgpp_SOURCE_DIR})
    FetchContent_MakeAvailable(svgpp)

endif()

include_directories(${svgpp_INCLUDE_DIR})
