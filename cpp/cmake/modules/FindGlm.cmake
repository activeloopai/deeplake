include(FetchContent)

set(glm_TAG 1.0.1)
set(glm_URL https://github.com/g-truc/glm/archive/refs/tags/${glm_TAG}.tar.gz)
set(glm_SHA256_CHECKSUM 9f3174561fd26904b23f0db5e560971cbf9b3cbda0b280f04d5c379d03bf234c)
set(glm_SOURCE_DIR ${DEFAULT_PARENT_DIR}/.ext/glm)
set(glm_INCLUDE_DIR ${glm_SOURCE_DIR})

message(STATUS "Looking for C++ include glm")
find_path(
    glm_LOCAL
    NAMES glm
    PATHS ${DEFAULT_PARENT_DIR}/.ext
    NO_DEFAULT_PATH)

if(${glm_LOCAL} STREQUAL glm_LOCAL-NOTFOUND)
    message(STATUS "Adding glm headers to project")
    FetchContent_Declare(
        glm
        URL ${glm_URL}
        URL_HASH SHA256=${glm_SHA256_CHECKSUM}
        SOURCE_DIR ${glm_SOURCE_DIR})
    FetchContent_MakeAvailable(glm)
else()
    message(STATUS "Looking for C++ include glm - found")
endif()

include_directories(${glm_INCLUDE_DIR})
