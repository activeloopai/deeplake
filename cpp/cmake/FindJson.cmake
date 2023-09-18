include(FetchContent)

set(json_URL https://github.com/nlohmann/json/releases/download/v3.11.2/include.zip)
set(json_URL_HASH e5c7a9f49a16814be27e4ed0ee900ecd0092bfb7dbfca65b5a421b774dccaaed)
set(json_SOURCE_DIR ${DEFAULT_PARENT_DIR}/external/json)
set(json_INCLUDE_DIR ${json_SOURCE_DIR}/include)

message(STATUS "Looking for C++ include nlohmann::json")
find_path(json_LOCAL
    NAMES nlohmann/json.hpp
    PATHS ${json_INCLUDE_DIR}
    PATH_SUFFIXES include
)
if (${json_LOCAL} STREQUAL json_LOCAL-NOTFOUND)
    message(STATUS "Adding nlohmann::json as an external project")

    FetchContent_Declare(
        json
        URL ${json_URL}
        URL_HASH SHA256=${json_URL_HASH}
        SOURCE_DIR ${json_SOURCE_DIR}
    )
    FetchContent_MakeAvailable(json)
else()
    message(STATUS "Looking for C++ include nlohmann::json - found")
endif()

include_directories(${json_INCLUDE_DIR})