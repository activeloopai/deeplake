include(FetchContent)

set(json_URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
set(json_URL_HASH 8c4b26bf4b422252e13f332bc5e388ec0ab5c3443d24399acb675e68278d341f)
set(json_SOURCE_DIR ${DEFAULT_PARENT_DIR}/external/json)

FetchContent_Declare(
    json
    URL ${json_URL}
    URL_HASH SHA256=${json_URL_HASH}
    SOURCE_DIR ${json_SOURCE_DIR}
)
FetchContent_MakeAvailable(json)
