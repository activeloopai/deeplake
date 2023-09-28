include(FetchContent)

set(backward_URL https://github.com/bombela/backward-cpp/archive/refs/tags/v1.6.tar.gz)
set(backward_URL_HASH c654d0923d43f1cea23d086729673498e4741fb2457e806cfaeaea7b20c97c10)
set(backward_SOURCE_DIR ${DEFAULT_PARENT_DIR}/external/backward)

FetchContent_Declare(
    backward
    URL ${backward_URL}
    URL_HASH SHA256=${backward_URL_HASH}
    SOURCE_DIR ${backward_SOURCE_DIR}
)
FetchContent_MakeAvailable(backward)
