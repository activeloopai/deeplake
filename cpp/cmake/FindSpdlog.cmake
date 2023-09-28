include(FetchContent)

set(spdlog_URL https://github.com/gabime/spdlog/archive/refs/tags/v1.12.0.tar.gz)
set(spdlog_URL_HASH 4dccf2d10f410c1e2feaff89966bfc49a1abb29ef6f08246335b110e001e09a9)
set(spdlog_SOURCE_DIR ${DEFAULT_PARENT_DIR}/external/spdlog)

FetchContent_Declare(
    spdlog
    URL ${spdlog_URL}
    URL_HASH SHA256=${spdlog_URL_HASH}
    SOURCE_DIR ${spdlog_SOURCE_DIR}
)
FetchContent_MakeAvailable(spdlog)
