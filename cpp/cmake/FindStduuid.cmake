include(FetchContent)

set(stduuid_URL https://github.com/mariusbancila/stduuid/archive/refs/tags/v1.2.3.zip)
set(stduuid_URL_HASH 0f867768ce55f2d8fa361be82f87f0ea5e51438bc47ca30cd92c9fd8b014e84e)
set(stduuid_SOURCE_DIR ${DEFAULT_PARENT_DIR}/external/stduuid)

FetchContent_Declare(
    stduuid
    URL ${stduuid_URL}
    URL_HASH SHA256=${stduuid_URL_HASH}
    SOURCE_DIR ${stduuid_SOURCE_DIR}
)
FetchContent_MakeAvailable(stduuid)
