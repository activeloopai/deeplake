include(FetchContent)
include(ExternalProject)

# Define PostgreSQL versions
set(postgres_versions
    "REL_17_0"
    "REL_18_0"
)

# Define corresponding SHA256 checksums for each version
set(postgres_SHA256_CHECKSUMS
    "16912fe4aef3c8f297b5da1b591741f132377c8b5e1b8e896e07fdd680d6bf34"
    "b155bd4a467b401ebe61b504643492aae2d0836981aa4a5a60f8668b94eadebc"
)

# Loop through each PostgreSQL version
foreach(postgres_version IN LISTS postgres_versions)
    # Find the index of the current version
    list(FIND postgres_versions ${postgres_version} postgres_index)

    # Get the corresponding SHA256 checksum for this version
    list(GET postgres_SHA256_CHECKSUMS ${postgres_index} postgres_SHA256_CHECKSUM)

    set(postgres_URL https://github.com/postgres/postgres/archive/refs/tags/${postgres_version}.zip)
    set(postgres_SOURCE_DIR ${DEFAULT_PARENT_DIR}/.ext/postgres-${postgres_version})
    set(postgres_INSTALL_DIR ${postgres_SOURCE_DIR}/install)

    if(APPLE)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -undefined dynamic_lookup -Wno-unused-command-line-argument")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -undefined dynamic_lookup -Wno-unused-command-line-argument")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -undefined dynamic_lookup")
    endif()

    ExternalProject_Add(
        configure_postgres_${postgres_version}
        URL ${postgres_URL}
        URL_HASH "SHA256=${postgres_SHA256_CHECKSUM}"
        SOURCE_DIR ${postgres_SOURCE_DIR}
        CONFIGURE_COMMAND ./configure --silent --without-icu --without-readline --without-zlib --prefix=${postgres_INSTALL_DIR} CFLAGS=-fPIC
        BUILD_COMMAND COMMAND $(MAKE) MAKELEVEL=0
        INSTALL_COMMAND $(MAKE) install
        BUILD_ALWAYS FALSE
        BUILD_IN_SOURCE TRUE
        COMMENT "Configuring and building PostgreSQL ${postgres_version}"
    )
endforeach()

set(postgres_INSTALL_DIR_REL_17_0 ${DEFAULT_PARENT_DIR}/.ext/postgres-REL_17_0/install)
set(postgres_INSTALL_DIR_REL_18_0 ${DEFAULT_PARENT_DIR}/.ext/postgres-REL_18_0/install)
