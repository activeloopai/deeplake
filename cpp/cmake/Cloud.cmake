if(NOT WIN32)
    # OpenSSL
    find_package(OpenSSL REQUIRED)
    list(APPEND INDRA_STATIC_LINK_LIBS OpenSSL::SSL OpenSSL::Crypto)
endif()

# Z
find_package(ZLIB REQUIRED)
list(APPEND INDRA_STATIC_LINK_LIBS ZLIB::ZLIB)

find_package(CURL CONFIG REQUIRED)

# GCS
find_package(google_cloud_cpp_storage CONFIG REQUIRED)
list(APPEND INDRA_CLOUD_DEPENDENCIES google-cloud-cpp::common google-cloud-cpp::storage)


# AWS_S3
find_package(AWSSDK REQUIRED COMPONENTS core s3 identity-management)
list(APPEND INDRA_CLOUD_DEPENDENCIES ${AWSSDK_LIBRARIES})

# Azure
find_package(azure-core-cpp REQUIRED COMPONENTS)
list(APPEND INDRA_CLOUD_DEPENDENCIES Azure::azure-core)

find_package(azure-identity-cpp REQUIRED COMPONENTS)
list(APPEND INDRA_CLOUD_DEPENDENCIES Azure::azure-identity)

find_package(azure-storage-blobs-cpp REQUIRED COMPONENTS)
list(APPEND INDRA_CLOUD_DEPENDENCIES Azure::azure-storage-blobs)

find_package(cpr REQUIRED)
list(APPEND INDRA_CLOUD_DEPENDENCIES cpr::cpr)
