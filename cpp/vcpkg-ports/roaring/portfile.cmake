vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO RoaringBitmap/CRoaring
    REF "v4.2.1"
    SHA512 59fe65fb79b8d0367a3e3f80deef332487060f16cd8f92151dd6282fda9a18b15221e4a2a39b361e0e34363445a9c5c7e19a38d8218cbb273c92cc4b5ac92720
    HEAD_REF master
)

string(COMPARE EQUAL "${VCPKG_LIBRARY_LINKAGE}" "static" ROARING_BUILD_STATIC)

# For emscripten builds, ensure pthread/atomics flags are passed through
if(VCPKG_TARGET_IS_EMSCRIPTEN)
    # Pass pthread flags without quotes to avoid emcc interpreting them as filenames
    vcpkg_cmake_configure(
        SOURCE_PATH "${SOURCE_PATH}"
        DISABLE_PARALLEL_CONFIGURE
        OPTIONS
            -DROARING_BUILD_STATIC=${ROARING_BUILD_STATIC}
            -DENABLE_ROARING_TESTS=OFF
            "-DCMAKE_C_FLAGS=-pthread"
            "-DCMAKE_CXX_FLAGS=-pthread"
    )
else()
    vcpkg_cmake_configure(
        SOURCE_PATH "${SOURCE_PATH}"
        DISABLE_PARALLEL_CONFIGURE
        OPTIONS
            -DROARING_BUILD_STATIC=${ROARING_BUILD_STATIC}
            -DENABLE_ROARING_TESTS=OFF
    )
endif()

vcpkg_cmake_install()

vcpkg_copy_pdbs()

vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/roaring)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

vcpkg_fixup_pkgconfig()

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")