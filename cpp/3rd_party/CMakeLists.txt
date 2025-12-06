set(TIFF_LIBRARY tiff CACHE INTERNAL "")

add_subdirectory(libtiff)
add_subdirectory(openjpeg)
add_subdirectory(sql-parser)
add_subdirectory(rnifti)
add_subdirectory(json)

set(TIFF_INCLUDE_DIR "${${TIFF_LIBRARY}_SOURCE_DIR}" "${${TIFF_LIBRARY}_BINARY_DIR}" CACHE INTERNAL "")
set(3RD_PARTY_LIBS ${TIFF_LIBRARY} openjp2 hsql json PARENT_SCOPE)
set(3RD_PARTY_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/openjpeg/openjp2 ${CMAKE_CURRENT_SOURCE_DIR}/rnifti ${CMAKE_CURRENT_BINARY_DIR}/libtiff PARENT_SCOPE)
