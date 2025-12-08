project(argus)

ADD_LIBRARY(rnifti1 "znzlib/znzlib.cpp" "niftilib/nifti1_io.cpp")
ADD_LIBRARY(rnifti2 "znzlib/znzlib.cpp" "niftilib/nifti2_io.cpp")

target_include_directories(rnifti1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${ZLIB_INCLUDE_DIRS})
target_include_directories(rnifti2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${ZLIB_INCLUDE_DIRS})
