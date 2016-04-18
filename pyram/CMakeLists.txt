cmake_minimum_required(VERSION 2.8)

project (VG-RAM)

set (PACKAGE_MAJOR_VERSION 1)
set (PACKAGE_MINOR_VERSION 0)
set (PACKAGE_PATCH_VERSION 0)
set (PACKAGE_VERSION ${PACKAGE_MAJOR_VERSION}.${PACKAGE_MINOR_VERSION}.${PACKAGE_PATCH_VERSION})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

option(USE_CUDA "Build VG-RAM with mixed CUDA and OpenMP support" OFF)
option(USE_MIXED_OMP_CUDA "Build VG-RAM with mixed CUDA and OpenMP support" OFF)

if (APPLE)
	set(CMAKE_MACOSX_RPATH 1)
endif(APPLE)

add_subdirectory (src)
add_subdirectory (mae)
