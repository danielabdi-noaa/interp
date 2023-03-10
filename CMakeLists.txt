cmake_minimum_required(VERSION 3.1)
project (interp)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
set(CMAKE_INSTALL_RPATH "/usr/local/lib")

# find required packages
find_package( OpenMP COMPONENTS CXX)
find_package( MPI COMPONENTS CXX)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package (eccodes COMPONENTS C REQUIRED)

# set release compiler flags
if (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} \
        -O3 -funroll-loops -march=native -DNDEBUG")
elseif (CMAKE_Fortran_COMPILER_ID STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} \
        -O3 -unroll -xHost -DNDEBUG")
endif()

# add subdirectories
add_subdirectory(src)
