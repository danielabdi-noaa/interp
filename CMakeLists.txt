cmake_minimum_required(VERSION 3.1)
project (interp)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# some options
option(ENABLE_GRIB "Enable grib2 processing with eccodes library" ON)
option(USE_ITERATIVE "Use iterative solver" OFF)
option(ENABLE_3D "Enable 3D interpolation" OFF)

# find required packages
find_package( OpenMP COMPONENTS CXX)
find_package( MPI COMPONENTS CXX)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
if(ENABLE_GRIB)
    find_package (eccodes COMPONENTS C REQUIRED)
endif()

# set release compiler flags
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -funroll-loops -march=native")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -unroll -xHost")
endif()

# add subdirectories
add_subdirectory(src)
