cmake_minimum_required(VERSION 3.10)
set (CMAKE_CXX_STANDARD 17)

# add executable
file(GLOB SOURCES "*.cpp")
add_executable(interp ${SOURCES})

# compiler definitions
option(USE_ITERATIVE "Use iterative solver" OFF)
if(USE_ITERATIVE)
    target_compile_definitions(interp PRIVATE USE_ITERATIVE)
endif()

# link packages
if(OpenMP_FOUND)
    target_link_libraries(interp PUBLIC OpenMP::OpenMP_CXX)
endif()
if(MPI_FOUND)
    target_link_libraries(interp PUBLIC MPI::MPI_CXX)
endif()
if (TARGET Eigen3::Eigen)
    target_link_libraries (interp PUBLIC Eigen3::Eigen)
endif()
if(eccodes_FOUND)
    target_link_libraries (interp PUBLIC eccodes)
endif()

# install
install(TARGETS interp DESTINATION ${PROJECT_SOURCE_DIR})
