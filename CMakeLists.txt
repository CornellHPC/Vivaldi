cmake_minimum_required(VERSION 3.20)
project(mpiio)
set(CMAKE_CXX_STANDARD 14)

# MPI
find_package(MPI REQUIRED)
if (NOT MPI_CXX_FOUND)
	message(FATAL_ERROR "Could not find MPI.")
endif()
set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

# CombBLAS
option(COMBBLAS_INSTALL	"Path to CombBLAS installation directory.")
if (NOT COMBBLAS_INSTALL)
	message(FATAL_ERROR "COMBBLAS_INSTALL option must be set.")
endif()
if (NOT EXISTS "${COMBBLAS_INSTALL}")
	message(FATAL_ERROR "COMBBLAS_INSTALL path does not exist.")
endif()
#set(CMAKE_CXX_FLAGS "-I${COMBBLAS_INSTALL}/include ${CMAKE_CXX_FLAGS}")

add_executable(main src/main.cpp)
target_include_directories(main PRIVATE "${COMBBLAS_INSTALL}/include")
target_link_libraries(main "${COMBBLAS_INSTALL}/lib/libCombBLAS.a")
