cmake_minimum_required(VERSION 3.21)
project(mpiio)
set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)
if (MPI_CXX_FOUND)
  set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
endif()

add_executable(main src/main.cpp)
