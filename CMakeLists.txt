cmake_minimum_required(VERSION 3.20)
project(distributed_popcorn LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

# OpenMP
find_package(OpenMP REQUIRED)
if (NOT OpenMP_CXX_FOUND)
  message(FATAL_ERROR "Could not find OpenMP.")
endif()

# MPI
find_package(MPI REQUIRED)
if (NOT MPI_CXX_FOUND)
  message(FATAL_ERROR "Could not find MPI.")
endif()
set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

# CUDA
find_package(CUDA REQUIRED)
if (NOT CUDA_FOUND)
  message(FATAL_ERROR "Could not find CUDA.")
endif()

# GPU Support
option(GPU "Enable GPU support.")
if (GPU)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUDA")
endif()

# CombBLAS
option(COMBBLAS_INSTALL	"Path to CombBLAS installation directory.")
if (NOT COMBBLAS_INSTALL)
  message(FATAL_ERROR "COMBBLAS_INSTALL option must be set.")
endif()
if (NOT IS_DIRECTORY "${COMBBLAS_INSTALL}")
  message(FATAL_ERROR "COMBBLAS_INSTALL path does not exist.")
endif()

# SLATE
option(SLATE_INSTALL "Path to SLATE installation directory.")
if (NOT SLATE_INSTALL)
  message(FATAL_ERROR "SLATE_INSTALL option must be set.")
endif()
if (NOT IS_DIRECTORY "${SLATE_INSTALL}")
  message(FATAL_ERROR "SLATE_INSTALL path does not exist.")
endif()

# Executable
file(GLOB SOURCES "src/main.cc" "src/**/*.cc" "src/**/*.cu")
add_executable(main ${SOURCES})

# Includes
target_include_directories(main PRIVATE "${COMBBLAS_INSTALL}/include")
target_include_directories(main PRIVATE "${SLATE_INSTALL}/include")

# Libraries
target_link_directories(main PRIVATE "${COMBBLAS_INSTALL}/lib")
target_link_directories(main PRIVATE "${SLATE_INSTALL}/lib")
target_link_libraries(main gomp cusparse cusolver blaspp lapackpp slate CombBLAS)
