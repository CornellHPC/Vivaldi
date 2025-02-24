cmake_minimum_required(VERSION 3.20)
project(cpop LANGUAGES C CXX CUDA)
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

# SLATE
option(SLATE_INSTALL "Path to SLATE installation directory.")
if (NOT SLATE_INSTALL)
  message(FATAL_ERROR "SLATE_INSTALL option must be set.")
endif()
if (NOT IS_DIRECTORY "${SLATE_INSTALL}")
  message(FATAL_ERROR "SLATE_INSTALL path does not exist.")
endif()

# Executable
file(GLOB SOURCES "src/main.cc" "src/**/*.cu" "src/**/*.cc" "src/cpop/**/*.cu" "src/cpop/**/*.cc")
add_executable(main ${SOURCES})

# Test
enable_testing()
file(GLOB SOURCES "src/test.cc" "src/**/*.cu" "src/**/*.cc" "src/cpop/**/*.cu" "src/cpop/**/*.cc")
add_executable(test_exe ${SOURCES})
add_test(cluster_test test_exe)

# Compiler options
target_compile_options(test_exe PRIVATE -g -O0 $<$<COMPILE_LANGUAGE:CUDA>:-G>)

# Sanitizers
if (SANITIZE)
  target_compile_options(main PRIVATE -fsanitize=address,leak -fno-omit-frame-pointer)
  target_link_options(main PRIVATE -fsanitize=address,leak -fno-omit-frame-pointer)
  target_compile_options(test_exe PRIVATE -fsanitize=address,leak -fno-omit-frame-pointer)
  target_link_options(test_exe PRIVATE -fsanitize=address,leak -fno-omit-frame-pointer)
endif()

# Includes
target_include_directories(main PRIVATE "${SLATE_INSTALL}/include")
target_include_directories(test_exe PRIVATE "${SLATE_INSTALL}/include")

# Libraries
target_link_directories(main PRIVATE "${SLATE_INSTALL}/lib")
target_link_libraries(main gomp cusparse cusolver blaspp lapackpp slate)
target_link_directories(test_exe PRIVATE "${SLATE_INSTALL}/lib")
target_link_libraries(test_exe gomp cusparse cusolver blaspp lapackpp slate)
