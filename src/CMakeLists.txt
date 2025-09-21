cmake_minimum_required(VERSION 3.20)
project(cpop LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "80")

# OpenMP
find_package(OpenMP REQUIRED)
if(NOT OpenMP_CXX_FOUND)
  message(FATAL_ERROR "Could not find OpenMP.")
endif()

# MPI
find_package(MPI REQUIRED)
if(NOT MPI_CXX_FOUND)
  message(FATAL_ERROR "Could not find MPI.")
endif()
set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

# CUDA
find_package(CUDA REQUIRED)
#if(NOT CUDA_FOUND)
#  message(FATAL_ERROR "Could not find CUDA.")
#endif()

# SLATE
option(SLATE_INSTALL "Path to SLATE installation directory.")
if(NOT SLATE_INSTALL)
  message(FATAL_ERROR "SLATE_INSTALL option must be set.")
endif()
if(NOT IS_DIRECTORY "${SLATE_INSTALL}")
  message(FATAL_ERROR "SLATE_INSTALL path does not exist.")
endif()

# Executable
file(GLOB SOURCES "main.cu" "**/*.cu" "**/*.cc" "cpop/**/*.cu" "cpop/**/*.cc")
add_executable(main ${SOURCES})

# Compile definitions
if(BASIC)
  message(STATUS "Basic is defined")
  target_compile_definitions(main PRIVATE BASIC)
else()
  message(STATUS "Basic is not defined")
endif()

# Test
enable_testing()

file(GLOB SOURCES "test.cc"  "**/*.cu" "**/*.cc" "cpop/**/*.cu" "cpop/**/*.cc")
add_executable(test_exe ${SOURCES})
add_test(cluster_test test_exe)

file(GLOB SOURCES "test_redist.cc"  "**/*.cu" "**/*.cc" "cpop/**/*.cu" "cpop/**/*.cc")
add_executable(test_redist_exe ${SOURCES})
add_test(2d1d_redist_test test_redist_exe)

file(GLOB SOURCES "test_15d.cc"  "**/*.cu" "**/*.cc" "cpop/**/*.cu" "cpop/**/*.cc")
add_executable(test_15d_exe ${SOURCES})
add_test(15d_cluster_test test_15d_exe)

# Compiler options
set(CMAKE_CXX_FLAGS "-lboost_program_options")
target_compile_options(test_exe PRIVATE -g -O0 $<$<COMPILE_LANGUAGE:CUDA>:-G>)
target_compile_options(test_redist_exe PRIVATE -g -O0 $<$<COMPILE_LANGUAGE:CUDA>:-G>)
target_compile_options(test_15d_exe PRIVATE -g -O0 $<$<COMPILE_LANGUAGE:CUDA>:-G>)

# Sanitizers
if(SANITIZE)
  target_compile_options(main PRIVATE -fsanitize=address,leak -fno-omit-frame-pointer)
  target_link_options(main PRIVATE -fsanitize=address,leak -fno-omit-frame-pointer)
  target_compile_options(test_exe PRIVATE -fsanitize=address,leak -fno-omit-frame-pointer)
  target_link_options(test_exe PRIVATE -fsanitize=address,leak -fno-omit-frame-pointer)
endif()

target_compile_options(main PRIVATE -O3)

# Includes
target_include_directories(main PRIVATE "${SLATE_INSTALL}/include")
target_include_directories(test_exe PRIVATE "${SLATE_INSTALL}/include")
target_include_directories(test_redist_exe PRIVATE "${SLATE_INSTALL}/include")
target_include_directories(test_15d_exe PRIVATE "${SLATE_INSTALL}/include")

# Libraries
target_link_directories(main PRIVATE "${SLATE_INSTALL}/lib")
target_link_libraries(main gomp cublas cusparse cusolver blaspp lapackpp slate)
target_link_directories(test_exe PRIVATE "${SLATE_INSTALL}/lib")
target_link_libraries(test_exe gomp cublas cusparse cusolver blaspp lapackpp slate)
target_link_directories(test_redist_exe PRIVATE "${SLATE_INSTALL}/lib")
target_link_libraries(test_redist_exe gomp cublas cusparse cusolver blaspp lapackpp slate)
target_link_directories(test_15d_exe PRIVATE "${SLATE_INSTALL}/lib")
target_link_libraries(test_15d_exe gomp cublas cusparse cusolver blaspp lapackpp slate)
