#ifndef DISTRIBUTED_POPCORN_CONST_H
#define DISTRIBUTED_POPCORN_CONST_H

// This file is included in kernel (".cu", ".cuh") files as well as C++ (".cc", ".hh") files
// We cannot include SLATE and CombBLAS headers here, for example, because they cannot be compiled by NVCC for CUDA.
// Only constants should be defined here.

#define DATA_TYPE float

#endif  // DISTRIBUTED_POPCORN_CONST_H