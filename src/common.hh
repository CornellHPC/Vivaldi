#ifndef DISTRIBUTED_POPCORN_COMMON_H
#define DISTRIBUTED_POPCORN_COMMON_H

#include <chrono>

// This is required by CombBLAS
using namespace std;

// #include "CombBLAS/CombBLAS.h"

// Macro "Error" is defined in CombBLAS and SLATE. It is unused in
// CombBLAS, so we undefine it here to prevent name collisions.
#undef Error

#include "slate/slate.hh"

#include "const.hh"

// // Sparse column format for CombBLAS
// using UDER = combblas::SpCCols<int64_t, DATA_TYPE>;

// // Semiring type for CombBLAS
// using SR = combblas::PlusTimesSRing<DATA_TYPE, DATA_TYPE>;

// Type defs for SLATE and CombBLAS matrix pointers
using slate_matrix = std::unique_ptr<slate::Matrix<float>>;
using sm_ptr = std::unique_ptr<slate::Matrix<DATA_TYPE>>;
// using c_dn_ptr = std::unique_ptr<combblas::DnParMat<int64_t, DATA_TYPE>>;
// using c_sp_ptr = std::unique_ptr<combblas::SpParMat<int64_t, DATA_TYPE, UDER>>;

// Nickname for the clock types
using hrc = std::chrono::high_resolution_clock;
using s = std::chrono::seconds;
using ms = std::chrono::milliseconds;

#endif  // DISTRIBUTED_POPCORN_COMMON_H
