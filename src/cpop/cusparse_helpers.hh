#ifndef CPOP_CUSPARSE_HELPERS_HH
#define CPOP_CUSPARSE_HELPERS_HH

#include "cuda_runtime.h"
#include "cusparse.h"
#include "mpi.h"

#include <cassert>
#include <iostream>

namespace cpop {

// all the functions here do exactly what you think they do

void print_arr_(float* arr, int64_t size);

void print_arr_(int64_t* arr, int64_t size);

void print_arr_(int* arr, int64_t size);

void print(cusparseDnMatDescr_t& M, int rank);

void print(cusparseDnVecDescr_t& M, int rank);

/**
 * @brief Print a local sparse matrix, e.g. local V (CSC)
 */
void print(cusparseSpMatDescr_t& M, int rank);

/**
 * @brief Print a global sparse matrix, e.g. global V (CSR)
 */
void print(cusparseSpMatDescr_t& M);

}  // namespace cpop

#endif  // CPOP_CUSPARSE_HELPERS_HH