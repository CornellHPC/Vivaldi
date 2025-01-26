#ifndef CPOP_COMPUTE_C_HH
#define CPOP_COMPUTE_C_HH

#include "cusparse.h"
#include "mpi.h"

namespace cpop {

/**
 * @brief Initializes the ET matrix before the SpMV routine
 * 
 * @param k The number of clusters
 * @param c A cuSPARSE descriptor for the resulting dense vector
 * @return int 
 */
int init_c(int k, cusparseDnVecDescr_t* c);

int init_z(int t, cusparseDnVecDescr_t* z);

int compute_z(cusparseSpMatDescr_t& lV, cusparseDnMatDescr_t& ET,
              cusparseDnVecDescr_t& z);

/**
 * @brief Computes the c norm vector by SpMV
 *
 * @param handle The cuSPARSE handle
 * @param lV The local CSC assignment matrix V
 * @param z The local z vector
 * @param c A cuSPARSE descriptor for the resulting dense vector
 * @return int
 */
int spmv(cusparseHandle_t& handle, cusparseSpMatDescr_t& lV,
         cusparseDnVecDescr_t& z, cusparseDnVecDescr_t& c);

}  // namespace cpop

#endif  // CPOP_COMPUTE_C_HH
