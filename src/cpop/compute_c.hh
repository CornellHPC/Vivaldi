#ifndef CPOP_COMPUTE_C_HH
#define CPOP_COMPUTE_C_HH

#include "cusparse.h"
#include "mpi.h"

namespace cpop {

/**
 * @brief Initializes the c vector
 * 
 * @param k The number of clusters
 * @param c A cuSPARSE descriptor which will become a k-size vector
 * @return int 
 */
int init_c(int k, cusparseDnVecDescr_t* c);

/**
 * @brief Initializes the z vector
 * 
 * @param t The tile size/width of K and ET
 * @param z A cuSPARSE descriptor which will become a t-size vector
 * @return int 
 */
int init_z(int t, cusparseDnVecDescr_t* z);

/**
 * @brief Computes z based on the local V matrix and ET (i.e. using the masking strategy)
 * 
 * @param lV The local CSC assignment matrix V
 * @param ET The local partition of the ET dense matrix
 * @param z Resulting z dense vector
 * @return int 
 */
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
