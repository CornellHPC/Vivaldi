#ifndef CPOP_COMPUTE_C_HH
#define CPOP_COMPUTE_C_HH

#include "cusparse.h"
#include "mpi.h"

namespace cpop {

/**
 * @brief Initializes the ET matrix before the SpMV routine
 * 
 * @param lV The local CSC assignment matrix V
 * @param c A cuSPARSE descriptor for the resulting dense vector
 * @return int 
 */
int init_c(cusparseSpMatDescr_t& lV, cusparseDnVecDescr_t* c);

/**
 * @brief Computes the c norm vector
 *
 * @param handle The cuSPARSE handle
 * @param lV The local CSC assignment matrix V
 * @param ET The transposed point-centroid product matrix ET
 * @param c A cuSPARSE descriptor for the resulting dense vector
 * @param comm The MPI communicator used to distribute ET
 * @return int
 */
int compute_c(cusparseHandle_t& handle, cusparseSpMatDescr_t& lV,
              cusparseDnMatDescr_t& ET, cusparseDnVecDescr_t& c, MPI_Comm comm);

}  // namespace cpop

#endif  // CPOP_COMPUTE_C_HH
