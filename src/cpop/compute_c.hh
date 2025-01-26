#ifndef CPOP_COMPUTE_C_HH
#define CPOP_COMPUTE_C_HH

#include "cusparse.h"
#include "mpi.h"

namespace cpop {

/**
 * @brief Computes the c norm vector
 *
 * @param handle The cuSPARSE handle
 * @param V The local CSC assignment matrix V
 * @param ET The transposed point-centroid product matrix ET
 * @param c_norm A cuSPARSE descriptor for the resulting dense vector
 * @param comm The MPI communicator used to distribute ET
 * @return int
 */
int compute_c(cusparseHandle_t& handle, cusparseSpMatDescr_t& lV,
              cusparseDnMatDescr_t& ET, cusparseDnVecDescr_t* c_norm,
              MPI_Comm comm);

}  // namespace cpop

#endif  // CPOP_COMPUTE_C_HH
