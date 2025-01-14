#ifndef CPOP_COMPUTE_C_HH
#define CPOP_COMPUTE_C_HH

#include "cusparse.h"
#include "mpi.h"

namespace cpop {

/**
 * @brief Computes the c norm vector
 *
 * @param handle The cuSPARSE handle
 * @param V The assignment matrix V
 * @param ET The transposed point-centroid product matrix ET
 * @param comm The MPI communicator used to distribute ET
 * @return A cuSPARSE descriptor for the resulting dense vector
 */
cusparseDnVecDescr_t compute_c(cusparseHandle_t& handle, cusparseSpMatDescr_t V,
                               cusparseDnMatDescr_t ET, MPI_Comm comm);

}  // namespace cpop

#endif  // CPOP_COMPUTE_C_HH
