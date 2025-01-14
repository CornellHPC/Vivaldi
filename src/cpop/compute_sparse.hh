#ifndef CPOP_COMPUTE_SPARSE_HH
#define CPOP_COMPUTE_SPARSE_HH

#include "cusparse.h"
#include "mpi.h"

namespace cpop {

/**
 * @brief Generates the V matrix using a round-robin assignment
 *
 * @param handle The cuSPARSE handle
 * @param m The number of points
 * @param k The number of clusters
 * @param comm The MPI communicator used to distribute the matrix
 * @return A cuSPARSE descriptor for the resulting sparse matrix
 */
cusparseSpMatDescr_t initialize_v(cusparseHandle_t& handle, int m, int k,
                                  MPI_Comm comm);

/**
 * @brief Multiplies a sparse matrix by a dense matrix
 *
 * @param handle The cuSPARSE handle
 * @param V The sparse matrix
 * @param K The dense matrix
 * @return a cuSPARSE descriptor for the resulting dense matrix
 */
cusparseDnMatDescr_t spmm(cusparseHandle_t& handle, cusparseSpMatDescr_t& V,
                          cusparseDnMatDescr_t& K);

}  // namespace cpop

#endif  // CPOP_COMPUTE_SPARSE_HH
