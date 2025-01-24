#ifndef CPOP_COMPUTE_SPARSE_HH
#define CPOP_COMPUTE_SPARSE_HH

#include "cusparse.h"
#include "mpi.h"

namespace cpop {

struct Vmat {
  // Global k-by-n matrix in CSR (CSR is faster for VK SpMM routine)
  cusparseSpMatDescr_t global_v;

  // Local k-by-t_size matrix in CSC (CSC is faster for c_norm initialization)
  cusparseSpMatDescr_t local_v;

  /**
   * @brief Initializes V by round robin assignment
   * 
   * @param handle The cuSPARSE handle
   * @param m The number of points
   * @param k The number of clusters
   * @param comm The MPI communicator used to distribute the matrix
   * @return 1 on error
   */
  int initialize(cusparseHandle_t& handle, int m, int k, MPI_Comm comm);

  /**
   * @brief Initializes V by global cluster assignments vector (e.g. result of argmin on D matrix).
   * Assignments vector and cluster sizes vectors are FREED by this method!
   * 
   * TODO: perhaps it makes more sense (after doing the D matrix computation stuff) to make this
   * argument a cuSPARSE dense vector.
   * 
   * @param handle The cuSPARSE handle
   * @param m The number of points
   * @param k The number of clusters
   * @param t_size The tile width
   * @param assignments The distributed vector of assignments (e.g. containing cluster indices).
   *                    Will be freed!
   * @param cluster_sizes_loc Degree k vector containing the sizes of the local cluster. Will be freed!
   *                          (TODO: To be calculated from D in a kernel along with argmins)
   * @param comm The MPI communicator used to distribute the matrix
   * @return 1 on error 
   */
  int compute_from_cluster_assignments(cusparseHandle_t& handle, int m, int k,
                                       int t_size, int* assignments,
                                       int32_t* cluster_sizes_loc,
                                       MPI_Comm comm);
};

/**
 * @brief Generates the global V matrix (k by n) in CSR using a round-robin assignment
 *
 * @param handle The cuSPARSE handle
 * @param m The number of points
 * @param k The number of clusters
 * @param comm The MPI communicator used to distribute the matrix
 * @return A cuSPARSE descriptor for the resulting sparse matrix
 */
// cusparseSpMatDescr_t initialize_global_v(cusparseHandle_t& handle, int m, int k,
//                                          MPI_Comm comm);

/**
 * @brief Generates the LOCAL V matrix (k by (n / p)) in CSC based on the assignment scheme of 
 * initialize_v. This local v is needed for the computation of c_norm.
 *
 * @param handle The cuSPARSE handle
 * @param m The number of points
 * @param k The number of clusters
 * @param comm The MPI communicator used to distribute the matrix
 * @return A cuSPARSE descriptor for the resulting sparse matrix
 */
// cusparseSpMatDescr_t initialize_local_v(cusparseHandle_t& handle, int m, int k,
//                                         MPI_Comm comm);

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
