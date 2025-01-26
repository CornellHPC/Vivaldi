#ifndef CPOP_COMPUTE_SPARSE_HH
#define CPOP_COMPUTE_SPARSE_HH

#include "cusparse.h"
#include "mpi.h"

namespace cpop {

/**
 * @brief Computes global assignments by AllgatherV from the local cluster assignments.
 * 
 * @param m The number of points
 * @param t The tile width (e.g. width of column of K on this process)
 * @param assignments t-size array of assignments (e.g. containing cluster indices)
 * @param n_procs The number of processes
 * @param t_sizes n_procs-size array of tile widths for each process
 * @return int* Newly-allocated m-size global assignments array
 */
int64_t* compute_g_assignments(int64_t m, int64_t t, int64_t* assignments,
                               int n_procs, int* t_sizes);

/**
 * @brief Computes global cluster sizes by Allreduce from the local cluster sizes.
 * 
 * @param k The number of clusters
 * @param cluster_sizes k-size array of local cluster sizes
 * @return int* Newly-allocated k-size global cluster sizes array
 */
int64_t* compute_g_cluster_sizes(int64_t k, int64_t* cluster_sizes);

/**
 * @brief Global k-by-m size V (global) matrix in CSR (CSR is faster for VK SpMM routine)
 * Does not free g_assignments or g_cluster_sizes arrays.
 * 
 * @param gV The output gV matrix
 * @param m The number of points
 * @param k The number of clusters
 * @param g_assignments m-size array of global cluster assignments  (e.g. containing cluster indices)
 * @param g_cluster_sizes k-size array of global cluster sizes
 * @return int 
 */
int create_gV_csr(cusparseSpMatDescr_t* gV, int64_t m, int64_t k,
                  int64_t* g_assignments, int64_t* g_cluster_sizes);

/**
 * @brief Local k-by-t size V (partial) matrix in CSC (CSC is faster for c_norm initialization)
 * Does not free assignments or g_cluster_sizes arrays.
 * 
 * @param lV The output lV matrix
 * @param t The tile width (e.g. width of column of K on this process)
 * @param k The number of clusters
 * @param assignments t-size array of assignments (e.g. containing cluster indices)
 * @param g_cluster_sizes k-size array of global cluster sizes
 * @return int
 */
int create_lV_csc(cusparseSpMatDescr_t* lV, int64_t t, int64_t k,
                  int64_t* assignments, int64_t* g_cluster_sizes);

/**
 * @brief Constructs global and local V matrices based on this process's local assignments and cluster sizes.
 * 
 * See above methods for parameters.
 */
int reinit_V(cusparseSpMatDescr_t* gV, cusparseSpMatDescr_t* lV, int64_t m,
             int64_t t, int64_t k, int64_t* assignments, int64_t* cluster_sizes,
             int* t_sizes, MPI_Comm comm);

/**
 * @brief Initializes V by round robin assignment
 * 
 * See above methods for parameters.
 */
int init_V(cusparseSpMatDescr_t* gV, cusparseSpMatDescr_t* lV, int64_t m,
           int64_t t, int64_t k, int* t_sizes, MPI_Comm comm);

/**
 * @brief Multiplies a sparse matrix by a dense matrix
 *
 * @param handle The cuSPARSE handle
 * @param V The sparse matrix
 * @param K The dense matrix
 * @param ET cuSPARSE descriptor for the resulting dense matrix
 * @return int
 */
int spmm(cusparseHandle_t& handle, cusparseSpMatDescr_t& V,
         cusparseDnMatDescr_t& K, cusparseDnMatDescr_t* ET);

}  // namespace cpop

#endif  // CPOP_COMPUTE_SPARSE_HH
