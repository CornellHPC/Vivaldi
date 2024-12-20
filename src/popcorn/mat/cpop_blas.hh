#ifndef DISTRIBUTED_POPCORN_MAT_BLAS_H
#define DISTRIBUTED_POPCORN_MAT_BLAS_H

// Local imports
#include "../../common.hh"

namespace popcorn {

/**
 * Factory for CombBLAS SpParMat that loads the specified data.
 *
 * Each rank should supply a specific range of rows. In
 * particular, for a sparse matrix with m rows distributed
 * amongst p processors, rank i is responsible for rows in
 * the range [i*m//p, (i+1)*m//p). Each point is formed by
 * one entry in rows, cols, and vals.
 *
 * @param row_ids is a vector of row indices for each entry
 * @param col_ids is a vector of column indices for each entry
 * @param vals is a vector of values for each entry
 * @param rows is num of rows
 * @param cols is num of cols
 * @param comm is the MPI communicator used for the matrix distribution
 */
c_sp_ptr initialize_from_coo(std::vector<float>& row_ids,
                             std::vector<float>& col_ids,
                             std::vector<DATA_TYPE>& vals, int64_t rows,
                             int64_t cols, MPI_Comm comm);

/**
 * Initializes and returns the V matrix for popcorn.
 * This does a round-robin assignment of points to clusters.
 *
 * @param points is the number of points to cluster
 * @param k is the number of clusters
 * @param comm is the MPI communicator
 * @return the V matrix
 */
c_sp_ptr initialize_v(int64_t points, int64_t k, MPI_Comm comm);

/**
 * @brief Sparse matrix-Dense matrix product, needed in ET=V*K
 * 
 * @param V matrix
 * @param K matrix
 * @return c_dn_ptr ET matrix
 */
c_dn_ptr spmm(c_sp_ptr& V, c_dn_ptr& K);

/**
 * @brief Initializes and returns the cnorm vector for popcorn.
 *
 * @param V is the sparse cluster assignment matrix
 * @param ET is the dense matrix E transposed.
 * @return std::vector<DATA_TYPE> cnorm vector
 */
std::vector<DATA_TYPE> initialize_cnorm(c_sp_ptr& V, c_dn_ptr& ET);

/**
 * @brief Computes the D matrix and stores it back again in ET
 * 
 * @param ET is the dense matrix E transposed
 * @param cnorm vector
 */
void compute_d(c_dn_ptr& ET, std::vector<DATA_TYPE> cnorm);

/**
 * @brief Reinitializes V based on the cluster distances matrix D
 * 
 * @param V is the sparse cluster assignment matrix
 * @param D is the cluster distances matrix
 * @return c_sp_ptr new V matrix
 */
c_sp_ptr reinitialize_v(c_sp_ptr& V, c_dn_ptr& D);

/**
 * Saves the cluster assignments to disk.
 * It computes the final assignment based on the distance matrix.
 *
 * @param filename is the name of the output file
 */
void save_assignments(c_sp_ptr& V, const char* filename);

}  // namespace popcorn

#endif  // DISTRIBUTED_POPCORN_MAT_BLAS_H