#ifndef DISTRIBUTED_POPCORN_MAT_SLATE_H
#define DISTRIBUTED_POPCORN_MAT_SLATE_H

// Local imports
#include "../../common.hh"
#include "../kernel/kernel.cuh"

namespace popcorn {

/**
 * Loads a dense matrix from a file. 
 * This returns the transposed dense matrix of the provided file.
 *
 * @param filename is a path to the binary file containing the matrix data
 * @param rows is the number of rows in the global matrix
 * @param cols is the number of columns in the global matrix
 * @param comm is the MPI communicator used for the matrix distribution
 * @return the loaded dense matrix.
 */
sm_ptr load_from_file(const char* filename, int64_t rows, int64_t cols, MPI_Comm comm);

/**
 * Generates kernel matrix K
 *
 * @param PT transposed points matrix
 * @param kernel_func kernel function to use (e.g. PolynomialKernel)
 * @return The transposed dense matrix.
 */
sm_ptr compute_k(sm_ptr& PT, Kernel& kernel_func);

/**
 * @brief Converts a SLATE matrix to a CombBLAS matrix.
 * 
 * @param K SLATE matrix, will be freed by this method
 * @return c_dn_ptr CombBLAS matrix
 */
c_dn_ptr to_combblas(sm_ptr& K);

}  // namespace popcorn

#endif  // DISTRIBUTED_POPCORN_MAT_SLATE_H