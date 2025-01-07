#include "../common.hh"
#include "kernel/kernel.cuh"

namespace popcorn {

/**
 * Loads a dense matrix from a file. 
 * This returns the transposed dense matrix of the provided file.
 *
 * @param fname is a path to the binary file containing the matrix data
 * @param rows is the number of rows in the dataset
 * @param cols is the number of columns in the dataset
 * @return the loaded dense matrix.
 */
slate_matrix load_data(const char* fname, int64_t rows, int64_t cols);

/**
 * Generates kernel matrix K
 *
 * @param PT transposed points matrix
 * @return the transposed dense matrix.
 */
slate_matrix compute_kernel_matrix(slate_matrix& PT);

}