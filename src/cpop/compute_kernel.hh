#ifndef CPOP_COMPUTE_KERNEL_HH
#define CPOP_COMPUTE_KERNEL_HH

#include "cusparse.h"
#include "slate/slate.hh"

namespace cpop {

/**
 * Loads a dense matrix from a file. 
 * This returns the transposed dense matrix of the provided file.
 *
 * @param fname is a path to the binary file containing the matrix data
 * @param rows is the number of rows in the dataset
 * @param cols is the number of columns in the dataset
 * @param comm is the MPI communicator used to distribute the matrix
 * @return the loaded dense matrix.
 */
slate::Matrix<float> load_matrix(const char* fname, int64_t rows, int64_t cols,
                                 MPI_Comm comm);

/**
 * Generates kernel matrix K
 *
 * @param PT transposed points matrix
 * @param gamma is the gamma parameter for the polynomial kernel
 * @param r is the r parameter for the polynomial kernel
 * @param c is the c parameter for the polynomial kernel
 * @return the transposed dense slate matrix on the device
 */
float* compute_kernel_matrix(slate::Matrix<float>& PT, float gamma, float c,
                             float r);

}  // namespace cpop

#endif  // CPOP_COMPUTE_KERNEL_HH
