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
 * @return the transposed dense slate matrix.
 */
cusparseDnMatDescr_t compute_kernel_matrix(slate::Matrix<float>& PT);

}  // namespace cpop

#endif  // CPOP_COMPUTE_KERNEL_HH
