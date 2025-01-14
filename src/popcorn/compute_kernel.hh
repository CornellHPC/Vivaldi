#include "cusparse_v2.h"
#include "slate/slate.hh"

#include "utils.hh"

namespace popcorn {

/**
 * Loads a dense matrix from a file. 
 * This returns the transposed dense matrix of the provided file.
 *
 * @param fname is a path to the binary file containing the matrix data
 * @param rows is the number of rows in the dataset
 * @param cols is the number of columns in the dataset
 * @param rank current mpi rank
 * @param size number of mpi processes
 * @return the loaded dense matrix.
 */
slate::Matrix<float> load_matrix(const char* fname, int64_t rows, int64_t cols,
                                 int rank, int size);

/**
 * Generates kernel matrix K
 *
 * @param PT transposed points matrix
 * @param rank current mpi rank
 * @param size number of mpi processes
 * @return the transposed dense slate matrix.
 */
cusparseDnMatDescr_t compute_kernel_matrix(slate::Matrix<float>& PT, int rank,
                                           int size);

}  // namespace popcorn
