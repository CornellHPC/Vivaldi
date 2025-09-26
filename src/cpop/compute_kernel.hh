#ifndef CPOP_COMPUTE_KERNEL_HH
#define CPOP_COMPUTE_KERNEL_HH

#include "cusparse.h"
#include "slate/slate.hh"
#include "utils.hh"

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
slate::Matrix<float> load_matrix2d(const char* fname, int64_t rows, int64_t cols,
                                 MPI_Comm comm);

/**
 * Extracts tiles from the kernel matrix.
 * This returns the number of elements in the tiles buffer.
 *
 * @param tiles is a pointer that will output the tile data (on device)
 * @param K is the distributed kernel matrix
 * @param col is the grid column of the distributed kernel matrix
 * @return the number of elements in the tiles.
 */
int64_t extract_kernel_tiles(float** tiles, slate::Matrix<float>& K, int col);

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
float* compute_kernel_matrix2d(Handle& handle, slate::Matrix<float>& PT, float gamma, float c, float r, bool redist);
float * redistribute_2d_1d(Handle& handle, float * K, const uint64_t m, const uint64_t mb);

}  // namespace cpop

#endif  // CPOP_COMPUTE_KERNEL_HH
