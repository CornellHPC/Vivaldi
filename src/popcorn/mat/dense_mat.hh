#ifndef DISTRIBUTED_POPCORN_DENSE_MAT_H
#define DISTRIBUTED_POPCORN_DENSE_MAT_H

#include <mpi.h>

#include <cstdint>

#include "../../common.hh"

namespace popcorn {

class SparseMat;

/**
 * A distributed dense matrix.
 *
 * This class abstracts the underlying distributed dense
 * matrix backend library so the algorithm can be described
 * in a library-agnostic manner.
 */
class DenseMat {
 public:
  /**
   * Loads a DenseMat from a file.
   *
   * @param filename is a path to the binary file containing the matrix data
   * @param m is the number of rows in the global matrix
   * @param n is the number of columns in the global matrix
   * @param comm is the MPI communicator used for the matrix distribution
   */
  static DenseMat *load_from_file(const char *filename, int64_t rows,
                                  int64_t cols, int64_t rows_per_block,
                                  int64_t cols_per_block, int64_t grid_dim,
                                  MPI_Comm comm);

  /**
   * Constructor from a SLATE dense matrix.
   *
   * @param sm is the SLATE dense matrix
   */
  static DenseMat *from_slate(slate::Matrix<DATA_TYPE> *sm,
                              int64_t rows_per_block, int64_t cols_per_block,
                              int64_t grid_size);

  /**
   * Constructor from a CombBLAS dense matrix.
   *
   * @param cm is the CombBLAS dense matrix
   */
  static DenseMat *from_combblas(
      const combblas::DnParMat<int64_t, DATA_TYPE> &cm);

  /**
   * Returns a new DenseMat that is transposed.
   *
   * @return The transposed DenseMat.
   */
  DenseMat transpose();

  /**
   * Returns a new DenseMat formed by applying provied function element-wise.
   *
   * @param f is the function to apply
   */
  DenseMat apply(DATA_TYPE f(DATA_TYPE));

  /**
   * Prints the DenseMat to the provided output stream.
   *
   * @param out is the output stream
   */
  void print(std::ostream &out = std::cout);

  /**
   * Performs a GEMM and returns a new DenseMat with the result.
   *
   * @param L is the left matrix
   * @param R is the right matrix
   */
  friend DenseMat gemm(DenseMat &L, DenseMat &R);

  /**
   * Performs a SPMM and returns a new DenseMat with the result.
   *
   * @param L is the left sparse matrix
   * @param R is the right dense matrix
   */
  friend DenseMat spmm(SparseMat &L, DenseMat &R);

 private:
  /**
   * Constructor
   */
  DenseMat(int64_t rows, int64_t cols, int64_t block_rows, int64_t block_cols,
           int64_t rows_per_block, int64_t cols_per_block, int64_t grid_dim,
           MPI_Comm comm, slate::Matrix<DATA_TYPE> *sm = NULL,
           combblas::DnParMat<int64_t, DATA_TYPE> *cm = NULL)
      : rows(rows),
        cols(cols),
        block_rows(block_rows),
        block_cols(block_cols),
        rows_per_block(rows_per_block),
        cols_per_block(cols_per_block),
        grid_dim(grid_dim),
        sm(sm),
        cm(cm),
        comm(comm) {};

  // Number of rows in the global matrix
  int64_t rows;

  // Number of columns in the global matrix
  int64_t cols;

  // Number of block rows in the global matrix
  int64_t block_rows;

  // Number of block columns in the global matrix
  int64_t block_cols;

  // Number of rows per block in the standard block size
  int64_t rows_per_block;

  // Number of columns per block in the standard block size
  int64_t cols_per_block;

  // Grid total size
  int64_t grid_dim;

  // MPI communicator used for distribution
  MPI_Comm comm;

  // Underlying SLATE matrix object
  slate::Matrix<DATA_TYPE> *sm;

  // Underlying CombBLAS matrix object
  combblas::DnParMat<int64_t, DATA_TYPE> *cm;
};

}  // namespace popcorn

#endif  // DISTRIBUTED_POPCORN_DENSE_MAT_H
