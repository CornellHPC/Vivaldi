#ifndef DENSE_MAT_H
#define DENSE_MAT_H

#include <cstdint>

#include <mpi.h>

#include "../common.hh"

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
   * Constructor that loads the specified file.
   *
   * @param filename is a path to the binary file containing the matrix data
   * @param m is the number of rows in the global matrix
   * @param n is the number of columns in the global matrix
   * @param comm is the MPI communicator used for the matrix distribution
   */
  DenseMat(const char *filename, int64_t m, int64_t n, MPI_Comm comm);

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
  // Number of rows in the global matrix
  int64_t rows;

  // Number of columns in the global matrix
  int64_t cols;

  // Number of block rows in the global matrix
  int64_t blockRows;

  // Number of block columns in the global matrix
  int64_t blockCols;

  // Number of rows per block in the standard block size
  int64_t rowsPerBlock;

  // Number of columns per block in the standard block size
  int64_t colsPerBlock;

  // Underlying SLATE matrix object
  slate::Matrix<DATA_TYPE> sm;

  // Underlying CombBLAS matrix object
  combblas::DnParMat<int64_t, DATA_TYPE> cm;

  // MPI communicator used for distribution
  MPI_Comm comm;

  /**
   * Constructor from a SLATE dense matrix.
   *
   * @param sm is the SLATE dense matrix
   */
  DenseMat(const slate::Matrix<DATA_TYPE> &sm);

  /**
   * Constructor from a CombBLAS dense matrix.
   *
   * @param cm is the CombBLAS dense matrix
   */
  DenseMat(const combblas::DnParMat<int64_t, DATA_TYPE> &cm);
};

#endif // DENSE_MAT_H
