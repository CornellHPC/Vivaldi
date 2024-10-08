#ifndef SPARSE_MAT_H
#define SPARSE_MAT_H

#include <cstdint>

#include <mpi.h>

#include "../common.hh"

template <typename UV> using UDER = combblas::SpCCols<int64_t, UV>;

class DenseMat;

/**
 * A distributed sparse matrix.
 *
 * This class abstracts the underlying distributed sparse
 * matrix backend library so the algorithm can be described
 * in a library-agnostic manner.
 */
class SparseMat {
public:
  /**
   * Constructor that loads the specified data.
   *
   * Each rank should supply a specific range of rows. In
   * particular, for a sparse matrix with m rows distributed
   * amongst p processors, rank i is responsible for rows in
   * the range [i*m//p, (i+1)*m//p). Each point is formed by
   * one entry in rows, cols, and vals.
   *
   * @param rows is a vector of row indices for each entry
   * @param cols is a vector of column indices for each entry
   * @param vals is a vector of values for each entry
   * @param comm is the MPI communicator used for the matrix distribution
   */
  SparseMat(std::vector<int64_t> &rows, std::vector<int64_t> &cols,
            std::vector<DATA_TYPE> &vals, MPI_Comm comm);

  /**
   * Prints the SparseMat to the provided output stream.
   *
   * @param out is the output stream
   */
  void print(std::ostream &out = std::cout);

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

  // Number of nonzeros in the global matrix
  int64_t nonzeros;

  // Underlying CombBLAS matrix object
  combblas::SpParMat<int64_t, DATA_TYPE, UDER<DATA_TYPE>> cm;

  // MPI communicator used for distribution
  MPI_Comm comm;
};

#endif // SPARSE_MAT_H
