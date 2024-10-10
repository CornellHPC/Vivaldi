#ifndef DISTRIBUTED_POPCORN_SPARSE_MAT_H
#define DISTRIBUTED_POPCORN_SPARSE_MAT_H

#include <mpi.h>

#include <cstdint>

#include "../../common.hh"

namespace popcorn {

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
   * @param row_ids is a vector of row indices for each entry
   * @param col_ids is a vector of column indices for each entry
   * @param vals is a vector of values for each entry
   * @param rows is num of rows
   * @param cols is num of cols
   * @param grid_dim is an int64_t of the MPI grid length (SLATE `p`)
   * @param comm is the MPI communicator used for the matrix distribution
   */
  SparseMat(std::vector<float> &row_ids, std::vector<float> &col_ids,
            std::vector<DATA_TYPE> &vals, int64_t rows, int64_t cols,
            int64_t grid_dim, MPI_Comm comm);

  /**
   * Prints the SparseMat to the provided output stream.
   *
   * @param prefix is the prefix string
   * @param out is the output stream
   */
  void print(std::string prefix, std::ostream &out = std::cout);

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

  // Grid total size
  int64_t grid_dim;

  // MPI communicator used for distribution
  MPI_Comm comm;

  // Underlying CombBLAS matrix object
  std::unique_ptr<combblas::SpParMat<int64_t, DATA_TYPE, UDER>> cm;
};

DenseMat spmm(SparseMat &L, DenseMat &R);

} // namespace popcorn

#endif // DISTRIBUTED_POPCORN_SPARSE_MAT_H
