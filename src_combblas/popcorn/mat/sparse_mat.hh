#ifndef DISTRIBUTED_POPCORN_SPARSE_MAT_H
#define DISTRIBUTED_POPCORN_SPARSE_MAT_H

// C++ standard imports
#include <cstdint>

// Library imports
#include <mpi.h>

// Local imports
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
   * @param comm is the MPI communicator used for the matrix distribution
   */
  SparseMat(std::vector<float>& row_ids, std::vector<float>& col_ids,
            std::vector<DATA_TYPE>& vals, int64_t rows, int64_t cols,
            MPI_Comm comm);

  /**
   * Initializes and returns the V matrix for popcorn.
   * This does a round-robin assignment of points to clusters.
   *
   * @param points is the number of points to cluster
   * @param k is the number of clusters
   * @param comm is the MPI communicator
   */
  static SparseMat initialize_v(int64_t points, int64_t k, MPI_Comm comm);

  /**
   * Initializes the updated V matrix for popcorn.
   * This does assignment based on the distance matrix.
   *
   * @param D is the distance matrix data transposed and flattened.
   */
  SparseMat initialize_v(float* D);

  /**
   * Saves the cluster assignments to disk.
   * It computes the final assignment based on the distance matrix.
   *
   * @param filename is the name of the output file
   */
  void save_assignments(const char* filename);

  /**
   * Transposes the sparse matrix in-place.
   */
  void transpose();

  /**
   * @param R is the right dense matrix
   */
  DenseMat spmm(DenseMat& R, double* cuda_memcpy_elapsed, double* non_memcpy_elapsed, double* malloc_elapsed);

  /**
   * Prints the SparseMat to the provided output stream.
   *
   * @param prefix is the prefix string
   * @param out is the output stream
   */
  void print(std::string prefix, std::ostream& out = std::cout);

  friend class DenseMat;

 private:
  /**
   * Constructor from a CombBLAS dense matrix.
   *
   * @param cm is the CombBLAS dense matrix
   */
  SparseMat(std::unique_ptr<combblas::SpParMat<int64_t, DATA_TYPE, UDER>> cm);

  // Number of rows in the global matrix
  int64_t rows;

  // Number of columns in the global matrix
  int64_t cols;

  // Number of nonzeros in the global matrix
  int64_t nonzeros;

  // Grid total size
  int64_t grid_dim;

  // The communicator grid used for distribution
  std::shared_ptr<combblas::CommGrid> grid;

  // Underlying CombBLAS matrix object
  std::unique_ptr<combblas::SpParMat<int64_t, DATA_TYPE, UDER>> cm;
};

DenseMat spmm(SparseMat& L, DenseMat& R);

}  // namespace popcorn

#endif  // DISTRIBUTED_POPCORN_SPARSE_MAT_H
