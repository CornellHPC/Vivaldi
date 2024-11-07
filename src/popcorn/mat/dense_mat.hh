#ifndef DISTRIBUTED_POPCORN_DENSE_MAT_H
#define DISTRIBUTED_POPCORN_DENSE_MAT_H

// C++ standard imports
#include <cstdint>
#include <memory>

// Library imports
#include <mpi.h>

// Local imports
#include "../../common.hh"
#include "../kernel/kernel.cuh"

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
   * @param rows is the number of rows in the global matrix
   * @param cols is the number of columns in the global matrix
   * @param comm is the MPI communicator used for the matrix distribution
   * @return the loaded dense matrix.
   */
  static DenseMat load_from_file(const char* filename, int64_t rows,
                                 int64_t cols, MPI_Comm comm);

  /**
   * Initializes and returns the cnorm vector for popcorn.
   *
   * @param V is the sparse cluster assignment matrix.
   * @param ET is the dense matrix E transposed.
   */
  static std::vector<DATA_TYPE> initialize_cnorm(SparseMat& V, DenseMat& ET);

  /**
   * Returns a new DenseMat that is transposed.
   *
   * @return The transposed DenseMat.
   */
  DenseMat transpose();

  /**
   * Applyies the provided kernel to each block of the matrix
   *
   * @param f is the function to apply
   */
  void apply(Kernel& k);

  /**
   * Performs a GEMM and returns a new DenseMat with the result.
   *
   * @param R is the right matrix
   */
  DenseMat gemm(DenseMat& R);

  /**
   * @brief Gets underlying data representation
   * 
   * @return DATA_TYPE*
   */
  DATA_TYPE* data();

  /**
   * Prints the SparseMat to the provided output stream.
   *
   * @param prefix is the prefix string
   * @param out is the output stream
   */
  void print(std::string prefix, std::ostream& out = std::cout);

  friend class SparseMat;

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

 private:
  /**
   * Constructor from a SLATE dense matrix.
   *
   * @param sm is the SLATE dense matrix
   */
  DenseMat(std::unique_ptr<slate::Matrix<DATA_TYPE>> sm);

  /**
   * Constructor from a CombBLAS dense matrix.
   *
   * @param cm is the CombBLAS dense matrix
   * @param comm is the MPI communicator
   */
  DenseMat(std::unique_ptr<combblas::DnParMat<int64_t, DATA_TYPE>> cm,
           MPI_Comm comm);

  /**
   * Converts the SLATE representation to CombBLAS.
   * This routine assumes there exists a 1-to-1 mapping
   * between SLATE tiles and CombBLAS tiles, which may
   * require point culling. It also assumes the matrix
   * is symmetric.
   */
  void to_combblas();

  /**
   * Converts the CombBLAS representation to SLATE.
   * This routine assumes there exists a 1-to-1 mapping
   * between CombBLAS tiles and SLATE tiles, which may
   * require point culling. It also assumes the matrix
   * is symmetric.
   */
  void to_slate();

  // MPI communicator used for distribution
  MPI_Comm comm;

  // Underlying SLATE matrix object
  std::unique_ptr<slate::Matrix<DATA_TYPE>> sm;

  // Underlying CombBLAS matrix object
  std::unique_ptr<combblas::DnParMat<int64_t, DATA_TYPE>> cm;
};

}  // namespace popcorn

#endif  // DISTRIBUTED_POPCORN_DENSE_MAT_H
