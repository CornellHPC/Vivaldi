#ifndef DISTRIBUTED_POPCORN_KERNEL_MATRIX_H
#define DISTRIBUTED_POPCORN_KERNEL_MATRIX_H

#include "../common.hh"
#include "../kernel/linear_kernel.cuh"
#include "matrix.hh"

namespace matrix {

/**
 * @brief Computes the B matrix from the sP points matrix. Used in the kernel
 * functions.
 *
 * @param sP matrix of size k by d
 * @return allocated matrix, with size k by k
 */
template <typename T> slate::Matrix<T> compute_b_mat(slate::Matrix<T> &sP) {
  MPI_Comm comm = sP.mpiComm();
  int p = square_grid_dim(comm);
  int m = sP.m();
  slate::Matrix<T> B(m, m, tile_dim(comm, m), tile_dim(comm, m), p, p, comm);
#ifdef CUDA
  B.insertLocalTiles(slate::Target::Devices);
#else
  B.insertLocalTiles(slate::Target::Host);
#endif

  slate::Matrix<T> sP_transpose = slate::transpose(sP);
  slate::gemm<T>(1.0f, sP, sP_transpose, 0.0f, B, get_slate_opts());
  return B;
}

template <typename T>
slate::Matrix<T> slate_point_mat_to_polynomial_kernel_mat(slate::Matrix<T> &sP,
                                                          T gamma, T c, T r) {
  auto B = compute_b_mat(sP);

  // TODO: get rid of CUDA hyper-parameter, since we should always be running on
  // CUDA
  for (int64_t j = 0; j < B.nt(); ++j) {   // i loops over block columns
    for (int64_t i = 0; i < B.mt(); ++i) { // j loops over block rows
      if (B.tileIsLocal(i, j)) {
        slate::Tile<T> tile = B.at(i, j, B.tileDevice(i, j));
        T *B_tile = tile.data();
        kernel::polynomial_kernel(tile.mb(), B_tile, gamma, c, r);
      }
    }
  }
  return B;
}

} // namespace matrix

#endif // DISTRIBUTED_POPCORN_KERNEL_MATRIX_H
