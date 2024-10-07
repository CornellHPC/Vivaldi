#include "kernel_matrix.hh"
#include "../util.hh"

slate::Matrix<DATA_TYPE> matrix::compute_b_mat(slate::Matrix<DATA_TYPE> &sP) {
  MPI_Comm comm = sP.mpiComm();
  int p = square_grid_dim(comm);
  int m = sP.m();
  slate::Matrix<DATA_TYPE> B(m, m, tile_dim(comm, m), tile_dim(comm, m), p, p,
                             comm);
#ifdef CUDA
  B.insertLocalTiles(slate::Target::Devices);
#else
  B.insertLocalTiles(slate::Target::Host);
#endif

  slate::Matrix<DATA_TYPE> sP_transpose = slate::transpose(sP);
  slate::gemm<DATA_TYPE>(1.0f, sP, sP_transpose, 0.0f, B, get_slate_opts());
  return B;
}

slate::Matrix<DATA_TYPE> matrix::slate_point_mat_to_polynomial_kernel_mat(
    slate::Matrix<DATA_TYPE> &sP, DATA_TYPE gamma, DATA_TYPE c, DATA_TYPE r) {
  auto B = compute_b_mat(sP);

  // TODO: get rid of CUDA hyper-parameter, since we should always be running on
  // CUDA
  for (int64_t j = 0; j < B.nt(); ++j) {   // i loops over block columns
    for (int64_t i = 0; i < B.mt(); ++i) { // j loops over block rows
      if (B.tileIsLocal(i, j)) {
        slate::Tile<DATA_TYPE> tile = B.at(i, j, B.tileDevice(i, j));
        DATA_TYPE *B_tile = tile.data();
        kernel::polynomial_kernel(tile.mb(), B_tile, gamma, c, r);
      }
    }
  }
  return B;
}
