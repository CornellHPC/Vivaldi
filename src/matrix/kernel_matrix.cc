#include "kernel_matrix.hh"

void matrix::compute_b_mat(slate::Matrix<DATA_TYPE> &B,
                           slate::Matrix<DATA_TYPE> &sP) {
  slate::Matrix<DATA_TYPE> sP_transpose = slate::transpose(sP);
  slate::gemm<DATA_TYPE>(1.0f, sP, sP_transpose, 0.0f, B, get_slate_opts());
}

slate::Matrix<DATA_TYPE> matrix::slate_point_mat_to_polynomial_kernel_mat(
    slate::Matrix<DATA_TYPE> &sP, DATA_TYPE gamma, DATA_TYPE c, DATA_TYPE r) {
  slate::GridOrder order;
  int nprow, npcol, myrow, mycol;
  sP.gridinfo(&order, &nprow, &npcol, &myrow, &mycol);

  // slate::Matrix<DATA_TYPE> MT = slate::transpose(M);
  slate::Matrix<DATA_TYPE> B(sP.m(), sP.m(), sP.mt(), sP.mt(), nprow, npcol,
                             sP.mpiComm());
#ifdef CUDA
  B.insertLocalTiles(slate::Target::Devices);
#else
  B.insertLocalTiles(slate::Target::Host);
#endif
  // fill_slate_mat_with_scalar(K, c);
  // slate::gemm<DATA_TYPE>(gamma, M, MT, (DATA_TYPE)1, K, get_slate_opts());
  // raise_slate_mat_to_power<DATA_TYPE>(K, r);

  compute_b_mat(B, sP);

  // TODO: get rid of CUDA hyper-parameter, since we should always be running on
  // CUDA
  for (int64_t j = 0; j < B.nt(); ++j) {    // i loops over block columns
    for (int64_t i = 0; i < B.mt(); ++i) {  // j loops over block rows
      if (B.tileIsLocal(i, j)) {
        slate::Tile<DATA_TYPE> tile = B.at(i, j, B.tileDevice(i, j));
        DATA_TYPE *B_tile = tile.data();
        kernel::polynomial_kernel(tile.mb(), B_tile, gamma, c, r);
      }
    }
  }
  return B;
}
