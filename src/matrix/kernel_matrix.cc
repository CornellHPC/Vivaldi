#include "kernel_matrix.hh"

slate::Matrix<DATA_TYPE> matrix::slate_point_mat_to_polynomial_kernel_mat(
    slate::Matrix<DATA_TYPE> &M, DATA_TYPE gamma, DATA_TYPE c, DATA_TYPE r) {
  slate::GridOrder order;
  int nprow, npcol, myrow, mycol;
  M.gridinfo(&order, &nprow, &npcol, &myrow, &mycol);

  slate::Matrix<DATA_TYPE> MT = slate::transpose(M);
  slate::Matrix<DATA_TYPE> K(M.m(), M.m(), M.mt(), M.mt(), nprow, npcol,
                             M.mpiComm());

#ifdef CUDA
  K.insertLocalTiles(slate::Target::Devices);
#else
  K.insertLocalTiles(slate::Target::Host);
#endif
  fill_slate_mat_with_scalar(K, c);
  slate::gemm<DATA_TYPE>(gamma, M, MT, (DATA_TYPE)1, K, get_slate_opts());
  // raise_slate_mat_to_power<DATA_TYPE>(K, r);

  return K;
}
