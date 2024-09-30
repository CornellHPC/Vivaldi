#ifndef DISTRIBUTED_POPCORN_KERNEL_MATRIX_H
#define DISTRIBUTED_POPCORN_KERNEL_MATRIX_H

#include "CombBLAS/CombBLAS.h"
#undef Error
#include "../utils/matrix.hh"
#include "slate/slate.hh"

template <typename scalar_type>
slate::Matrix<scalar_type>
slate_point_mat_to_polynomial_kernel_mat(slate::Matrix<scalar_type> M,
                                         scalar_type gamma, scalar_type c,
                                         scalar_type r) {
  slate::GridOrder order;
  int nprow, npcol, myrow, mycol;
  M.gridinfo(&order, &nprow, &npcol, &myrow, &mycol);

  slate::Matrix<scalar_type> MT = slate::transpose(M);
  slate::Matrix<scalar_type> K(M.m(), M.m(), M.mt(), M.mt(), nprow, npcol,
                               M.mpiComm());

  #ifdef CUDA
  K.insertLocalTiles(slate::Target::Devices);
#else
  K.insertLocalTiles(slate::Target::Host);
#endif
  fill_slate_mat_with_scalar(K, c);
  slate::gemm<scalar_type>(gamma, M, MT, (scalar_type)1, K, get_slate_opts());
  // raise_slate_mat_to_power<scalar_type>(K, r);

  return K;
}

#endif // DISTRIBUTED_POPCORN_KERNEL_MATRIX_H
