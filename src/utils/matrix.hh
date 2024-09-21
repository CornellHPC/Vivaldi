#ifndef DISTRIBUTED_POPCORN_MATRIX_H
#define DISTRIBUTED_POPCORN_MATRIX_H

#include <cstdint>
#include <math.h>
#include <memory>

#include "CombBLAS/CombBLAS.h"
#undef Error
#include "slate/slate.hh"

void grid_size(int mpi_size, int *p_out, int *q_out) {
  int p, q;
  for (p = int(sqrt(mpi_size)); p > 0; --p) {
    q = int(mpi_size / p);
    if (p * q == mpi_size)
      break;
  }
  *p_out = p;
  *q_out = q;
}

template <typename scalar_type>
void fill_mat_with_buffer(slate::Matrix<scalar_type> M, scalar_type *buf) {
  int64_t m = M.m(), n = M.n(), mb = M.tileMb(0), nb = M.tileNb(0);
  for (int64_t j = 0; j < M.nt(); ++j) {   // i loops over block columns
    for (int64_t i = 0; i < M.mt(); ++i) { // j loops over block rows
      if (M.tileIsLocal(i, j)) {
        slate::Tile<scalar_type> tile = M(i, j);
        int64_t lda = tile.stride();
        scalar_type *A = tile.data();

        for (int64_t jj = 0; jj < tile.nb(); ++jj) {   // jj loops over columns
          for (int64_t ii = 0; ii < tile.mb(); ++ii) { // ii loops over rows
            int64_t global_row = i * mb + ii;
            int64_t global_column = j * nb + jj;
            A[ii + jj * lda] = buf[global_row + global_column * m];
          }
        }
      }
    }
  }
}

template <typename scalar_type>
void fill_mat_with_scalar(slate::Matrix<scalar_type> M, scalar_type value) {
  int64_t m = M.m(), n = M.n(), mb = M.tileMb(0), nb = M.tileNb(0);
  scalar_type *buf = (scalar_type *)malloc(m * n * sizeof(scalar_type));
  for (int i = 0; i < m * n; ++i)
    buf[i] = value;
  fill_mat_with_buffer(M, buf);
  free(buf);
}

template <typename scalar_type>
slate::Matrix<scalar_type>
point_mat_to_polynomial_kernel_mat(slate::Matrix<scalar_type> M,
                                   scalar_type gamma, scalar_type c,
                                   scalar_type r) {
  slate::GridOrder order;
  int nprow, npcol, myrow, mycol;
  M.gridinfo(&order, &nprow, &npcol, &myrow, &mycol);

  slate::Matrix<scalar_type> MT = slate::transpose(M);
  slate::Matrix<scalar_type> C(M.m(), M.m(), M.mt(), M.mt(), nprow, npcol,
                               M.mpiComm());

  C.insertLocalTiles();
  fill_mat_with_scalar<scalar_type>(C, (scalar_type)c);
  slate::gemm<scalar_type>(gamma, M, MT, (scalar_type)1, C);

  return C;
}

template <typename scalar_type>
combblas::DnParMat<int64_t, scalar_type>
slate_mat_to_combblas_dpm(slate::Matrix<scalar_type> M) {
  slate::GridOrder order;
  int nprow, npcol, myrow, mycol;
  M.gridinfo(&order, &nprow, &npcol, &myrow, &mycol);

  std::shared_ptr<combblas::CommGrid> grid =
      std::make_shared<combblas::CommGrid>(M.mpiComm(), nprow, npcol);
  combblas::DnParMat<int64_t, scalar_type> A(grid, M.m(), M.n(),
                                             (scalar_type)0);

  // TODO: Copy data
  return A;
}

#endif // DISTRIBUTED_POPCORN_MATRIX_H
