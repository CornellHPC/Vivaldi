#ifndef DISTRIBUTED_POPCORN_MATRIX_H
#define DISTRIBUTED_POPCORN_MATRIX_H

#include <math.h>

#include "slate/slate.hh"

void grid_size(int mpi_size, int* p_out, int* q_out) {
  int p, q;
  for (p = int(sqrt(mpi_size)); p > 0; --p) {
    q = int(mpi_size / p);
    if (p * q == mpi_size) break;
  }
  *p_out = p;
  *q_out = q;
}

template <typename scalar_type>
void fill_matrix_from_buffer(slate::Matrix<scalar_type> M, scalar_type* buf) {
  int64_t m = M.m(), n = M.n(), mb = M.tileMb(0), nb = M.tileNb(0);
  for (int64_t j = 0; j < M.nt(); ++j) {    // i loops over block columns
    for (int64_t i = 0; i < M.mt(); ++i) {  // j loops over block rows
      if (M.tileIsLocal(i, j)) {
        slate::Tile<scalar_type> tile = M(i, j);
        int64_t lda = tile.stride();
        scalar_type* A = tile.data();

        for (int64_t jj = 0; jj < tile.nb(); ++jj) {    // jj loops over columns
          for (int64_t ii = 0; ii < tile.mb(); ++ii) {  // ii loops over rows
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
scalar_type* fill_matrix(slate::Matrix<scalar_type> M, scalar_type value) {
  int64_t m = M.m(), n = M.n(), mb = M.tileMb(0), nb = M.tileNb(0);
  scalar_type* buf = (scalar_type*)malloc(m * n * sizeof(scalar_type));
  for (int i = 0; i < m * n; ++i) buf[i] = value;
  fill_matrix_from_buffer(M, buf);
  return buf;
}

template <typename scalar_type>
void construct_kernel_matrix_with_gemm(slate::Matrix<scalar_type> M,
                                       slate::Matrix<scalar_type> C,
                                       scalar_type gamma, scalar_type c,
                                       scalar_type r) {
  slate::Matrix<scalar_type> MT = slate::transpose(M);
  auto buf = fill_matrix<scalar_type>(C, (scalar_type)c);
  slate::gemm<scalar_type>(gamma, M, MT, (scalar_type)1, C);
  free(buf);
}

#endif  // DISTRIBUTED_POPCORN_MATRIX_H