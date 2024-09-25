#ifndef DISTRIBUTED_POPCORN_MATRIX_H
#define DISTRIBUTED_POPCORN_MATRIX_H

#include <math.h>

#include <cstdint>
#include <memory>

#include "CombBLAS/CombBLAS.h"
#undef Error
#include "slate/slate.hh"

void grid_size(int mpi_size, int *p_out, int *q_out) {
  int p, q;
  for (p = int(sqrt(mpi_size)); p > 0; --p) {
    q = int(mpi_size / p);
    if (p * q == mpi_size) break;
  }
  *p_out = p;
  *q_out = q;
}

template <typename scalar_type>
void fill_slate_mat_with_buffer(slate::Matrix<scalar_type> M,
                                scalar_type *buf) {
  int64_t m = M.m(), n = M.n(), mb = M.tileMb(0), nb = M.tileNb(0);
  for (int64_t j = 0; j < M.nt(); ++j) {    // i loops over block columns
    for (int64_t i = 0; i < M.mt(); ++i) {  // j loops over block rows
      if (M.tileIsLocal(i, j)) {
        slate::Tile<scalar_type> tile = M(i, j);
        int64_t lda = tile.stride();
        scalar_type *A = tile.data();

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
void raise_slate_mat_to_power(slate::Matrix<scalar_type> M, scalar_type power) {
  int64_t m = M.m(), n = M.n(), mb = M.tileMb(0), nb = M.tileNb(0);
  for (int64_t j = 0; j < M.nt(); ++j) {    // i loops over block columns
    for (int64_t i = 0; i < M.mt(); ++i) {  // j loops over block rows
      if (M.tileIsLocal(i, j)) {
        slate::Tile<scalar_type> tile = M(i, j);
        int64_t lda = tile.stride();
        scalar_type *A = tile.data();

        for (int64_t jj = 0; jj < tile.nb(); ++jj) {    // jj loops over columns
          for (int64_t ii = 0; ii < tile.mb(); ++ii) {  // ii loops over rows
            A[ii + jj * lda] = pow(A[ii + jj * lda], power);
          }
        }
      }
    }
  }
}

template <typename scalar_type>
void fill_slate_mat_with_scalar(slate::Matrix<scalar_type> M,
                                scalar_type value) {
  int64_t m = M.m(), n = M.n(), mb = M.tileMb(0), nb = M.tileNb(0);
  scalar_type *buf = (scalar_type *)malloc(m * n * sizeof(scalar_type));
  for (int i = 0; i < m * n; ++i) buf[i] = value;
  fill_slate_mat_with_buffer(M, buf);
  free(buf);
}

template <typename scalar_type>
scalar_type get_slate_mat_value(slate::Matrix<scalar_type> M, int64_t ii,
                                int64_t jj) {
  int64_t mt = M.mt(), nt = M.nt();
  int64_t i = ii / mt;  // Tile row index
  int64_t j = jj / nt;  // Tile column index
  ii -= i * mt;         // Element row index
  jj -= j * nt;         // Element column index

  slate::Tile<scalar_type> T = M(i, j);
  scalar_type v = T(ii, jj);
  return v;
}

template <typename scalar_type>
combblas::DnParMat<int64_t, scalar_type> slate_mat_to_combblas_dpm(
    slate::Matrix<scalar_type> M) {
  slate::GridOrder order;
  int nprow, npcol, myrow, mycol;
  M.gridinfo(&order, &nprow, &npcol, &myrow, &mycol);

  int wholerank;
  MPI_Comm_rank(M.mpiComm(), &wholerank);

  std::shared_ptr<combblas::CommGrid> grid =
      std::make_shared<combblas::CommGrid>(M.mpiComm(), nprow, npcol);
  combblas::DnParMat<int64_t, scalar_type> D(grid, M.m(), M.n(),
                                             (scalar_type)0);

  int rowrank = grid->GetRankInProcRow(wholerank);
  int colrank = grid->GetRankInProcCol(wholerank);

  int64_t rows_per_proc = D.getgnrow() / grid->GetGridRows();
  int64_t ii_s = rowrank * rows_per_proc;
  int64_t ii_e = ii_s + D.getnrow();

  int64_t cols_per_proc = D.getgncol() / grid->GetGridCols();
  int64_t jj_s = colrank * cols_per_proc;
  int64_t jj_e = jj_s + D.getncol();

  int x = 0;
  for (int ii = ii_s; ii < ii_e; ++ii) {
    for (int jj = jj_s; jj < jj_e; ++jj) {
      D.setarr(x++, get_slate_mat_value(M, ii, jj));
    }
  }

  return D;
}

#endif  // DISTRIBUTED_POPCORN_MATRIX_H
