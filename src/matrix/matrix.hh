#ifndef DISTRIBUTED_POPCORN_MATRIX_H
#define DISTRIBUTED_POPCORN_MATRIX_H

#include <math.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../common.hh"
#include "../util.hh"

// Define CombBLAS sparse matrix format
template <typename UV> using UDER = combblas::SpCCols<int64_t, UV>;

namespace matrix {

slate::Options get_slate_opts() {
#ifdef CUDA
  return {{slate::Option::Target, slate::Target::Devices}};
#else
  return {{slate::Option::Target, slate::Target::HostTask}};
#endif
}

template <typename T>
void fill_slate_mat_with_buffer(slate::Matrix<T> &M, T *buf) {
  int64_t m = M.m(), n = M.n(), mb = M.tileMb(0), nb = M.tileNb(0);
  for (int64_t j = 0; j < M.nt(); ++j) {   // i loops over block columns
    for (int64_t i = 0; i < M.mt(); ++i) { // j loops over block rows
      if (M.tileIsLocal(i, j)) {
#ifdef CUDA
        slate::Tile<T> tile = M.at(i, j, M.tileDevice(i, j));
#else
        slate::Tile<T> tile = M.at(i, j, slate::HostNum);
#endif
        int64_t lda = tile.stride();
        T *A = tile.data();

        for (int64_t jj = 0; jj < tile.nb(); ++jj) { // jj loops over columns
          int64_t global_column = j * nb + jj;
          int64_t global_row_start = i * mb;

#ifdef CUDA
          cudaMemcpy(A + jj * lda, buf + global_column * m + global_row_start,
                     sizeof(T) * tile.mb(), cudaMemcpyHostToDevice);
#else
          memcpy(A + jj * lda, buf + global_column * m + global_row_start,
                 sizeof(T) * tile.mb());
#endif
        }
      }
    }
  }
}

template <typename T>
T get_slate_mat_value(slate::Matrix<T> &M, int64_t ii, int64_t jj) {
  MPI_Comm comm = M.mpiComm();
  int mb = tile_dim(comm, M.m());
  int nb = tile_dim(comm, M.n());

  int64_t i = ii / mb; // Tile row index
  int64_t j = jj / nb; // Tile column index
  ii -= i * mb;        // Element row index
  jj -= j * nb;        // Element column index

  slate::Tile<T> tile = M(i, j);
  return tile(ii, jj);
}

template <typename T>
combblas::DnParMat<int64_t, T> slate_mat_to_combblas_dpm(slate::Matrix<T> &M) {
  MPI_Comm comm = M.mpiComm();
  int rank = M.mpiRank();

  slate::GridOrder order;
  int nprow, npcol, myrow, mycol;
  M.gridinfo(&order, &nprow, &npcol, &myrow, &mycol);

  std::shared_ptr<combblas::CommGrid> grid =
      std::make_shared<combblas::CommGrid>(comm, nprow, npcol);
  combblas::DnParMat<int64_t, T> D(grid, M.m(), M.n(), (T)0);

  int rowrank = grid->GetRankInProcRow(rank);
  int colrank = grid->GetRankInProcCol(rank);

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

template <typename T>
combblas::SpParMat<int64_t, T, UDER<T>>
initialize_combblas_v_matrix(int m, int k, MPI_Comm comm) {
  int rank, p, P;
  MPI_Comm_rank(comm, &rank);
  p = square_grid_dim(comm);
  P = p * p;

  std::shared_ptr<combblas::CommGrid> grid =
      std::make_shared<combblas::CommGrid>(comm, p, p);

  int *points_per_cluster = (int *)calloc(sizeof(int), k);
  for (int i = 0; i < k; ++i) {
    points_per_cluster[i] = (m / k) + ((i < m % k) ? 1 : 0);
  }

  int *points_per_process = (int *)calloc(sizeof(int), P);
  for (int i = 0; i < P; ++i) {
    points_per_process[i] = (m / P) + ((i < m % P) ? 1 : 0);
  }

  int row_start = 0;
  for (int i = 0; i < rank; ++i) {
    row_start += points_per_process[i];
  }
  int row_end = row_start + points_per_process[rank];

  std::vector<float> lrow_ids, lcol_ids, lvals;
  for (int row = row_start, col = row_start % k; row < row_end;
       ++row, col = (col + 1) % k) {
    lrow_ids.push_back(row);
    lcol_ids.push_back(col);
    lvals.push_back(1.0f / points_per_cluster[col]);
  }

  combblas::FullyDistVec<int64_t, T> drows(lrow_ids, grid);
  combblas::FullyDistVec<int64_t, T> dcols(lcol_ids, grid);
  combblas::FullyDistVec<int64_t, T> dvals(lvals, grid);

  combblas::SpParMat<int64_t, T, UDER<T>> V{m, k, drows, dcols, dvals, false};
  V.Transpose();

  free(points_per_cluster);
  free(points_per_process);
  return V;
}

} // namespace matrix

#endif // DISTRIBUTED_POPCORN_MATRIX_H
