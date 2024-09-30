#include "matrix.hh"

void matrix::grid_size(int mpi_size, int *p_out, int *q_out) {
  int p, q;
  for (p = int(sqrt(mpi_size)); p > 0; --p) {
    q = int(mpi_size / p);
    if (p * q == mpi_size) break;
  }
  *p_out = p;
  *q_out = q;
}

slate::Options matrix::get_slate_opts() {
#ifdef CUDA
  return {{slate::Option::Target, slate::Target::Devices}};
#else
  return {{slate::Option::Target, slate::Target::HostTask}};
#endif
}

void matrix::fill_slate_mat_with_buffer(slate::Matrix<DATA_TYPE> M,
                                        DATA_TYPE *buf) {
  int64_t m = M.m(), n = M.n(), mb = M.tileMb(0), nb = M.tileNb(0);
  for (int64_t j = 0; j < M.nt(); ++j) {    // i loops over block columns
    for (int64_t i = 0; i < M.mt(); ++i) {  // j loops over block rows
      if (M.tileIsLocal(i, j)) {
#ifdef CUDA
        slate::Tile<DATA_TYPE> tile = M.at(i, j, M.tileDevice(i, j));
#else
        slate::Tile<DATA_TYPE> tile = M.at(i, j, slate::HostNum);
#endif
        int64_t lda = tile.stride();
        DATA_TYPE *A = tile.data();

        for (int64_t jj = 0; jj < tile.nb(); ++jj) {  // jj loops over columns
          int64_t global_column = j * nb + jj;
          int64_t global_row_start = i * mb;

#ifdef CUDA
          cudaMemcpy(A + jj * lda, buf + global_column * m + global_row_start,
                     sizeof(DATA_TYPE) * tile.mb(), cudaMemcpyDeviceToDevice);
#else
          memcpy(A + jj * lda, buf + global_column * m + global_row_start,
                 sizeof(DATA_TYPE) * tile.mb());
#endif
        }
      }
    }
  }
}

void matrix::raise_slate_mat_to_power(slate::Matrix<DATA_TYPE> M,
                                      DATA_TYPE power) {
  int64_t m = M.m(), n = M.n(), mb = M.tileMb(0), nb = M.tileNb(0);
  for (int64_t j = 0; j < M.nt(); ++j) {    // i loops over block columns
    for (int64_t i = 0; i < M.mt(); ++i) {  // j loops over block rows
      if (M.tileIsLocal(i, j)) {
        slate::Tile<DATA_TYPE> tile = M(i, j);
        int64_t lda = tile.stride();
        DATA_TYPE *A = tile.data();

        for (int64_t jj = 0; jj < tile.nb(); ++jj) {    // jj loops over columns
          for (int64_t ii = 0; ii < tile.mb(); ++ii) {  // ii loops over rows
            A[ii + jj * lda] = pow(A[ii + jj * lda], power);
          }
        }
      }
    }
  }
}

void matrix::fill_slate_mat_with_scalar(slate::Matrix<DATA_TYPE> M,
                                        DATA_TYPE value) {
  int64_t m = M.m(), n = M.n(), mb = M.tileMb(0), nb = M.tileNb(0);
  DATA_TYPE *buf = (DATA_TYPE *)malloc(m * n * sizeof(DATA_TYPE));
  for (int i = 0; i < m * n; ++i) buf[i] = value;
  fill_slate_mat_with_buffer(M, buf);
  free(buf);
}

DATA_TYPE matrix::get_slate_mat_value(slate::Matrix<DATA_TYPE> M, int64_t ii,
                                      int64_t jj) {
  int64_t mt = M.mt(), nt = M.nt();
  int64_t i = ii / mt;  // Tile row index
  int64_t j = jj / nt;  // Tile column index
  ii -= i * mt;         // Element row index
  jj -= j * nt;         // Element column index

  slate::Tile<DATA_TYPE> T = M(i, j);
  DATA_TYPE v = T(ii, jj);
  return v;
}

combblas::DnParMat<int64_t, DATA_TYPE> matrix::slate_mat_to_combblas_dpm(
    slate::Matrix<DATA_TYPE> M) {
  slate::GridOrder order;
  int nprow, npcol, myrow, mycol;
  M.gridinfo(&order, &nprow, &npcol, &myrow, &mycol);

  int wholerank;
  MPI_Comm_rank(M.mpiComm(), &wholerank);

  std::shared_ptr<combblas::CommGrid> grid =
      std::make_shared<combblas::CommGrid>(M.mpiComm(), nprow, npcol);
  combblas::DnParMat<int64_t, DATA_TYPE> D(grid, M.m(), M.n(), (DATA_TYPE)0);

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

// TODO: Make this the V matrix
combblas::SpParMat<int64_t, DATA_TYPE, UDER<DATA_TYPE>>
matrix::initialize_combblas_v_matrix(combblas::DnParMat<int64_t, DATA_TYPE> &K) {
  std::shared_ptr<combblas::CommGrid> grid =
      std::make_shared<combblas::CommGrid>(MPI_COMM_WORLD, 2, 2);
  std::vector<float> lrow_ids, lcol_ids, lvals;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank % 2) {
    lrow_ids.push_back(rank);
    lcol_ids.push_back(rank);
    lvals.push_back(1);
  }

  combblas::FullyDistVec<int64_t, DATA_TYPE> drows(lrow_ids, grid);
  combblas::FullyDistVec<int64_t, DATA_TYPE> dcols(lcol_ids, grid);
  combblas::FullyDistVec<int64_t, DATA_TYPE> dvals(lvals, grid);

  combblas::SpParMat<int64_t, DATA_TYPE, UDER<DATA_TYPE>> V{
      4, 4, drows, dcols, dvals, false};

  return V;
}
