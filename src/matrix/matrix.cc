#include "matrix.hh"

void matrix::grid_size(int mpi_size, int *p_out, int *q_out) {
  int p, q;
  for (p = int(sqrt(mpi_size)); p > 0; --p) {
    q = int(mpi_size / p);
    if (p * q == mpi_size)
      break;
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

void matrix::fill_slate_mat_with_buffer(slate::Matrix<DATA_TYPE> &M,
                                        DATA_TYPE *buf) {
  int64_t m = M.m(), n = M.n(), mb = M.tileMb(0), nb = M.tileNb(0);
  for (int64_t j = 0; j < M.nt(); ++j) {   // i loops over block columns
    for (int64_t i = 0; i < M.mt(); ++i) { // j loops over block rows
      if (M.tileIsLocal(i, j)) {
#ifdef CUDA
        slate::Tile<DATA_TYPE> tile = M.at(i, j, M.tileDevice(i, j));
#else
        slate::Tile<DATA_TYPE> tile = M.at(i, j, slate::HostNum);
#endif
        int64_t lda = tile.stride();
        DATA_TYPE *A = tile.data();

        for (int64_t jj = 0; jj < tile.nb(); ++jj) { // jj loops over columns
          int64_t global_column = j * nb + jj;
          int64_t global_row_start = i * mb;

#ifdef CUDA
          cudaMemcpy(A + jj * lda, buf + global_column * m + global_row_start,
                     sizeof(DATA_TYPE) * tile.mb(), cudaMemcpyHostToDevice);
#else
          memcpy(A + jj * lda, buf + global_column * m + global_row_start,
                 sizeof(DATA_TYPE) * tile.mb());
#endif
        }
      }
    }
  }
}

DATA_TYPE matrix::get_slate_mat_value(slate::Matrix<DATA_TYPE> &M, int64_t ii,
                                      int64_t jj) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::cout << "Rank " << rank << " getting index (" << ii << ", " << jj << ")" << std::endl;

  int64_t mt = M.mt(), nt = M.nt();
  int64_t i = ii / mt; // Tile row index
  int64_t j = jj / nt; // Tile column index
  ii -= i * mt;        // Element row index
  jj -= j * nt;        // Element column index

  slate::Tile<DATA_TYPE> T = M(i, j);
  DATA_TYPE v = T(ii, jj);

  std::cout << "Rank " << rank << " finished index (" << ii << ", " << jj << ")" << std::endl;
  return v;
}

combblas::DnParMat<int64_t, DATA_TYPE>
matrix::slate_mat_to_combblas_dpm(slate::Matrix<DATA_TYPE> &M) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);




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

  std::cout << "Rank " << rank << " made it to line 103" << std::endl;

  int x = 0;
  for (int ii = ii_s; ii < ii_e; ++ii) {
    for (int jj = jj_s; jj < jj_e; ++jj) {
      D.setarr(x++, get_slate_mat_value(M, ii, jj));
    }
  }

  return D;
}

combblas::SpParMat<int64_t, DATA_TYPE, UDER<DATA_TYPE>>
matrix::initialize_combblas_v_matrix(const int m, const int k) {
  // TODO: Dynamic process grid construction
  std::shared_ptr<combblas::CommGrid> grid =
      std::make_shared<combblas::CommGrid>(MPI_COMM_WORLD, 2, 2);

  // TODO: Dynamic process count
  const int P = 2 * 2;

  int *points_per_cluster = (int *)calloc(sizeof(int), k);
  for (int i = 0; i < k; ++i) {
    points_per_cluster[i] = (m / k) + ((i < m % k) ? 1 : 0);
  }

  int *points_per_process = (int *)calloc(sizeof(int), P);
  for (int i = 0; i < P; ++i) {
    points_per_process[i] = (m / P) + ((i < m % P) ? 1 : 0);
  }

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

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

  combblas::FullyDistVec<int64_t, DATA_TYPE> drows(lrow_ids, grid);
  combblas::FullyDistVec<int64_t, DATA_TYPE> dcols(lcol_ids, grid);
  combblas::FullyDistVec<int64_t, DATA_TYPE> dvals(lvals, grid);

  combblas::SpParMat<int64_t, DATA_TYPE, UDER<DATA_TYPE>> V{
      m, k, drows, dcols, dvals, false};
  V.Transpose();

  free(points_per_cluster);
  free(points_per_process);
  return V;
}
